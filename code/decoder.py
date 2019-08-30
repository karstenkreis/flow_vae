from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from code.logger import log


def decoder_factory(decoder_type, **kwargs):
    if decoder_type == "NONCONV_DECODER":
        return NonconvDecoder(**kwargs)
    elif decoder_type == "IAF_DECODER":
        return IAFDecoder(**kwargs)
    elif decoder_type == "SNF_DECODER":
        return SNFDecoder(**kwargs)
    else:
        raise Exception("Unrecognized normalizing flow type: %s" % decoder_type)


class DecoderBaseClass(object):
    """Baseclass for all encoders."""
    def __init__(self, output_size, **kwargs):
        self._output_size = output_size

    def run(self, latent_state, is_training):
        linear_output = self._run_network(latent_state, is_training)
        return tf.sigmoid(linear_output)

    def _run_network(self, input_images, is_training):
        """Every encoder needs to implement a _run_network function"""
        raise NotImplementedError


class NonconvDecoder(DecoderBaseClass):
    """A very simple encoder consisting of several fully connected layers."""
    def __init__(self, nn_hidden_layers, **kwargs):
        super(NonconvDecoder, self).__init__(**kwargs)
        self._nn_hidden_layers = nn_hidden_layers

    def _run_network(self, latent_state, is_training):
        log("Building multilayer, fully-connected, non-convolutional decoder")
        with tf.variable_scope("NonConv_decoder"):
            normalizer_params = {'center': True, 'scale': True, 'is_training': is_training, 'trainable': True,
                                 'fused': False}

            hidden = latent_state
            for layer in self._nn_hidden_layers:
                hidden = tf.contrib.layers.fully_connected(tf.concat([hidden, latent_state], axis=1), num_outputs=layer, activation_fn=tf.nn.elu,
                                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                                           normalizer_params=normalizer_params)

            linear_output = tf.contrib.layers.fully_connected(
                tf.concat([hidden, latent_state], axis=1), num_outputs=self._output_size, activation_fn=None, biases_initializer=None)

            return linear_output

class IAFDecoder(DecoderBaseClass):
    """A CNN ResNet decoder, similar to the one used in the IAF paper (see https://arxiv.org/abs/1606.04934)."""
    def __init__(self, **kwargs):
        super(IAFDecoder, self).__init__(**kwargs)

    def _run_network(self, latent_state, is_training):
        log("Building IAF decoder with transposed CNN ResNet (6 blocks total, 2 upsampling blocks).")
        with tf.variable_scope("IAF_decoder"):
            def resize_nearest_neighbor(x, scale):
                input_shape = map(int, x.get_shape().as_list())
                size = [int(input_shape[1] * scale), int(input_shape[2] * scale)]
                x = tf.image.resize_nearest_neighbor(x, size)
                return x

            def resnet_block_T(input, upsample, filters):
                hidden = tf.contrib.layers.batch_norm(input, center=True, scale=True,
                                                      is_training=is_training, fused=False)
                hidden = tf.nn.elu(hidden)
                hidden = tf.contrib.layers.conv2d(hidden, num_outputs=filters, kernel_size=3, stride=1,
                                                  padding="SAME", activation_fn=None, biases_initializer=None)
                hidden = tf.contrib.layers.batch_norm(hidden, center=True, scale=True,
                                                      is_training=is_training, fused=False)
                hidden = tf.nn.elu(hidden)
                if upsample:
                    hidden = tf.contrib.layers.conv2d_transpose(hidden, num_outputs=filters, kernel_size=3,
                                                                stride=2, padding="SAME", activation_fn=None,
                                                                biases_initializer=None)
                    input_resized = resize_nearest_neighbor(input, 2)
                else:
                    hidden = tf.contrib.layers.conv2d(hidden, num_outputs=filters, kernel_size=3, stride=1,
                                                      padding="SAME", activation_fn=None, biases_initializer=None)
                    input_resized = input

                return input_resized + 0.1 * hidden

            filters = 64
            hidden = tf.reshape(latent_state, shape=[-1, 1, 1, latent_state.get_shape()[1].value])
            hidden = tf.contrib.layers.conv2d_transpose(hidden, num_outputs=filters, kernel_size=7, stride=1,
                                                        padding="VALID", activation_fn=tf.nn.elu,
                                                        biases_initializer=None)

            for i in range(3):
                for j in range(2):
                    hidden = resnet_block_T(hidden, upsample=(i > 0) and (j == 0), filters=filters)

            hidden = tf.contrib.layers.batch_norm(hidden, center=True, scale=True, is_training=is_training,
                                                  fused=False)
            output = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, padding="SAME",
                                              activation_fn=None)
            return tf.contrib.layers.flatten(output)


class SNFDecoder(DecoderBaseClass):
    """A CNN decoder, similar to the one used in the SNF paper (see https://arxiv.org/abs/1803.05649)."""
    def __init__(self, **kwargs):
        super(SNFDecoder, self).__init__(**kwargs)

    def _run_network(self, latent_state, is_training):
        log("Building SNF decoder with gated transposed convolutions.")
        with tf.variable_scope("SNF_decoder"):
            def gated_conv_2d_T(input, input_channels, output_channels, kernel_size, stride, padding,
                                output_padding, output_shape):
                output_conv = tf.contrib.layers.conv2d_transpose(input, num_outputs=output_channels,
                                                                 kernel_size=kernel_size, stride=stride,
                                                                 padding=padding, activation_fn=tf.nn.elu)
                gate_conv = tf.contrib.layers.conv2d_transpose(input, num_outputs=output_channels,
                                                               kernel_size=kernel_size, stride=stride,
                                                               padding=padding, activation_fn=tf.nn.sigmoid)
                return gate_conv * output_conv

            number_latents = latent_state.get_shape()[1].value
            hidden = tf.reshape(latent_state, shape=[-1, 1, 1, latent_state.get_shape()[1].value])
            hidden = gated_conv_2d_T(hidden, number_latents, 64, 7, 1, "VALID", 0, output_shape=[7])
            hidden = gated_conv_2d_T(hidden, 64, 64, 5, 1, "SAME", 0, output_shape=[7])
            hidden = gated_conv_2d_T(hidden, 64, 32, 5, 2, "SAME", 1, output_shape=[14])
            hidden = gated_conv_2d_T(hidden, 32, 32, 5, 1, "SAME", 0, output_shape=[14])
            hidden = gated_conv_2d_T(hidden, 32, 32, 5, 2, "SAME", 1, output_shape=[28])
            hidden = gated_conv_2d_T(hidden, 32, 32, 5, 1, "SAME", 0, output_shape=[28])
            output = tf.contrib.layers.conv2d(hidden, num_outputs=1, kernel_size=1, stride=1, padding="SAME",
                                              activation_fn=None)
            return tf.contrib.layers.flatten(output)
