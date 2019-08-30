from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from code.logger import log


def encoder_factory(encoder_type, **kwargs):
    if encoder_type == "NONCONV_ENCODER":
        return NonconvEncoder(**kwargs)
    elif encoder_type == "IAF_ENCODER":
        return IAFEncoder(**kwargs)
    elif encoder_type == "SNF_ENCODER":
        return SNFEncoder(**kwargs)
    else:
        raise Exception("Unrecognized normalizing flow type: %s" % encoder_type)


class EncoderBaseClass(object):
    """Baseclass for all encoders."""
    def __init__(self, num_latent_units, **kwargs):
        self._num_latent_units = num_latent_units

    def run(self, input_images, context_size, is_training):
        linear_output, last_hidden = self._run_network(input_images, is_training)

        if context_size is not None and context_size > 0:
            with tf.variable_scope("Context_predictor"):
                normalizer_params = {'center': True, 'scale': True, 'is_training': is_training, 'trainable': True,
                                     'fused': False}
                flow_context = tf.contrib.layers.fully_connected(last_hidden, num_outputs=context_size, activation_fn=None,
                                                                 normalizer_fn=tf.contrib.layers.batch_norm,
                                                                 normalizer_params=normalizer_params)
        else:
            flow_context = None

        latent_mean, latent_log_std = tf.split(value=linear_output, num_or_size_splits=2, axis=1)
        return latent_mean, latent_log_std, flow_context

    def _run_network(self, input_images, is_training):
        """Every encoder needs to implement a _run_network function"""
        raise NotImplementedError


class NonconvEncoder(EncoderBaseClass):
    """A very simple encoder consisting of several fully connected layers."""
    def __init__(self, nn_hidden_layers, **kwargs):
        super(NonconvEncoder, self).__init__(**kwargs)
        self._nn_hidden_layers = nn_hidden_layers

    def _run_network(self, input_images, is_training):
        log("Building multilayer, fully-connected, non-convolutional encoder")
        with tf.variable_scope("NonConv_encoder"):
            normalizer_params = {'center': True, 'scale': True, 'is_training': is_training, 'trainable': True,
                                 'fused': False}

            hidden = input_images
            for layer in self._nn_hidden_layers:
                hidden = tf.contrib.layers.fully_connected(tf.concat([hidden, input_images], axis=1), num_outputs=layer, activation_fn=tf.nn.elu,
                                                           normalizer_fn=tf.contrib.layers.batch_norm,
                                                           normalizer_params=normalizer_params)

            linear_output = tf.contrib.layers.fully_connected(
                tf.concat([hidden, input_images], axis=1), num_outputs=2 * self._num_latent_units, activation_fn=None,
                normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=normalizer_params,
                biases_initializer=None)

            return linear_output, hidden


class IAFEncoder(EncoderBaseClass):
    """A CNN ResNet encoder, similar to the one used in the IAF paper (see https://arxiv.org/abs/1606.04934)."""
    def __init__(self, **kwargs):
        super(IAFEncoder, self).__init__(**kwargs)

    def _run_network(self, input_images, is_training):
        log("Building IAF encoder with CNN ResNet (6 blocks total, 2 downsampling blocks via strided convolutions).")
        with tf.variable_scope("IAF_encoder"):
            def resize_nearest_neighbor(x, scale):
                input_shape = map(int, x.get_shape().as_list())
                size = [int(input_shape[1] * scale), int(input_shape[2] * scale)]
                x = tf.image.resize_nearest_neighbor(x, size)
                return x

            def resnet_block(input, downsample, filters):
                if downsample:
                    input_resized = resize_nearest_neighbor(input, 0.5)
                    stride = 2
                else:
                    input_resized = input
                    stride = 1

                hidden = tf.contrib.layers.batch_norm(input, center=True, scale=True,
                                                      is_training=is_training, fused=False)
                hidden = tf.nn.elu(hidden)
                hidden = tf.contrib.layers.conv2d(hidden, num_outputs=filters, kernel_size=3, stride=stride,
                                                  padding="SAME", activation_fn=None, biases_initializer=None)
                hidden = tf.contrib.layers.batch_norm(hidden, center=True, scale=True,
                                                      is_training=is_training, fused=False)
                hidden = tf.nn.elu(hidden)
                hidden = tf.contrib.layers.conv2d(hidden, num_outputs=filters, kernel_size=3, stride=1,
                                                  padding="SAME", activation_fn=None, biases_initializer=None)
                return input_resized + 0.1 * hidden

            filters = 64
            hidden = tf.reshape(input_images, shape=[-1, 28, 28, 1])

            for i in range(3):
                for j in range(2):
                    hidden = resnet_block(hidden, downsample=(i > 0) and (j == 0), filters=filters)
            hidden = tf.contrib.layers.batch_norm(hidden, center=True, scale=True,
                                                  is_training=is_training,
                                                  fused=False)
            hidden = tf.contrib.layers.flatten(hidden)

            normalizer_params = {'center': True, 'scale': True, 'is_training': is_training,
                                 'fused': False}
            last_hidden = tf.contrib.layers.fully_connected(
                hidden, num_outputs=450, activation_fn=tf.nn.elu, normalizer_fn=tf.contrib.layers.batch_norm,
                normalizer_params=normalizer_params, biases_initializer=None)
            linear_output = tf.contrib.layers.fully_connected(
                last_hidden, num_outputs=2*self._num_latent_units, activation_fn=None,
                normalizer_fn=tf.contrib.layers.batch_norm, normalizer_params=normalizer_params,
                biases_initializer=None)
            return linear_output, last_hidden


class SNFEncoder(EncoderBaseClass):
    """A CNN encoder, similar to the one used in the SNF paper (see https://arxiv.org/abs/1803.05649)."""
    def __init__(self, **kwargs):
        super(SNFEncoder, self).__init__(**kwargs)

    def _run_network(self, input_images, is_training):
        log("Building SNF encoder with gated convolutions.")
        with tf.variable_scope("SNF_encoder"):
            def gated_conv_2d(input, input_channels, output_channels, kernel_size, stride, padding):
                output_conv = tf.contrib.layers.conv2d(input, num_outputs=output_channels, kernel_size=kernel_size,
                                                       stride=stride, padding=padding, activation_fn=tf.nn.elu)
                gate_conv = tf.contrib.layers.conv2d(input, num_outputs=output_channels, kernel_size=kernel_size,
                                                     stride=stride, padding=padding, activation_fn=tf.nn.sigmoid)
                return gate_conv * output_conv

            hidden = tf.reshape(input_images, shape=[-1, 28, 28, 1])
            hidden = gated_conv_2d(hidden, 1, 32, 5, 1, "SAME")
            hidden = gated_conv_2d(hidden, 32, 32, 5, 2, "SAME")
            hidden = gated_conv_2d(hidden, 32, 64, 5, 1, "SAME")
            hidden = gated_conv_2d(hidden, 64, 64, 5, 2, "SAME")
            hidden = gated_conv_2d(hidden, 64, 64, 5, 1, "SAME")
            hidden = gated_conv_2d(hidden, 64, 256, 7, 1, "VALID")
            last_hidden = tf.contrib.layers.flatten(hidden)
            linear_output = tf.contrib.layers.fully_connected(last_hidden, num_outputs=2*self._num_latent_units,
                                                              activation_fn=None)
            return linear_output, last_hidden