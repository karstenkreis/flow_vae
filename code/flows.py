from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from code.logger import log


def flow_factory(flow_type, **kwargs):
    if flow_type == "IDENTITY_FLOW":
        return IdentityFlow(**kwargs)
    elif flow_type == "IAF":
        return IAF(**kwargs)
    else:
        raise Exception("Unrecognized normalizing flow type: %s" % flow_type)


class FlowBaseClass(object):
    """Baseclass for all flows."""
    def __init__(self, flow_layers, context_size, batchsize, **kwargs):

        self._flow_layers = flow_layers
        self._context_size = context_size
        self._batchsize = batchsize

    def run(self, input, context, is_training):
        log_det_J = tf.zeros(shape=input.get_shape())
        output = input

        for flow_step in range(self._flow_layers):
            output, log_det_J_step = self._apply_flow_layer(input=output, context=context, flow_step=flow_step,
                                                            is_training=is_training)
            assert log_det_J.get_shape() == log_det_J_step.get_shape()
            log_det_J += log_det_J_step

        return output, log_det_J

    def _apply_flow_layer(self, input, context, flow_step, is_training):
        """Every flow needs to implement an _apply_flow_layer function"""
        raise NotImplementedError

    def get_context_size(self):
        return self._context_size


class IdentityFlow(FlowBaseClass):
    """This flow doesn't do anything. Just for debugging."""
    def __init__(self, **kwargs):
        super(IdentityFlow, self).__init__(**kwargs)

    def _apply_flow_layer(self, input, context, flow_step, is_training):
        return input, tf.zeros(shape=input.get_shape())


class IAF(FlowBaseClass):
    """Inverse autoregressive flow (see https://arxiv.org/abs/1606.04934)"""
    def __init__(self, cmade_hidden_layers, cmade_batchnorm, cmade_context_SNF_like, flow_shift_only, num_latent_units,
                 **kwargs):
        super(IAF, self).__init__(**kwargs)

        self._cmade_context_SNF_like = cmade_context_SNF_like
        self._cmade_hidden_layers = cmade_hidden_layers
        self._flow_shift_only = flow_shift_only
        self._cmade_batchnorm = cmade_batchnorm  # bool, if True, rather use batchnorm in the cMADE
        self._cmade_masks_even, self._cmade_masks_odd = self._construct_masks(num_latent_units, self._cmade_hidden_layers)

        if self._flow_shift_only:
            self._transformer_function = self._iaf_transformer_function_shiftonly
        else:
            self._transformer_function = self._iaf_transformer_function_shift_and_scale

    @staticmethod
    def _iaf_transformer_function_shift_and_scale(m_t, s_t, approx_post_samples):
        sigma_t = tf.sigmoid(s_t)
        transformed = sigma_t * approx_post_samples + (1.0 - sigma_t) * m_t
        log_det_J = tf.log(sigma_t)
        return transformed, log_det_J

    @staticmethod
    def _iaf_transformer_function_shiftonly(m_t, s_t, approx_post_samples):
        return approx_post_samples + m_t, tf.zeros(shape=approx_post_samples.get_shape())

    def _apply_flow_layer(self, input, context, flow_step, is_training):
        reverse_dependencies = False if flow_step % 2 == 0 else True
        log("Building inverse autoregressive flow layer", str(flow_step), "and reversed dependencies are",
            reverse_dependencies)

        with tf.variable_scope("cMADE_layer_" + str(flow_step)):
            m_t, s_t = self._run_cMADE(input, context=context,
                                       masks=self._cmade_masks_odd if reverse_dependencies else self._cmade_masks_even,
                                       is_training=is_training)

        assert m_t.get_shape() == input.get_shape()
        if s_t is not None:
            assert s_t.get_shape() == input.get_shape()

        transformed, log_det_j = self._transformer_function(m_t, s_t, input)

        return transformed, log_det_j

    def _construct_masks(self, n_input, n_hiddens):
        units = [n_input] + n_hiddens + [n_input]
        for layer in range(len(units)-1):
            n_in = units[layer]
            n_out = units[layer+1]
            if (n_in > n_out and n_in % n_out != 0) or (n_out > n_in and n_out % n_in != 0):
                raise ValueError("Number of units in cMADE layers don't evenly divide each other. Cannot construct "
                                 "suitable masks!")
        masks_even = self._get_masks(n_input, n_hiddens, switched_dependencies=False)
        masks_odd = self._get_masks(n_input, n_hiddens, switched_dependencies=True)

        # check that the product is lower or upper triangular with 0 diagonal for both even and odd masks.
        mask_prod_even = np.eye(n_input)
        mask_prod_odd = np.eye(n_input)
        assert len(masks_even) == len(masks_odd)
        for mask_even, mask_odd in zip(masks_even, masks_odd):
            assert not (mask_even == mask_odd).all()
            mask_prod_even = np.matmul(mask_prod_even, mask_even)
            mask_prod_odd = np.matmul(mask_prod_odd, mask_odd)
        assert np.allclose(mask_prod_even, np.triu(mask_prod_even) - np.diag(np.diag(mask_prod_even)))
        assert np.allclose(mask_prod_odd, np.tril(mask_prod_odd) - np.diag(np.diag(mask_prod_odd)))
        assert not (mask_prod_even == mask_prod_odd).all()
        np.testing.assert_array_equal(mask_prod_odd * mask_prod_even, np.zeros(shape=np.shape(mask_prod_even)))

        return masks_even, masks_odd

    @staticmethod
    def _get_masks(n_input, n_hiddens, switched_dependencies=False):
        """Build masks. Works only if n_in % n_out == 0 or n_out % n_in == 0 for all layers"""
        masks = []
        units = [n_input] + n_hiddens + [n_input]
        for layer in range(len(units)-1):
            n_in = units[layer]
            n_out = units[layer+1]

            mask = np.ones((n_in, n_out)) if not switched_dependencies else np.zeros((n_in, n_out))
            diagonalzeros = True if layer == len(units)-2 else False

            assert n_in % n_out == 0 or n_out % n_in == 0
            if n_out >= n_in:
                k = int(n_out / n_in)
                for i in range(n_in):
                    mask[i, :i * k] = 0 if not switched_dependencies else 1
                    if diagonalzeros and not switched_dependencies:
                        mask[i, i * k:(i + 1) * k] = 0
                    if not diagonalzeros and switched_dependencies:
                        mask[i, i * k:(i + 1) * k] = 1
            else:
                k = int(n_in / n_out)
                for i in range(n_out):
                    mask[(i + 1) * k:, i] = 0 if not switched_dependencies else 1
                    if diagonalzeros and not switched_dependencies:
                        mask[i * k:(i + 1) * k, i] = 0
                    if not diagonalzeros and switched_dependencies:
                        mask[i * k:(i + 1) * k, i] = 1

            masks.append(mask.astype(np.float32))

        return masks

    def _linear_layer_cMADE(self, input, output_size, name, mask=None, positive_bias_s=0.0, bias=True):
        weights = tf.get_variable(name=name+"_weights", dtype=tf.float32,
                                  shape=[input.get_shape()[1].value, output_size], trainable=True,
                                  initializer=tf.contrib.layers.xavier_initializer())

        if mask is not None:
            masked_weights = mask * weights
        else:
            masked_weights = weights

        output = tf.matmul(input, masked_weights)

        if bias:
            biases = tf.get_variable(name=name+"_biases", dtype=tf.float32, shape=[1, output_size],
                                     trainable=True, initializer=tf.constant_initializer(positive_bias_s))
            output += biases

        return output

    def _run_cMADE(self, approx_post_samples, context, masks, is_training):
        first_hidden_masked = self._linear_layer_cMADE(approx_post_samples, self._cmade_hidden_layers[0],
                                                       name="cMADE_layer_0", mask=masks[0],
                                                       bias=False if self._cmade_batchnorm else True)

        if self._cmade_context_SNF_like:
            if self._cmade_batchnorm:
                with tf.variable_scope("cMADE_BN_0", reuse=None):
                    first_hidden_masked = tf.contrib.layers.batch_norm(first_hidden_masked, center=True, scale=True,
                                                                       is_training=is_training, fused=False)

            first_hidden_masked_act = tf.nn.elu(first_hidden_masked)

            if context is not None:
                assert context.get_shape()[1] == first_hidden_masked_act.get_shape()[1], \
                    "When the context is fed into the MADE as in the Sylvester Normalizing flow paper, the context " \
                    "size and the width of the first MADE layer have to be the same."
                hidden = context + first_hidden_masked_act
            else:
                hidden = first_hidden_masked_act

        else:
            if context is not None:
                first_hidden_context = self._linear_layer_cMADE(context, self._cmade_hidden_layers[0],
                                                            name="cMADE_layer_0_context", mask=None, bias=False)

                first_hidden_combined = first_hidden_context + first_hidden_masked
            else:
                first_hidden_combined = first_hidden_masked

            if self._cmade_batchnorm:
                with tf.variable_scope("cMADE_BN_0", reuse=None):
                    first_hidden_combined = tf.contrib.layers.batch_norm(first_hidden_combined, center=True, scale=True,
                                                                         is_training=is_training, fused=False)

            hidden = tf.nn.elu(first_hidden_combined)

        for layer in range(1, len(self._cmade_hidden_layers)):
            hidden = self._linear_layer_cMADE(hidden, self._cmade_hidden_layers[layer],
                                              name="cMADE_layer_"+str(layer), mask=masks[layer],
                                              bias=False if self._cmade_batchnorm else True)

            if self._cmade_batchnorm:
                with tf.variable_scope("cMADE_BN_"+str(layer), reuse=None):
                    hidden = tf.contrib.layers.batch_norm(hidden, center=True, scale=True, is_training=is_training,
                                                          fused=False)

            hidden = tf.nn.elu(hidden)

        m_out = self._linear_layer_cMADE(hidden, approx_post_samples.get_shape()[1].value,
                                         name="cMADE_layer_out_m", mask=masks[-1],
                                         bias=False if self._cmade_batchnorm else True)

        if self._cmade_batchnorm:
            with tf.variable_scope("cMADE_BN_out_m", reuse=None):
                m_out = tf.contrib.layers.batch_norm(m_out, center=True, scale=True, is_training=is_training,
                                                     fused=False)

        if self._flow_shift_only:
            return m_out, None
        else:
            if self._cmade_batchnorm:
                s_out = self._linear_layer_cMADE(hidden, approx_post_samples.get_shape()[1].value,
                                                 name="cMADE_layer_out_s", mask=masks[-1], bias=False)
                with tf.variable_scope("cMADE_BN_out_s", reuse=None):
                    s_out = tf.contrib.layers.batch_norm(s_out, center=True, scale=True,
                                                         param_initializers={'beta': tf.constant_initializer(3.0),
                                                                             'gamma': tf.constant_initializer(1.0)},
                                                         is_training=is_training, fused=False)
            else:
                s_out = self._linear_layer_cMADE(hidden, approx_post_samples.get_shape()[1].value,
                                                 name="cMADE_layer_out_s", mask=masks[-1], positive_bias_s=2.0)
            return m_out, s_out
