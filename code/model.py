from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import code.loss_functions as loss_functions
import code.flows as flows
import code.encoder as encoder
import code.decoder as decoder
from code.hais import HamiltonianAnnealedImportanceSampling
from code.logger import log


class Model(object):

    def __init__(self, modeltype, dataset, hais_params, num_is_samples_nll, warmup_epochs, approx_post_offset,
                 num_latent_units, encoder_params, flow_params, decoder_params, opt_params):
        assert modeltype in ["train", "val", "test"]
        log("\nBuilding model of type:", modeltype)
        self._modeltype = modeltype
        self._dataset = dataset
        self._batchsize = self._dataset.get_batchsize(self._modeltype)
        self._warmup_epochs = warmup_epochs
        self._approx_post_offset = approx_post_offset
        self._num_latent_units = num_latent_units
        self._num_is_samples_nll = num_is_samples_nll

        # batch data
        self._image, self._label, self._image_greyscale = self._get_batch()

        # build auxiliary variables, encoder, flow, decoder, losses, samplers, train ops, etc.
        self._build_variables()
        with tf.variable_scope("model_scope", reuse=None if self._modeltype == "train" else True):
            self._encoder = self._build_encoder(encoder_params)
            self._flow = self._build_flow(flow_params) if flow_params is not None else None
            self._decoder = self._build_decoder(decoder_params)
            with tf.variable_scope("var_scope", reuse=None if self._modeltype == "train" else True):
                self._rec_loss, self._kld_loss, self._decoding = self._build_losses(is_mode=False)

        with tf.variable_scope("model_scope", reuse=tf.AUTO_REUSE):
            if modeltype != "train":
                self.generator_log_pdf = self._build_generator_log_pdf_function()
                self._hais = self._build_hais(hais_params) if hais_params is not None else None
                with tf.variable_scope("var_scope", reuse=True):
                    rec_loss_samples, kld_loss_samples, _ = self._build_losses(is_mode=True)
                self._is_sample_op = rec_loss_samples + kld_loss_samples

            if modeltype == "train":
                self._run_step_op = self._build_run_ops(opt_params)
                self._image_sampler = self._build_image_sampler()

    ### Internal class functions ###

    def _get_batch(self):
        image, label = self._dataset.get_batch(self._modeltype)
        image_normalized = tf.cast(image, dtype=tf.float32)/(self._dataset.get_normalization()[2]-1)
        image_flattened = tf.reshape(image_normalized, shape=[self._batchsize, np.prod(self._dataset.get_image_shape())])
        image_binarized = tf.floor(image_flattened + tf.random.uniform(shape=[self._batchsize, np.prod(self._dataset.get_image_shape())]))
        return image_binarized, label, image_flattened

    def _build_variables(self):
        self._training_mode = tf.placeholder(dtype=tf.bool, shape=(), name="training_mode")
        self._fixed_image_batch = tf.placeholder(dtype=tf.float32,
                                                 shape=[self._batchsize, np.prod(self._dataset.get_image_shape())],
                                                 name="fixed_image_batch")

        with tf.variable_scope("global_step_scope", reuse=tf.AUTO_REUSE):
            self._steps = tf.get_variable(name='global_step', shape=[], initializer=tf.zeros_initializer(),
                                          trainable=False, dtype=tf.float32)

        self._epoch_elbo_node = tf.placeholder(dtype=tf.float32, shape=(), name="epoch_elbo_node")
        self._epoch_recloss_node = tf.placeholder(dtype=tf.float32, shape=(), name="epoch_recloss_node")
        self._epoch_kldloss_node = tf.placeholder(dtype=tf.float32, shape=(), name="epoch_kldloss_node")

        with tf.variable_scope("global_loss_scope"):
            self._epoch_elbo_var = tf.get_variable(name='epoch_nll_var_' + self._modeltype, shape=[], trainable=False,
                                                   dtype=tf.float32, initializer=tf.zeros_initializer)
            self._epoch_recloss_var = tf.get_variable(name='epoch_mse_var_' + self._modeltype, shape=[], trainable=False,
                                                      dtype=tf.float32, initializer=tf.zeros_initializer)
            self._epoch_kldloss_var = tf.get_variable(name='epoch_kld_var_' + self._modeltype, shape=[], trainable=False,
                                                      dtype=tf.float32, initializer=tf.zeros_initializer)

        self._assign_op_elbo = tf.assign(self._epoch_elbo_var, self._epoch_elbo_node)
        self._assign_op_recloss = tf.assign(self._epoch_recloss_var, self._epoch_recloss_node)
        self._assign_op_kldloss = tf.assign(self._epoch_kldloss_var, self._epoch_kldloss_node)

        with tf.variable_scope("model_scope"):
            tf.summary.scalar("ELBO " + self._modeltype, self._epoch_elbo_var)
            tf.summary.scalar("RECLOSS " + self._modeltype, self._epoch_recloss_var)
            tf.summary.scalar("KLDLOSS " + self._modeltype, self._epoch_kldloss_var)

    def _build_encoder(self, encoder_params):
        return encoder.encoder_factory(num_latent_units=self._num_latent_units, **encoder_params)

    def _build_flow(self, flow_params):
        return flows.flow_factory(batchsize=self._batchsize, num_latent_units=self._num_latent_units, **flow_params)

    def _build_decoder(self, decoder_params):
        return decoder.decoder_factory(output_size=np.prod(self._dataset.get_image_shape()), **decoder_params)

    def _build_image_sampler(self):
        log("\nBuilding new image sampler")
        with tf.variable_scope("var_scope", reuse=True):
            draw_latent_state = tf.random_normal(shape=[1, self._num_latent_units])
        return self._decoder.run(draw_latent_state, is_training=False)

    def _build_losses(self, is_mode):
        if is_mode:
            log("\nBuilding sampling ops for importance sampling for log-likelihood estimation.")
            input_images = self._fixed_image_batch
        else:
            log("\nBuilding training objectives.")
            input_images = self._image

        # run encoder
        latent_mean, latent_log_std, flow_context = self._encoder.run(
            input_images=input_images, context_size=self._flow.get_context_size() if self._flow is not None else None,
            is_training=self._training_mode)

        # draw samples via reparametrization trick
        approx_post_samples = self._draw_latent_state_sample(latent_mean, latent_log_std)

        # run flow, if there is a flow
        if self._flow is not None:
            approx_post_samples_flow, logdet_J_flow = self._flow.run(
                input=approx_post_samples, context=flow_context, is_training=self._training_mode)
        else:
            approx_post_samples_flow, logdet_J_flow = None, None

        # calculate kld loss
        kld_loss = loss_functions.kld_loss(approx_post_samples, latent_mean, latent_log_std + self._approx_post_offset,
                                           logdet_J_flow, approx_post_samples_flow)

        # run decoder
        decodings = self._decoder.run(
            latent_state=approx_post_samples_flow if self._flow is not None else approx_post_samples,
            is_training=self._training_mode)

        # calculate reconst. loss
        rec_loss = loss_functions.rec_loss(decodings, input_images)

        if is_mode:
            return rec_loss, kld_loss, decodings
        else:
            return tf.reduce_mean(rec_loss), tf.reduce_mean(kld_loss), decodings

    def _draw_latent_state_sample(self, latent_mean, latent_log_std):
        return latent_mean + tf.exp(latent_log_std + self._approx_post_offset) * \
               tf.random_normal(shape=latent_mean.get_shape())

    def _get_warmup(self):
        if self._warmup_epochs > 0:
            num_epochs_evaluated = tf.floor(self._steps / self._dataset.get_steps_per_epoch(self._modeltype))
            return tf.minimum(1.0, num_epochs_evaluated / self._warmup_epochs)
        else:
            return 1.0

    def _build_hais(self, hais_params):
        return HamiltonianAnnealedImportanceSampling(model=self, **hais_params)

    def _build_run_ops(self, opt_params):
        train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        optimizer = self._build_optimizer(**opt_params)
        warmup = self._get_warmup()
        tf.summary.scalar("warmup_scale", warmup)
        grads_and_vars = optimizer.compute_gradients(self._rec_loss + warmup * self._kld_loss, var_list=train_vars)

        log('\nBuilding optimizer for ELBO objective for variables:')
        for g, v in grads_and_vars:
            log("Variable: %s" % v.name)
            if g is None:
                log('No gradient information association with variable', v.name, 'with shape', v.get_shape())

        step_increment_op = tf.assign_add(self._steps, 1)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            ll_opt_op = optimizer.apply_gradients(grads_and_vars, name="opt_op")

        with tf.control_dependencies([ll_opt_op, step_increment_op]):
            run_step_op = [tf.identity(self._rec_loss), tf.identity(self._kld_loss)]

        return run_step_op

    def _build_optimizer(self, base_learning_rate, learning_rate_decay_epochs, decay_scale_factor):
        num_epochs_evaluated = tf.floor(self._steps / self._dataset.get_steps_per_epoch(self._modeltype))
        num_learning_rate_periods = tf.to_float(tf.floor(tf.divide(num_epochs_evaluated, learning_rate_decay_epochs)))
        learning_rate = base_learning_rate * tf.pow(decay_scale_factor, num_learning_rate_periods)
        if self._modeltype == "train":
            tf.summary.scalar("learning rate", learning_rate)
        return tf.train.AdamOptimizer(learning_rate=learning_rate)

    def _build_generator_log_pdf_function(self):
        log("\nBuilding generator log pdf function for latent space inputs")

        def generator_log_pdf_function(latentstate):
            num_batches = int(latentstate.get_shape()[0].value / self._batchsize)
            with tf.variable_scope("var_scope", reuse=True):
                decoder_output = self._decoder.run(latentstate, is_training=False)
            return -loss_functions.rec_loss(decoder_output, tf.tile(self._fixed_image_batch, [num_batches, 1]))

        return generator_log_pdf_function

    def _run_step(self, sess):
        return sess.run(self._run_step_op, feed_dict={self._training_mode: True})

    def _run_elbo(self, sess):
        return sess.run([self._rec_loss, self._kld_loss], feed_dict={self._training_mode: False})

    def _update_variables_for_tensorboard(self, sess, epoch_rec_loss, epoch_kld_loss, epoch_elbo):
        sess.run(self._assign_op_elbo, feed_dict={self._epoch_elbo_node: epoch_elbo})
        sess.run(self._assign_op_recloss, feed_dict={self._epoch_recloss_node: epoch_rec_loss})
        sess.run(self._assign_op_kldloss, feed_dict={self._epoch_kldloss_node: epoch_kld_loss})

    ### Functions for interaction with external code that is not part of this class ###

    def train_epoch(self, sess):
        steps = self._dataset.get_steps_per_epoch(self._modeltype)
        rec_loss = 0.0
        kld_loss = 0.0
        elbo = 0.0
        for _ in range(steps):
            this_rec_loss, this_kld_loss = self._run_step(sess)
            rec_loss += this_rec_loss
            kld_loss += this_kld_loss
            elbo += this_rec_loss + this_kld_loss

        epoch_rec_loss_train = rec_loss / steps
        epoch_kld_loss_train = kld_loss / steps
        epoch_elbo_train = elbo / steps

        self._update_variables_for_tensorboard(sess, epoch_rec_loss_train, epoch_kld_loss_train, epoch_elbo_train)
        return epoch_rec_loss_train, epoch_kld_loss_train, epoch_elbo_train

    def generate_new_images(self, sess):
        return sess.run(self._image_sampler, feed_dict={self._training_mode: False})

    def get_data_images(self, sess):
        return sess.run(self._image_greyscale)

    def get_input_and_reconst(self, sess):
        return sess.run([self._image, self._decoding], feed_dict={self._training_mode: False})

    def calc_elbo(self, sess):
        steps = self._dataset.get_steps_per_epoch(self._modeltype)
        rec_loss = 0.0
        kld_loss = 0.0
        elbo = 0.0
        for _ in range(steps):
            this_rec_loss, this_kld_loss = self._run_elbo(sess)
            rec_loss += this_rec_loss
            kld_loss += this_kld_loss
            elbo += this_rec_loss + this_kld_loss

        epoch_rec_loss = rec_loss / steps
        epoch_kld_loss = kld_loss / steps
        epoch_elbo = elbo / steps

        self._update_variables_for_tensorboard(sess, epoch_rec_loss, epoch_kld_loss, epoch_elbo)
        return elbo / steps

    def calc_nll(self, sess):
        steps_per_epoch = self._dataset.get_steps_per_epoch(self._modeltype)
        image_batches = [sess.run(self._image) for _ in range(steps_per_epoch)]
        total_nll_is = 0.0
        total_nll_hais = 0.0
        is_weights = np.zeros((self._num_is_samples_nll, self._batchsize), dtype=np.float128)

        for batch_number, image in enumerate(image_batches):
            # Calculate via naive importance sampling using approximate posterior as proposal distribution
            for is_sample in range(self._num_is_samples_nll):
                is_weights[is_sample] = - sess.run(self._is_sample_op,
                                                   feed_dict={self._fixed_image_batch: image, self._training_mode: False})
            mean_val_is = np.mean(is_weights, axis=0, keepdims=True)
            ll_is = mean_val_is + np.log(np.mean(np.exp(is_weights - mean_val_is), axis=0))
            this_batch_is_nll_estimate = -np.mean(ll_is)
            total_nll_is += this_batch_is_nll_estimate
            log("Batch", str(batch_number + 1), "of", steps_per_epoch, "nll estimate via naive Importance Sampling:",
                this_batch_is_nll_estimate)

            # Calculate via Hamiltonian annealed importance sampling
            if self._hais is not None:
                hais_weights = self._hais.run(sess, feed_dict={self._fixed_image_batch: image, self._training_mode: False})
                this_batch_hais_nll_estimate = -np.mean(hais_weights)
                total_nll_hais += this_batch_hais_nll_estimate
                log("Batch", str(batch_number + 1), "of", steps_per_epoch, "nll estimate via Hamiltonian Annealed Importance Sampling:",
                    this_batch_hais_nll_estimate)

        if self._hais is not None:
            return total_nll_is / steps_per_epoch, total_nll_hais / steps_per_epoch
        else:
            return total_nll_is / steps_per_epoch, None

    def get_batchsize(self):
        return self._batchsize

    def get_latentsize(self):
        return self._num_latent_units

    def get_prior_sample(self, num_batches):
        return tf.random_normal(shape=[num_batches * self._batchsize, self._num_latent_units])

    @staticmethod
    def prior_log_pdf(latentstate):
        return tf.reduce_sum(loss_functions.logprob(latentstate, 0.0, 0.0, include_pi_term=True), axis=1)

    def get_normal_approx_post_sample(self, num_batches):
        """This does not sample from the actual approximate posterior, only from the base distribution, which is fed
        to the flow. We cannot use the flow-based approximate posterior for HAIS-based log-likelihood evaluation,
        because we would have to invert the flow, which isn't simple. The hope is that this base distribution is still
        better than the prior."""
        input_images = tf.tile(self._fixed_image_batch, [num_batches, 1])

        with tf.variable_scope("var_scope", reuse=True):
            latent_mean, latent_log_std, _ = self._encoder.run(
                input_images=input_images,
                context_size=self._flow.get_context_size() if self._flow is not None else None,
                is_training=self._training_mode)

        return self._draw_latent_state_sample(latent_mean, latent_log_std)

    def normal_approx_post_log_pdf(self, latentstate):
        """This does not calculate the log-pdf using the actual approximate posterior, only the base distribution,
        which is fed to the flow. We cannot use the flow-based approximate posterior for HAIS-based log-likelihood
        evaluation, because we would have to invert the flow, which isn't simple. The hope is that this base
        distribution is still better than the prior."""
        num_batches = int(latentstate.get_shape()[0].value / self._batchsize)
        input_images = tf.tile(self._fixed_image_batch, [num_batches, 1])

        with tf.variable_scope("var_scope", reuse=True):
            latent_mean, latent_log_std, _ = self._encoder.run(
                input_images=input_images,
                context_size=self._flow.get_context_size() if self._flow is not None else None,
                is_training=self._training_mode)

        return tf.reduce_sum(loss_functions.logprob(
            latentstate, latent_mean, latent_log_std + self._approx_post_offset, include_pi_term=True), axis=1)
