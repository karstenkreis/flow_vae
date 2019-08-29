from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import math


def rec_loss(decodings, input_images):
    """negative log-likelihood assuming Bernoulli distribution"""
    assert decodings.get_shape() == input_images.get_shape()
    data_likelihood = 1.0 - input_images + (2.0 * input_images - 1.0) * decodings
    neg_log_likelihood = -1.0 * tf.log(tf.minimum(1.0 - 1e-10, tf.maximum(1e-10, data_likelihood)))
    return tf.reduce_sum(neg_log_likelihood, axis=1)


def kld_loss(approx_post_samples, latent_mean, latent_log_std, logdet_J_flow, approx_post_samples_flow):
    """kld between prior (standard, independent Normal distributions for all latents) and approximate posterior.
    Evaluated based on samples."""
    logprobs = logprobs_sample_based(latent_mean=latent_mean, latent_log_std=latent_log_std,
                                     approx_post_samples=approx_post_samples,
                                     approx_post_samples_nf=approx_post_samples_flow)

    if logdet_J_flow is not None:
        kld = tf.reduce_sum(logprobs - logdet_J_flow, axis=1)
    else:
        kld = tf.reduce_sum(logprobs, axis=1)

    return kld


def logprobs_sample_based(latent_mean, latent_log_std, approx_post_samples, approx_post_samples_nf):
    logprob_approx_post = logprob(approx_post_samples, latent_mean, latent_log_std)
    if approx_post_samples_nf is None:
        logprob_prior = logprob(approx_post_samples, 0.0, 0.0)
    else:
        logprob_prior = logprob(approx_post_samples_nf, 0.0, 0.0)

    return logprob_approx_post - logprob_prior


def logprob(samples, mean, log_std, include_pi_term=False):
    log_p = -1.0*log_std - tf.square(samples-mean)/(2.0 * tf.exp(2.0*log_std))
    if include_pi_term:
        log_p -= 0.5*tf.log(2.0*math.pi)
    return log_p
