from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from code.logger import log


class HamiltonianAnnealedImportanceSampling(object):
    """Annealed Importance Sampling (arxiv.org/abs/1611.04273). The sampler used to sample the probability 
    distributions along the annealing path is Hamiltonian Monte Carlo (HMC).
    """
    def __init__(self, model, num_ais_chains=16, leapfrog_steps=10, leapfrog_stepsize=0.01, anneal_steps=500,
                 use_encoder=False, target_acceptance_rate=0.65, avg_acceptance_slowness=0.9, stepsize_min=0.0001,
                 stepsize_max=0.5, stepsize_dec=0.98, stepsize_inc=1.02):
        log("\nBuilding Hamiltonian Annealed Importance Sampling for log-likelihood estimation.")
        self.sampler = HamiltonianMonteCarlo()

        self.num_ais_chains = num_ais_chains
        self.anneal_steps = anneal_steps
        self.latent_size = model.get_latentsize()
        self.batchsize = model.get_batchsize()

        self.zv = None
        self.z = tf.placeholder(tf.float32, [self.batchsize*self.num_ais_chains, self.latent_size], name='z')
        self.t = tf.placeholder(tf.float32, [], name='t')
        with tf.variable_scope("ais", reuse=tf.AUTO_REUSE):
            self.stepsize = tf.get_variable(name="hais_stepsize", initializer=leapfrog_stepsize, trainable=False)
            self.avg_acceptance_rate = tf.get_variable(name="hais_avg_acceptance_rate",
                                                       initializer=target_acceptance_rate, trainable=False)

        if use_encoder:
            self.start_distribution_sample = model.get_normal_approx_post_sample(self.num_ais_chains)
            self.start_distribution_log_pdf_function = model.normal_approx_post_log_pdf
        else:
            self.start_distribution_sample = model.get_prior_sample(self.num_ais_chains)
            self.start_distribution_log_pdf_function = model.prior_log_pdf

        self.prior_log_pdf_function = model.prior_log_pdf
        self.generator_log_pdf_function = model.generator_log_pdf

        self.lld = tf.reshape(-self._energy_function(self.z), [self.num_ais_chains, self.batchsize])

        accept, final_pos, final_vel = self.sampler.hmc_move(initial_pos=self.z, energy_fn=self._energy_function,
                                                             stepsize=self.stepsize, n_steps=leapfrog_steps)

        self.new_z, self.accept = self.sampler.hmc_updates(initial_pos=self.z, stepsize=self.stepsize,
                                                           avg_acceptance_rate=self.avg_acceptance_rate,
                                                           final_pos=final_pos, accept=accept,
                                                           stepsize_min=stepsize_min, stepsize_max=stepsize_max,
                                                           stepsize_dec=stepsize_dec, stepsize_inc=stepsize_inc,
                                                           target_acceptance_rate=target_acceptance_rate,
                                                           avg_acceptance_slowness=avg_acceptance_slowness)

    def _energy_function(self, z):
        """The energy as a function of latent configuration z along the annealing path. It is just the interpolation
        between the negative log-probability of the starting distribution (prior or approx. posterior) and the negative
        log-probability of the target distribution, this is, the posterior distrbution."""
        return -((1.0 - self.t)*self.start_distribution_log_pdf_function(z) + self.t*(self.prior_log_pdf_function(z) +
                                                                                    self.generator_log_pdf_function(z)))

    def _log_likelihood(self, t, sess, feed_dict):
        """Calculates Log-likelihood as a function of interpolation parameter t as well as current session and feed
        dictionary."""
        feed_dict[self.t] = t
        feed_dict[self.z] = self.zv
        return sess.run(self.lld, feed_dict=feed_dict)

    def run(self, sess, feed_dict):
        """Runs annealed importance sampling for given input (feed_dict) and session. Returns the importance weights."""
        weights = np.zeros([self.num_ais_chains, self.batchsize], dtype=np.float128)
        schedule = self._get_schedule(num=self.anneal_steps)
        self.zv = sess.run(self.start_distribution_sample, feed_dict=feed_dict)
        for (t0, t1) in zip(schedule[:-1], schedule[1:]):
            new_u = self._log_likelihood(t1, sess, feed_dict)
            prev_u = self._log_likelihood(t0, sess, feed_dict)
            weights += new_u - prev_u
            new_z, accept = self._step(t1, sess, feed_dict)
        return weights

    def _step(self, t, sess, feed_dict):
        """Runs a single step of annealed importance sampling for interpolation parameter t as well as current session
        and feed dictionary. Return new latent space configuration, acceptance matrix, and velocities in latent
        space."""
        feed_dict[self.t] = t
        feed_dict[self.z] = self.zv
        new_z, accept = sess.run([self.new_z, self.accept], feed_dict=feed_dict)
        self.zv = new_z
        return new_z, accept

    @staticmethod
    def _get_schedule(num, rad=4):
        """Returns the interpolation schedule along the annealing path. "num" is the number of interpolation points. The
        interpolation follows a sigmoid curve since the distribution typically changes the most at the beginning and the
        end of the interpolation procedure. "rad" specifies the steepness of the sigmoid (higher value equals steeper
        sigmoid)"""
        if num == 1:
            return np.array([0.0, 1.0])
        t = np.linspace(-rad, rad, num)
        s = 1.0 / (1.0 + np.exp(-t))
        return (s - np.min(s)) / (np.max(s) - np.min(s))


class HamiltonianMonteCarlo(object):
    """Hamiltonian Monte Carlo class"""

    @staticmethod
    def _hamiltonian(velocity, potential_energy):
        """Returns Hamiltonian as a function of configuration p, velocity v, and configurational energy function f"""
        return potential_energy + 0.5 * tf.reduce_sum(tf.multiply(velocity, velocity), axis=1)

    @staticmethod
    def _metropolis_hastings_accept(energy_prev, energy_next):
        """Returns the Metropolis Hastings accept/reject probabilities as a function of previous and new energies."""
        ediff = energy_prev - energy_next
        return (tf.exp(ediff) - tf.random_uniform(tf.shape(energy_prev))) >= 0.0

    @staticmethod
    def _simulate_dynamics(initial_pos, initial_vel, stepsize, n_steps, energy_fn):
        """
        Runs n_steps of Hamiltonian dynamics, using initial_pos and initial_vel as initial position and velocity.
        stepsize is the leapfrog integrator's stepsize and energy function is the energy function used when solving
        Hamilton's equations of motion. It returns the final position and velocity as well as the initial and final
        energy.
        """
        def leapfrog(pos, vel, energy_init, energy_final, step, i):
            """Leapfrog updates."""
            # Get both energy and gradient of energy
            energy = energy_fn(pos)
            de_dpos = tf.gradients(tf.reduce_sum(energy), pos)[0]

            # If it's the first iteration, energy is the initial energy. Hence, save it to energy_init, otherwise,
            # leave energy_init untouched
            energy_init = tf.cond(tf.equal(i, 0), lambda: energy, lambda: energy_init)

            # If it's the last iteration, energy is the final energy. Hence, save it to energy_final, otherwise, leave
            # energy_final untouched
            energy_final = tf.cond(tf.equal(i, n_steps), lambda: energy, lambda: energy_final)

            # During first iteration we only make a half step (half velocity and full position update) and during last
            # iteration, we run the other half (half velocity update only). In between, we run full steps. Hence, for
            # n calls in the while loop, leapfrog actually runs only n-1 full leapfrog steps (which is why below we
            # indeed run the while loop for n_steps+1 iteractions).
            new_vel = tf.cond(tf.logical_or(tf.equal(i, 0), tf.equal(i, n_steps)),
                              lambda: vel - 0.5 * step * de_dpos,
                              lambda: vel - step * de_dpos)
            new_pos = tf.cond(tf.equal(i, n_steps),
                              lambda: pos,
                              lambda: pos + step * new_vel)

            return [new_pos, new_vel, energy_init, energy_final, step, tf.add(i, 1)]

        def condition(pos, vel, energy_init, energy_final, step, i):
            """Stop condition for the tf_while loop. Note that this function must have the same interface like the
            leapfrog function to work with the tf_while loop. Hence the unused parameters in the interface."""
            return tf.less(i, n_steps+1)

        i = tf.constant(0)
        energy_init = tf.zeros([initial_pos.get_shape()[0]])
        energy_final = tf.zeros([initial_pos.get_shape()[0]])
        final_pos, final_vel, energy_init, energy_final, _, _ = tf.while_loop(cond=condition, body=leapfrog,
                                                                              loop_vars=[initial_pos, initial_vel,
                                                                                         energy_init, energy_final,
                                                                                         stepsize, i],
                                                                              parallel_iterations=1, back_prop=False)
        return final_pos, final_vel, energy_init, energy_final

    def hmc_move(self, initial_pos, energy_fn, stepsize, n_steps):
        """Returns everything necessary for a full Hamiltonian Monte Carlo move, consisting of drawing a new random
        velocity, running the Hamiltonian dynamics, and calculating the accept/reject decisions using the Metropolis
        Hastings criterion. The function takes the initial position, the energy function, the leapfrog stepsize and the
        number of leapfrog steps as input and returns the accept/reject decisions, the final position, and the final
        velocity."""
        initial_vel = tf.random_normal(tf.shape(initial_pos))
        final_pos, final_vel, energy_init, energy_final = self._simulate_dynamics(initial_pos=initial_pos,
                                                                                  initial_vel=initial_vel,
                                                                                  stepsize=stepsize, n_steps=n_steps,
                                                                                  energy_fn=energy_fn)
        accept = self._metropolis_hastings_accept(energy_prev=self._hamiltonian(initial_vel, energy_init),
                                                  energy_next=self._hamiltonian(final_vel, energy_final))
        return accept, final_pos, final_vel

    @staticmethod
    def hmc_updates(initial_pos, stepsize, avg_acceptance_rate, final_pos, accept, target_acceptance_rate, stepsize_inc,
                    stepsize_dec, stepsize_min, stepsize_max, avg_acceptance_slowness):
        """Actually performs the Hamiltonian Monte Carlo move, given the initial and final positions as well as the
        acceptance decisions. It returns the new positions and move acceptance decisions and it also adapts the leapfrop
        stepsize to reach the target acceptance rate in the Metropolis Hastings decisions."""
        new_stepsize_ = tf.where(avg_acceptance_rate > target_acceptance_rate, stepsize_inc, stepsize_dec) * stepsize
        new_stepsize = tf.maximum(tf.minimum(new_stepsize_, stepsize_max), stepsize_min)
        new_acceptance_rate = tf.add(avg_acceptance_slowness * avg_acceptance_rate,
                                     (1.0 - avg_acceptance_slowness) * tf.reduce_mean(tf.to_float(accept)))
        with tf.control_dependencies([stepsize.assign(new_stepsize), avg_acceptance_rate.assign(new_acceptance_rate)]):
            new_pos = tf.where(accept, final_pos, initial_pos)
        return new_pos, accept
