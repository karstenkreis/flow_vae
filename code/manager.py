from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from code.model import Model
from code.analysis import Analysis

from code.logger import log
import code.logger as logger
import code.datasets as datasets
import pprint
import time


class Manager(object):
    def __init__(self, name, num_epochs, model_params, data_params, restart_filename=None):
        self._name = name
        self._restart_filename = restart_filename

        self._num_epochs = num_epochs
        self._epoch = 0

        self._model_params = model_params
        self._data_params = data_params

        self._model_train = None
        self._model_val = None
        self._model_test = None
        self._analysis = None

        self._sess = None
        self._saver = None
        self._summary_op = None
        self._summary_writer = None

        self._name = name + '_ID_%d' % np.random.randint(100000, 1000000)
        logger.LOG_FILENAME = self._name + ".log"

        self._tensorboard_dir = 'tensorboard_dir/'
        log("Tensorflow version:", tf.__version__)
        log("This calculation's name: %s" % self._name)
        log("Logging into file: %s" % ('logs/' + logger.LOG_FILENAME))
        log("Using tensorboard dir: %s" % self._tensorboard_dir)

        log("\nPrinting this calculation's parameters:")
        pp = pprint.PrettyPrinter(indent=2)
        log("\nmodel params:\n", pp.pformat(self._model_params))
        log("\ndata params:\n", pp.pformat(self._data_params))
        log("\nnum_epochs:", self._num_epochs)

    ### Internal class functions ###

    def _build_model(self, dataset):
        with tf.variable_scope("model_scope", reuse=None):
            self._model_train = Model(modeltype="train", dataset=dataset, **self._model_params)

        with tf.variable_scope("model_scope", reuse=True):
            if dataset.using_validation_set():
                self._model_val = Model(modeltype="val", dataset=dataset, **self._model_params)
            if dataset.using_test_set():
                self._model_test = Model(modeltype="test", dataset=dataset, **self._model_params)

        self._analysis = Analysis(self._name, dataset, self._model_train, self._model_val, self._model_test)

        self._summary_op = tf.summary.merge_all()

    def _init_saver_and_variables(self):
        assert self._sess is not None
        # init variables
        tf.global_variables_initializer().run(session=self._sess)

        # set up saver (trainable variables and batchnorm running averages)
        variables_to_save = tf.get_collection(tf.GraphKeys.VARIABLES, scope="model_scope/var_scope/")
        log("\nVariables to be saved:")
        for v in variables_to_save:
            log(v.name)
        self._saver = tf.train.Saver(variables_to_save, max_to_keep=1)

        # reload parameter state if necessary
        if self._restart_filename is not None:
            restore_file_name = self._tensorboard_dir + self._restart_filename + "/PARAMETERSTATE.ckpt"
            log("\nTrying restore file: " + restore_file_name)
            try:
                self._saver.restore(self._sess, restore_file_name)
                self._name += '_restart'
                log("Successfully restored variables from checkpoint!")
            except ValueError as exception:
                log("\n!! WARNING: !!")
                log("Tried to restore variables from checkpoint, but failed as ValueError:", exception.message)
                log("Will start from freshly initialized variables.")

        # Instantiate a SummaryWriter to output summaries and the graph
        self._summary_writer = tf.summary.FileWriter(self._tensorboard_dir + self._name, graph=self._sess.graph)

        # Finalize graph
        self._sess.graph.finalize()
        log("\nComputation graph finalized.")

    def _save_checkpoint(self):
        assert (self._sess is not None) and (self._saver is not None)
        start_time_checkpoint = time.time()
        save_path = self._saver.save(self._sess, self._tensorboard_dir + self._name + "/PARAMETERSTATE.ckpt")
        total_time_checkpoint = time.time() - start_time_checkpoint
        log("\nModel saved in file: %s in %0.2f secs." % (save_path, total_time_checkpoint))

    def _update_tensorboard(self):
        assert (self._sess is not None) and (self._summary_op is not None) and (self._summary_writer is not None)
        log('Updating tensorboard for epoch %d' % self._epoch)
        summary_str = self._sess.run(self._summary_op)
        self._summary_writer.add_summary(summary_str, global_step=self._epoch)
        self._summary_writer.flush()

    def _train_model(self):
        assert self._sess is not None

        log("\n\nStarting training. Number of epochs:", self._num_epochs)
        for step_full_train in xrange(self._num_epochs):
            self._epoch += 1
            log("\nRunning epoch:", self._epoch)

            start_time = time.time()
            rec_loss_train, kld_loss_train, elbo_loss_train = self._model_train.train_epoch(sess=self._sess)
            log("loss train set, ELBO (batchnorm in training mode):", elbo_loss_train)
            log("loss train set, reconst. term (batchnorm in training mode):", rec_loss_train)
            log("loss train set, KLD term (batchnorm in training mode):", kld_loss_train)

            if self._model_val is not None:
                elbo_loss_val = self._model_val.calc_elbo(sess=self._sess)
                log("loss val set, ELBO (batchnorm in inference mode):", elbo_loss_val)
            if self._model_test is not None:
                elbo_loss_test = self._model_test.calc_elbo(sess=self._sess)
                log("loss test set, ELBO (batchnorm in inference mode):", elbo_loss_test)
            end_time = time.time()

            log("Epoch runtime in seconds (train set):", end_time - start_time)
            self._update_tensorboard()

            if step_full_train % 500 == 0 and step_full_train > 0:
                self._save_checkpoint()

        self._save_checkpoint()

    def _run_analysis(self):
        self._analysis.draw_samples_from_data(sess=self._sess)
        self._analysis.draw_samples_from_model(sess=self._sess)
        self._analysis.compare_input_reconst(sess=self._sess)
        analysis_results_dict = self._analysis.calc_metrics(sess=self._sess)
        return analysis_results_dict

    ### Functions for interaction with external code that is not part of this class ###

    def run_model(self):
        log("\n\n --- Loading data --- ")
        dataset = datasets.Datasets(**self._data_params)

        log("\n\n --- Building models --- ")
        self._build_model(dataset)

        with tf.Session() as self._sess:
            log("\n\n --- Initializing variables --- ")
            self._init_saver_and_variables()

            log("\n\n --- Training model --- ")
            if self._num_epochs > 0:
                self._train_model()
            log("\n\nTraining models done")

            log("\n\n --- Analysing model and drawing samples from model --- ")
            self._run_analysis()

            log("\n\n --- FINISHED --- ")

        tf.reset_default_graph()
