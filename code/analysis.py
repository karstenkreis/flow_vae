from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from code.logger import log


class Analysis(object):

    def __init__(self, name, dataset, train_model, val_model, test_model):
        log("\n\nBuilding analysis module")
        self._name = name
        self._dataset = dataset
        self._train_model = train_model
        self._val_model = val_model
        self._test_model = test_model

    def calc_metrics(self, sess):
        log("\nCalculation of negative log-likelihoods and ELBO for all data sets (batchnorm in inference mode):")
        elbo_train_set = self._train_model.calc_elbo(sess)
        elbo_val_set = self._val_model.calc_elbo(sess) if self._val_model is not None else None
        elbo_test_set = self._test_model.calc_elbo(sess) if self._test_model is not None else None

        nll_val_set_is, nll_val_set_hais = self._val_model.calc_nll(sess) if self._val_model is not None else (None, None)
        nll_test_set_is, nll_test_set_hais = self._test_model.calc_nll(sess) if self._test_model is not None else (None, None)

        log("\nneg. ELBO training set:", elbo_train_set)
        if self._val_model is not None:
            log("neg. ELBO validation set:", elbo_val_set)
        if self._test_model is not None:
            log("neg. ELBO test set:", elbo_test_set)

        if self._val_model is not None:
            log("NLL (standard importance sampling via approx. posterior) validation set:", nll_val_set_is)
        if self._test_model is not None:
            log("NLL (standard importance sampling via approx. posterior) test set:", nll_test_set_is)

        if self._val_model is not None and nll_val_set_hais is not None:
            log("NLL (Hamiltonian annealed importance sampling) validation set:", nll_val_set_hais)
        if self._test_model is not None and nll_test_set_hais is not None:
            log("NLL (Hamiltonian annealed importance sampling) test set:", nll_test_set_hais)

    def draw_samples_from_model(self, sess):
        import matplotlib.pyplot as plt
        log("\nDrawing image/data samples from model")
        log("Saving into: logs/"  + self._name + "_model_samples.png")
        new_image_batch = np.concatenate([self._train_model.generate_new_images(sess) for _ in range(100)], axis=0)
        images = np.reshape(new_image_batch, [-1] + self._dataset.get_image_shape())
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(8, 8))
        columns = 10
        rows = 10
        for i in range(1, columns * rows + 1):
            img = images[i-1]
            fig.add_subplot(rows, columns, i)
            plt.imshow(img, cmap='gray')
            plt.axis('off')
        plt.savefig('logs/' + self._name + '_model_samples.png', bbox_inches='tight', dpi=400)
        plt.clf()

    def draw_samples_from_data(self, sess):
        if self._dataset.get_batchsize("test") >= 100:
            import matplotlib.pyplot as plt
            log("\nDrawing samples from raw data (just for comparison)")
            log("Saving into: logs/" + self._name + "_data_samples.png")
            image_batch = self._test_model.get_data_images(sess)
            images = np.reshape(image_batch, [-1] + self._dataset.get_image_shape())
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(8, 8))
            columns = 10
            rows = 10
            for i in range(1, columns * rows + 1):
                img = images[i-1]
                fig.add_subplot(rows, columns, i)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            plt.savefig('logs/' + self._name + '_data_samples.png', bbox_inches='tight', dpi=400)
            plt.clf()
        else:
            log("Not drawing samples from data. Batchsize too small. Need at least 100 samples.")

    def compare_input_reconst(self, sess):
        if self._dataset.get_batchsize("test") >= 100:
            import matplotlib.pyplot as plt
            log("\nComparing inputs with reconstructions")
            log("Saving into: logs/" + self._name + "_reconstructions.png")
            image_batch, reconst_batch = self._test_model.get_input_and_reconst(sess)
            images = np.reshape(image_batch, [-1] + self._dataset.get_image_shape())
            reconsts = np.reshape(reconst_batch, [-1] + self._dataset.get_image_shape())
            plt.style.use('dark_background')
            fig = plt.figure(figsize=(16, 8))
            columns = 21
            rows = 10
            imagecounter = 0
            reconstcounter = 0
            for i in range(0, columns * rows):
                if i % 21 < 10:
                    img = images[imagecounter]
                    imagecounter += 1
                elif i % 21 > 10:
                    img = reconsts[reconstcounter]
                    reconstcounter += 1
                else:
                    img = np.zeros(shape=self._dataset.get_image_shape())
                fig.add_subplot(rows, columns, i+1)
                plt.imshow(img, cmap='gray')
                plt.axis('off')
            plt.savefig('logs/' + self._name + '_reconstructions.png', bbox_inches='tight', dpi=400)
            plt.clf()
        else:
            log("Not visualizing reconstructions. Batchsize too small. Need at least 100 samples.")
