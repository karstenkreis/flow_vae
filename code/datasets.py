from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow_datasets as tfds
import tensorflow as tf
from code.logger import log


class Datasets(object):

    def __init__(self, batchsize_train, batchsize_val, batchsize_test, trainsize, valsize, testsize):
        log("Building dataset: mnist")
        self._use_validation_set = valsize > 0
        self._use_test_set = testsize > 0

        self._batchsize_train = batchsize_train
        self._batchsize_val = batchsize_val
        self._batchsize_test = batchsize_test

        self._trainsize = None
        self._valsize = None
        self._testsize = None

        self._density_function, self._raw_np_data = None, None

        dataset_train, dataset_val, dataset_test = self._autogen_datasets_from_tf_datasets(
            trainsize, valsize, testsize, batchsize_train, batchsize_val, batchsize_test)

        self._steps_per_epoch_train = int(self._trainsize/batchsize_train)
        dataset_train = dataset_train.shuffle(buffer_size=trainsize)
        dataset_train = dataset_train.batch(batch_size=batchsize_train, drop_remainder=True)
        dataset_train = dataset_train.repeat()
        dataset_train = dataset_train.prefetch(1)
        self._train_iterator = dataset_train.make_one_shot_iterator()

        if self._use_validation_set:
            self._steps_per_epoch_val = int(self._valsize / batchsize_val)
            dataset_val = dataset_val.shuffle(buffer_size=valsize)
            dataset_val = dataset_val.batch(batch_size=batchsize_val)
            dataset_val = dataset_val.repeat()
            dataset_val = dataset_val.prefetch(1)
            self._val_iterator = dataset_val.make_one_shot_iterator()

        if self._use_test_set:
            self._steps_per_epoch_test = int(self._testsize / batchsize_test)
            dataset_test = dataset_test.shuffle(buffer_size=testsize)
            dataset_test = dataset_test.batch(batch_size=batchsize_test)
            dataset_test = dataset_test.repeat()
            dataset_test = dataset_test.prefetch(1)
            self._test_iterator = dataset_test.make_one_shot_iterator()

    ### Internal class functions ###

    def _autogen_datasets_from_tf_datasets(self, trainsize, valsize, testsize, batchsize_train, batchsize_val,
                                           batchsize_test):
        log("Ignoring testsize, since dataset comes with fixed train/val and test data splits.")

        assert trainsize + valsize == 60000
        total_number_images = 70000
        assert trainsize % batchsize_train == 0

        if self._use_validation_set:
            assert valsize % batchsize_val == 0
            full_dataset = tfds.load("mnist", data_dir=None, split=tfds.Split.TRAIN).shuffle(buffer_size=60000)
            dataset_train = full_dataset.take(trainsize)
            dataset_val = full_dataset.skip(trainsize)
        else:
            dataset_train = tfds.load("mnist", data_dir=None, split=tfds.Split.TRAIN)
            dataset_val = None

        if self._use_test_set:
            assert testsize % batchsize_test == 0
            dataset_test = tfds.load("mnist", data_dir=None, split=tfds.Split.TEST)
        else:
            dataset_test = None

        self._trainsize = trainsize
        self._valsize = valsize
        self._testsize = testsize

        log("Total number of images in dataset:", total_number_images)
        log("Number of training images:", self._trainsize)
        log("Number of validation images:", self._valsize)
        log("Number of test images:", self._testsize)
        assert self._trainsize + self._valsize + self._testsize == total_number_images

        return dataset_train, dataset_val, dataset_test

    ### Functions for interaction with external code that is not part of this class ###

    def get_batch(self, which_dataset):
        assert which_dataset in ["train", "val", "test"]

        # get the raw batch
        if which_dataset == "train":
            raw_batch = self._train_iterator.get_next()
        elif which_dataset == "val":
            assert self._use_validation_set
            raw_batch = self._val_iterator.get_next()
        elif which_dataset == "test":
            assert self._use_test_set
            raw_batch = self._test_iterator.get_next()
        else:
            raise ValueError("which_dataset string not recognized:", which_dataset)

        images = raw_batch['image']
        labels = raw_batch['label'] if 'label' in raw_batch else None

        return images, labels

    def get_batchsize(self, which_dataset):
        assert which_dataset in ["train", "val", "test"]

        if which_dataset == "train":
            return self._batchsize_train
        elif which_dataset == "val":
            assert self._use_validation_set
            return self._batchsize_val
        elif which_dataset == "test":
            assert self._use_test_set
            return self._batchsize_test
        else:
            raise ValueError("which_dataset string not recognized:", which_dataset)

    @staticmethod
    def get_image_shape():
        return [28, 28]

    def get_steps_per_epoch(self, which_dataset):
        assert which_dataset in ["train", "val", "test"]

        if which_dataset == "train":
            return self._steps_per_epoch_train
        elif which_dataset == "val":
            assert self._use_validation_set
            return self._steps_per_epoch_val
        elif which_dataset == "test":
            assert self._use_test_set
            return self._steps_per_epoch_test
        else:
            raise ValueError("which_dataset string not recognized:", which_dataset)

    def using_validation_set(self):
        return self._use_validation_set

    def using_test_set(self):
        return self._use_test_set

    def get_normalization(self):
        """The normalization values come from the train set. The fixed test set was not used for calculation."""
        return [tf.constant(0.13015, shape=[1, 1, 1, 1]), tf.constant(0.3069, shape=[1, 1, 1, 1]), 256.0]

