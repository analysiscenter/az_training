"""File contains classes with main regression alghoritms"""
# pylint: disable=too-few-public-methods

import sys

import tensorflow as tf

sys.path.append('../dataset')
from dataset.models.tf import TFModel
from dataset import Batch, action

NUM_DIM_LIN = 13
class InitBatch(Batch):
    """ Simple batch class with load function and some components """
    def __init__(self, index, *args, **kwargs):
        """ INIT """
        super().__init__(index, *args, **kwargs)
        self.input_tensor = None
        self.labels = None

    @property
    def components(self):
        """ Define components. """
        return 'input_tensor', 'labels'

    @action
    def load(self, src, fmt='blosc', components=None, *args, **kwargs):
        _ = args, kwargs, components
        """ Loading data to self.x and self.y
        Args:
            * src: data in format (x, y)
            * fmt: format file
        Output:
            self """
        self.input_tensor = src[0][self.indices].reshape(-1, src[0].shape[1])
        self.labels = src[1][self.indices].reshape(-1, 1)
        return self

class LinearRegression(TFModel):
    """ Class with logistic regression model """
    def _build(self, *args, **kwargs):
        """function to build logistic regression """
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1, name='dense')
        tf.identity(dense, name='predictions')


class LogisticRegression(TFModel):
    """ Class with Linear regression model """
    def _build(self, *args, **kwargs):
        """function to build Linear regression """
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1, name='dense')

        predictions = tf.identity(dense, name='predictions')
        tf.sigmoid(predictions, 'predicted_labels')

class PoissonRegression(TFModel):
    """ Class with Poisson regression model """
    def _build(self, *args, **kwargs):
        """function to build Poisson regression """
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1)
        predictions = tf.identity(dense, name='predictions')

        tf.cast(tf.exp(predictions), tf.int32, name='predicted_labels')
