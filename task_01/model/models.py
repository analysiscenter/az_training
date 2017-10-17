"""File contains classes with main regression alghoritms"""
# pylint: disable=too-few-public-methods
# pylint: disable=unused-variable
# pylint: disable=unused-argument
import sys

import tensorflow as tf

sys.path.append('../..')
sys.path.append('')
from dataset.dataset.models.tf import TFModel
from dataset import Batch, action

NUM_DIM_LIN = 13
class InitBatch(Batch):
    """ Simple batch class with load function and some components """
    def __init__(self, index, *args, **kwargs):
        """ INIT """
        super().__init__(index, *args, **kwargs)
        self.input_data = None
        self.labels = None

    @property
    def components(self):
        """ Define components. """
        return 'input_data', 'labels'

    @action
    def load(self, src, fmt='blosc', components=None, *args, **kwargs):
        """ Loading data to self.x and self.y
        Args:
            * src: data in format (x, y)
            * fmt: format file
        Output:
            self """
        self.input_data = src[0][self.indices].reshape(-1, src[0].shape[1])
        self.labels = src[1][self.indices].reshape(-1, 1)
        return self

class LinearRegression(TFModel):
    """ Class with logistic regression model """
    def _build(self, *args, **kwargs):
        """function to build logistic regression """
        data_shape = [None] + [13]
        input_data = tf.placeholder(name='input_data', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_data, 1, name='dense')
        predictions = tf.identity(dense, name='predictions')


class LogisticRegression(TFModel):
    """ Class with Linear regression model """
    def _build(self, *args, **kwargs):
        """function to build Linear regression """
        data_shape = [None] + [2]
        input_data = tf.placeholder(name='input_data', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_data, 1, name='dense')

        predictions = tf.identity(dense, name='predictions')
        predicted_labels = tf.sigmoid(predictions)

class PoissonRegression(TFModel):
    """ Class with Poisson regression model """
    def _build(self, *args, **kwargs):
        """function to build Poisson regression """
        data_shape = [None] + [13]
        input_data = tf.placeholder(name='input_data', dtype=tf.float32, shape=data_shape)
        targets = tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_data, 1)
        predictions = tf.identity(dense, name='predictions')

        predicted_labels = tf.cast(tf.exp(predictions), tf.int32, name='predicted_labels')
