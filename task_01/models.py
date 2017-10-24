"""File contains classes with main regression alghoritms"""
import sys

import tensorflow as tf

sys.path.append('..')
from dataset.dataset.models.tf import TFModel

class LinearRegression(TFModel):
    """ Class with logistic regression model """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)

        # we create 'targets' tensor to use it in loss computation
        tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1, name='dense')
        tf.identity(dense, name='predictions')


class LogisticRegression(TFModel):
    """ Class with Linear regression model """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)

        tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1, name='dense')
        predictions = tf.identity(dense, name='predictions')

        tf.sigmoid(predictions, name='predicted_labels')

class PoissonRegression(TFModel):
    """ Class with Poisson regression model """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        data_shape = [None] + list(self.get_from_config('data_shape'))
        input_tensor = tf.placeholder(name='input_tensor', dtype=tf.float32, shape=data_shape)

        tf.placeholder(name='targets', dtype=tf.float32, shape=[None, 1])

        dense = tf.layers.dense(input_tensor, 1)
        predictions = tf.identity(dense, name='predictions')

        tf.cast(tf.exp(predictions), tf.int32, name='predicted_labels')
