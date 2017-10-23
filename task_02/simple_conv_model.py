""" File with convolution network """
import sys

import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv2d_block


class ConvModel(TFModel):
    """ Simple convolution model """

    def _build(self, *args, **kwargs):
        _ = args, kwargs
        image = tf.placeholder(tf.float32, [None, 784], 'input')
        image = tf.reshape(image, shape=[-1, 28, 28, 1])

        net = conv2d_block(image, filters=4, kernel_size=(7, 7), layout='cpa', pool_size=(6, 6),
                           pool_strides=(2, 2))
        net = conv2d_block(net, filters=16, kernel_size=(5, 5), layout='cpa', pool_size=(5, 5),
                           pool_strides=(2, 2))
        net = conv2d_block(net, filters=32, kernel_size=(3, 3), layout='cpa', pool_size=(2, 2),
                           pool_strides=(2, 2))

        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 128)
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, 10)
        tf.identity(net, name='predictions')

        targets = tf.placeholder(tf.float32, [None, 10], name='targets')
        prob = tf.nn.softmax(net, name='prob_predictions')

        labels_hat = tf.cast(tf.argmax(prob, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(targets, axis=1), tf.float32, name='labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')
