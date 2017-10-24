""" New convolution model """

import sys

import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv2d_block


class ConvMnist(TFModel):
    """ Class to build convomodel """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        x = tf.placeholder(tf.float32, [None, 28, 28], name='input')
        keep_prob = tf.placeholder(tf.float32, name='dropout_rate')

        training = self.is_training
        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])
        net = conv2d_block(x_as_pics, filters=16, kernel_size=(7, 7), layout='cpna', strides=2, pool_size=(4, 4),
                           pool_strides=(2, 2))
        net = conv2d_block(net, filters=32, kernel_size=(5, 5), layout='cpna', pool_size=(3, 3),
                           pool_strides=(2, 2))
        net = conv2d_block(net, filters=64, kernel_size=(3, 3), layout='cpna', pool_size=(2, 2),
                           pool_strides=(1, 1), dropout_rate=keep_prob, is_training=training)

        net = tf.contrib.layers.flatten(net)
        net = tf.contrib.layers.fully_connected(net, 128)
        net = tf.layers.dropout(net, keep_prob, training=training)
        net = tf.contrib.layers.fully_connected(net, 10)
        tf.identity(net, name='predictions')

        y_ = tf.placeholder(tf.int32, [None, 10], name='targets')
        tf.nn.softmax(net, name='prob')

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')
