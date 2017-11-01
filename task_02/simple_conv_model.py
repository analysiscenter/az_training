""" File with convolution network """

import sys

import tensorflow as tf

sys.path.append('..')
from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block


class ConvModel(TFModel):
    """ Class to build conv model """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        keep_prob = tf.placeholder(tf.float32, name='dropout_rate')
        training = self.is_training

        image = tf.placeholder(tf.float32, [None, 28, 28], name='input')
        image = tf.reshape(image, shape=[-1, 28, 28, 1])
        net = conv_block(2, image, filters=16, kernel_size=(7, 7), layout='cpna', pool_size=(3, 3),
                         pool_strides=(2, 2), is_training=training)
        net = conv_block(2, net, filters=32, kernel_size=(5, 5), layout='cpna', pool_size=(3, 3),
                         pool_strides=(2, 2), is_training=training)
        net = conv_block(2, net, filters=64, kernel_size=(3, 3), layout='cpna', pool_size=(2, 2),
                         pool_strides=(1, 1), dropout_rate=keep_prob, is_training=training)

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
