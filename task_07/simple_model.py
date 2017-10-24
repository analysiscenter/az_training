""" File with simple model example of convolution network """
import sys

import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block

class ConvModel(TFModel):
    """ Class to build conv model """
    def _build(self, *args, **kwargs):
        _ = args, kwargs
        input_images = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input')
        input_labels = tf.placeholder(tf.int32, [None], name='labels')

        net = conv_block(2, input_images, filters=16, kernel_size=(7, 7), strides=2, layout='cpa', pool_size=(4, 4),
                         pool_strides=(2, 2))
        net = conv_block(2, net, filters=32, kernel_size=(5, 5), layout='cpa', pool_size=(3, 3),
                         pool_strides=(2, 2))

        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, units=256, activation=tf.nn.relu)
        net = tf.layers.dense(net, 10)
        tf.identity(net, name='predictions')

        encoded_labels = tf.one_hot(input_labels, depth=10, name='targets')

        prediction = tf.argmax(net, 1)
        correct_prediction = tf.equal(prediction, tf.argmax(encoded_labels, 1))
        tf.reduce_mean(tf.cast(correct_prediction, 'float'), name='accuracy')
