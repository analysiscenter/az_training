""" Custom class for MNIST classifier CNN
"""
import sys
import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..")
from dataset import Batch, action, model, inbatch_parallel
from dataset import ImagesBatch
from dataset.dataset.models.tf import TFModel

class MyMnistModel(TFModel):
    @classmethod
    def default_config(cls):
        """ Specify default value for dropout """
        config = TFModel.default_config()
        config['body']['keep_prob'] = 0.1
        return config

    @classmethod
    def body(cls, inputs, **kwargs):
        """ CNN model

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor

        Returns
        -------
        net : tf.Tensor
            output of the dense layer.
        """

        kwargs = cls.fill_params('body', **kwargs)
        print('kwargs ', kwargs)
        keep_prob = cls.pop('keep_prob', kwargs)
        training = cls.pop('is_training', kwargs)

        net = tf.layers.conv2d(inputs, filters=4, kernel_size=(7, 7), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(6, 6), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(net, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(5, 5), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)


        net = tf.layers.conv2d(net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)


        net = tf.contrib.layers.flatten(net)

        net = tf.layers.dropout(net, keep_prob, training=training)


        net = tf.layers.dense(net, 128, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        net = tf.nn.relu(net)

        net = tf.layers.dense(net, 10, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        tf.nn.softmax(logits=net, name='predicted_prob')

        y_target = tf.get_default_graph().get_tensor_by_name('MyMnistModel/inputs/targets:0')
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_target, name='loss')

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   150, 0.85, staircase=True)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            cls.train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,
                                                                                       global_step=global_step)
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.int64)
        tf.identity(labels_hat, name='predicted_labels')
        return net
