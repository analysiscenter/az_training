""" Custom class for MNIST classifier CNN
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..//..")
from dataset import Batch, action, model, inbatch_parallel
from dataset import ImagesBatch
from dataset.dataset.models.tf import TFModel


class MyMnistModel(TFModel):
    def _build_config(self, names=None):
        names = names if names else ['images', 'labels']
        config = super()._build_config(names)

        config['input_block']['inputs'] = self.inputs['images']
        config['body']['smth'] = 'smth'
        return config


    @classmethod
    def body(cls, inputs, smth, *args, **kwargs):
        print('start ', inputs.get_shape().as_list())

        # x = tf.placeholder(tf.float32, [None, 28, 28], name='input_images')
        # x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])        
        

        # training = self.is_training
        net = tf.layers.conv2d(inputs, filters=4, kernel_size=(7,7), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(6, 6), strides=(2, 2), padding='same')
        # net = tf.layers.batch_normalization(net, training=training)
        # net = tf.nn.relu(net)
        
        net = tf.layers.conv2d(net, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(5, 5), strides=(2, 2), padding='same')
        # net = tf.layers.batch_normalization(net, training=training)
        # net = tf.nn.relu(net)


        net = tf.layers.conv2d(net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        # net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)

        
        net = tf.contrib.layers.flatten(net)

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        net = tf.nn.dropout(net, keep_prob)

        print('before ', net.get_shape().as_list())
        net = tf.layers.dense(net, 128, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))
        print('before ', net.get_shape().as_list())

        net = tf.nn.relu(net)
        
        net = tf.layers.dense(net, 10, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        return net