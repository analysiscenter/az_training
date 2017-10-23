""" Custom class for MNIST classifier CNN
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..")
from dataset import Batch, action, model, inbatch_parallel
from dataset.dataset.image import ImagesBatch
from dataset.dataset.models.tf import TFModel

class MyMnistModel(TFModel):
    def _build(self, *args, **kwargs):
        x = tf.placeholder(tf.float32, [None, 28, 28], name='input_images')
        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])        
        

        training = self.is_training
        net = tf.layers.conv2d(x_as_pics, filters=4, kernel_size=(7,7), strides=(1, 1), padding='same')
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

        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        net = tf.nn.dropout(net, keep_prob)


        net = tf.layers.dense(net, 128, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        net = tf.nn.relu(net)
        
        net = tf.layers.dense(net, 10, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        predictions = tf.identity(net, name='predictions')
        probs = tf.nn.softmax(logits=net, name='predicted_prob')

        y_ = tf.placeholder(tf.float32, [None, 10], name='input_labels')

        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')	
        
        global_step = tf.Variable(0, trainable=False)
        targets = tf.identity(y_, name='targets')
        
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32)
        predicted_labels = tf.identity(labels_hat, name='predicted_labels')

        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32)
        true_labels = tf.identity(labels, name='true_labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')



        # print('VARS ', tf.get_collection(tf.GraphKeys.VARIABLES))