""" Custom class for Regression models
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("../..")
from dataset import Batch, action, model, inbatch_parallel
from dataset.dataset.image import ImagesBatch
from dataset.dataset.models.tf import TFModel

class RegressionModel(TFModel):
    def _build(self, *args, **kwargs):
        model_type = self.get_from_config('model_type', 'linear')
        print(model_type)

        dim_shape = self.get_from_config('dimension', 0)
        x_features = tf.placeholder(tf.float32, [None, None], 'input_features')
        y_target = tf.placeholder(tf.float32, [None, 1], name='targets')
        
        weights = tf.Variable(tf.ones([dim_shape, 1]), name='weights')

        bias = tf.Variable(tf.ones([1]), name='bias')

        wx_b = tf.add(tf.matmul(x_features, weights), bias)


        if model_type == 'logistic':
            logit = tf.sign(wx_b, name='predictions')
            y_cup = tf.sign(wx_b, name='predicted_value')

            margin = tf.multiply(y_target, wx_b)
            cost = tf.reduce_mean(tf.add(tf.log(tf.add(tf.ones([1]), tf.exp(-margin))), \
                tf.multiply(tf.reduce_sum(tf.square(weights)), 0.1)))
            tf.losses.add_loss(cost)
            y_cup_int = tf.cast(y_cup, tf.int32)
            y_target_int = tf.cast(y_target, tf.int32)
            accuracy = tf.contrib.metrics.accuracy(y_cup_int, y_target_int, name='acc_0') 
            accy = tf.identity(accuracy, name='accuracy')



        elif model_type == 'poisson':
            logit = tf.identity(wx_b, name='predictions')
            y_cup = tf.cast(tf.exp(logit), tf.int32, name='predicted_value')

            cost = tf.reduce_mean(tf.add(tf.nn.log_poisson_loss(y_target, logit, compute_full_loss=False), \
                tf.multiply(tf.reduce_sum(tf.square(weights)), 0.01)))
            tf.losses.add_loss(cost)


        elif model_type == 'linear':
            logit = tf.identity(wx_b, name='predictions')
            y_cup = tf.identity(logit, name='predicted_value')
            mse = tf.reduce_mean(tf.square(y_cup - y_target), name='mse')


        else:
            print('you assigned model_type as %s ' % model_type, 'while model_type must be linear, logistic or poisson')
            raise Exception

