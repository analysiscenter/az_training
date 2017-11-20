""" Custom class for Regression models
"""
import sys

import tensorflow as tf

sys.path.append("../..")
from dataset.dataset.models.tf import TFModel

class RegressionModel(TFModel):
    """ Regression model

    ** Configuration **

    inputs : dict
        dict with keys, for example 'features' and 'labels' (see :meth:`._make_inputs`)

    body : dict
        dimension : int
            dimension of features
        model_type : linear, logistic or poisson
            type of regression model

    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body']['model_type'] = 'linear'
        return config

    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        kwargs = cls.fill_params('body', **kwargs)
        model_type = cls.pop('model_type', kwargs)
        print('model_type = {}'.format(model_type))
        dim_shape = cls.pop('dimension', kwargs)

        weights = tf.Variable(tf.ones([dim_shape, 1]), name='weights')
        bias = tf.Variable(tf.ones([1]), name='bias')
        wx_b = tf.add(tf.matmul(inputs, weights), bias)


        if model_type == 'logistic':
            logit = tf.sign(wx_b, name='predictions')
            y_cup = tf.sign(wx_b, name='predicted_value')
            y_target = tf.get_default_graph().get_tensor_by_name('RegressionModel/inputs/targets:0')
            margin = tf.multiply(y_target, wx_b)
            cost = tf.reduce_mean(tf.add(tf.log(tf.add(tf.ones([1]), tf.exp(-margin))), \
                                  tf.multiply(tf.reduce_sum(tf.square(weights)), 0.1)))
            tf.losses.add_loss(cost)

            y_cup_int = tf.cast(y_cup, tf.int32)
            y_target_int = tf.cast(y_target, tf.int32)
            accy = tf.contrib.metrics.accuracy(y_cup_int, y_target_int, name='accuracy_0')
            accuracy = tf.identity(accy, name='accuracy')

        elif model_type == 'poisson':
            pass

        elif model_type == 'linear':
            logit = tf.identity(wx_b, name='predictions')
            y_cup = tf.identity(logit, name='predicted_value')
            y_target = tf.get_default_graph().get_tensor_by_name('RegressionModel/inputs/targets:0')
            mse = tf.reduce_mean(tf.square(y_cup - y_target), name='mse')


        else:
            print('you assigned model_type as %s ' % model_type,
                  'while model_type must be linear, logistic or poisson')
            raise Exception

        return wx_b
