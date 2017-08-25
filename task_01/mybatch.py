"""File contains main class named myBatch.
And function to generate data."""
import sys

import tensorflow as tf
import numpy as np

sys.path.append('..')
from dataset import Batch, action, model

NUM_DIM_LIN = 13
class MyBatch(Batch):
    """ The core batch class which can load data, generate dataset
    and train linear regression, logistic regression, poisson regression"""

    def __init__(self, index, *args, **kwargs):
        """ INIT """
        super().__init__(index, *args, **kwargs)
        self.x = None
        self.y = None

    @property
    def components(self):
        """ Define componentis. """
        return 'x', 'y'

    @action
    def load(self, src, fmt='blosc'):
        """ Loading data to self.x and self.y
        Args:
            * src: data in format (x, y)
            * fmt: format file
        Output:
            self """
        self.x = src[0][self.indices].reshape(-1, src[0].shape[1])
        self.y = src[1][self.indices].reshape(-1, 1)
        return self
    @action
    def train(self, models, session, dict_params):
        """ Train funcion for all types of regressions.

        Args:
            model: fit funtion.

            session: tensorflow session.

            dict_params: parameters of model.

        Outpt:
            self. """

        x, y = models[0]
        optimizer, cost, _ = models[1]
        weights = models[2]
        _, loss, params = session.run([optimizer, cost, weights], feed_dict={x: self.x, y: self.y})
        dict_params['w'].append(params[0])
        dict_params['b'].append(params[1])
        dict_params['loss'].append(loss)

        return self
    @action
    def predict(self, models, session, predict):
        """ Predict for all models
        Args:
            session: tf Session
            predict: list to save prediction
        Output:
            self """
        x, _ = models[0]
        pred = models[1][-1]
        predict.append(session.run([pred], feed_dict={x: self.x}))
        return self

############## linear regression ##############
    @action(model='linear_regression')
    def predict_linear(self, models, session, predict):
        """ Predict for linear regression
        Args:
            session: tf Session
            predict: list to save prediction
        Output:
            self """
        self.predict(models, session, predict)
        return self

    @action(model='linear_regression')
    def train_linear_model(self, models, session, dict_params):
        """Train linear regression.

        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

        Output:
            self. """
        self.train(models, session, dict_params)
        return self

    @model()
    def linear_regression():
        """ Function with graph of linear regression.

        Output:
            array with shape = (3,2)
            x: data.
            y: answers to data.
            train: function - optimizer.
            loss: quality of model.
            w: slope coefficient of straight line.
            b: bias. """
        x = tf.placeholder(name='input', dtype=tf.float32, shape=[None, NUM_DIM_LIN])
        y = tf.placeholder(name='true_y', dtype=tf.float32, shape=[None, 1])

        w = tf.Variable(np.random.uniform(-1, 1, size=NUM_DIM_LIN).reshape(-1, 1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.uniform(-1, 1, size=1).reshape(-1, 1), dtype=tf.float32)

        predict = tf.add(tf.matmul(x, w, name='output'), b)
        loss = tf.add(tf.reduce_mean(tf.square(predict - y)), tf.multiply(tf.reduce_sum(tf.square(w)), 0.1))

        optimize = tf.train.AdamOptimizer(learning_rate=0.2)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w, b]]

############## logistic regression ##############
    @action(model='logistic_regression')
    def predict_logistic(self, models, session, predict):
        """ Predict for logistic regression
        Args:
            session: tf Session
            predict: list to save prediction
        Output:
            self """
        self.predict(models, session, predict)
        return self

    @action(model='logistic_regression')
    def train_logistic_model(self, models, session, dict_params):
        """ Train logistic regression.
        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

            result: result of prediction.

            test: data to predict.

        Output:
            self. """
        self.train(models, session, dict_params)
        return self

    @model()
    def logistic_regression():
        """ Function with graph of logistic regression.

        Output:
            array with shape = (3,2(3))
            x: data.
            y: answers to data.
            train: function - optimizer.
            loss: quality of model.
            predict: model prediction.
            w: slope coefficient of straight line.
            b: bias. """
        x = tf.placeholder(name='input', dtype=tf.float32, shape=[None, 2])
        y = tf.placeholder(name='true_y', dtype=tf.float32, shape=[None, 1])

        w = tf.Variable(tf.random_uniform(shape=(2, 1), minval=-1, maxval=1, name='weight', dtype=tf.float32))
        b = tf.Variable(tf.random_uniform(shape=(1, 1), minval=-1, maxval=1, dtype=tf.float32))
        # w = tf.Variable(tf.zeros([2,1]))
        # b = tf.Variable(tf.zeros([1,1]))
        mul = tf.matmul(x, w, name='output')#.shape()
        mul = tf.add(mul, b)
        predict = tf.sigmoid(mul)
        loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=mul))

        optimize = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w, b], [mul]]

############## logistic regression ##############
    @action(model='poisson_regression')
    def predict_poisson(self, models, session, predict):
        """ Predict for logistic regression
        Args:
            session: tf Session
            predict: list to save prediction
        Output:
            self """
        self.predict(models, session, predict)
        return self

    @action(model='poisson_regression')
    def train_poisson_model(self, models, session, dict_params):
        """ Train logistic regression.
        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

            result: result of prediction.

            test: data to predict.

        Output:
            self. """
        self.train(models, session, dict_params)
        return self

    @model()
    def poisson_regression():
        """ Function with graph of poisson regression.

        Output:
            array with shape = (3,2(3))
            x: data.
            y: answers to data.
            train: function - optimizer.
            loss: quality of model.
            predict: model prediction.
            w: array of weights."""
        x = tf.placeholder(name='input', shape=[None, NUM_DIM_LIN], dtype=tf.float32)
        y = tf.placeholder(name='true_y', shape=[None, 1], dtype=tf.float32)

        w = tf.Variable(np.random.uniform(-1, 1, size=NUM_DIM_LIN)\
            .reshape(NUM_DIM_LIN, 1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.uniform(-1, 1, size=1).reshape(1, 1), dtype=tf.float32)
        predict = tf.add(tf.matmul(x, w), b)
        y_pred = tf.exp(predict)
        loss = tf.reduce_mean(tf.nn.log_poisson_loss(y, predict, compute_full_loss=False))

        optimize = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, y_pred], [w, b]]
