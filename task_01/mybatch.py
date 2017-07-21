"""File contains main class named myBatch.
And function to generate data."""
import sys

import tensorflow as tf
import numpy as np

sys.path.append('..')
from dataset import Batch, action, model


def generate_linear_data(size=10):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.

    Output:
        x: array [0..size]
        y: array [0..size] with some random noize. """
    x = np.random.randint(-10, 10, size)
    w = np.random.random(1)
    error = np.random.randint(-3, 3, size)
    y = x * w + error

    return x, y

def generate_logistic_data(size=10):
    """ Generation of data for fit logistic regression.
    Args:
        size: length of data.

    Output:
        x: random numbers from the range of -10 to 10
        y: array of 1 or 0. if x[i] < 0 y[i] = 0 else y[i] = 1 """
    x = np.array(np.random.randint(-100, 100, size), dtype=np.float32)
    y = np.sign(x)

    return x, y

def generate_poisson_data(lam, size=10):
    """ Generation of data for fit poisson regression.

    size: size of data.

    lambd: Poisson distribution parameter.

    Output:
        y: array of poisson distribution numbers.
        x: matrix with shape(size,3) with random numbers of uniform distribution. """
    x = np.random.random(size * 3).reshape(size, 3)
    y = np.random.poisson(np.exp(np.dot(x, lam)))

    return x, y

def generate(size=10, ttype='linear', lam=np.array([0, 0, 0])):

    """ Generate data for thee types of regression.
    Args:
        ttype: name of using algorithm:
            * 'linear' for linear regression. (default)
            * 'logistic' for logistic regression.
            * 'poisson' for poisson regression.

        lam: array with lambda's as parameters of poisson distribution.

    Output:
        self """
    data_dict = {'linear': generate_linear_data,
                 'logistic': generate_logistic_data,
                 'poisson': generate_poisson_data}
    if ttype != 'poisson':
        x, y = data_dict[ttype](size)
    else:
        x, y = data_dict[ttype](lam, size)  # pylint: disable=too-many-function-args
    return x, y

class MyBatch(Batch):
    """ The core batch class which can load data, generate dataset
    and train linear regression, logistic regression, poisson regression"""

    def __init__(self, index, *args, **kwargs):
        """ Initialization of variable from parent class - Batch. """
        super().__init__(index, *args, **kwargs)

    @property
    def components(self):
        """ Define componentis. """
        return 'x', 'y'

    @model()
    def linear_model():
        """ Function with graph of linear regression.

        Output:
            array with shape = (3,2)
            x: data.
            y: answers to data.
            train: function - optimizer.
            loss: quality of model.
            w: slope coefficient of straight line.
            b: bias. """
        x = tf.placeholder(name='input', dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1, 1), dtype=tf.float32)

        predict = tf.add(tf.multiply(w, x, name='output'), b)
        loss = tf.reduce_mean(tf.square(predict - y))

        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.007)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss], [w, b]]

    @action(model='linear_model')
    def train_linear_model(self, models, session, dict_params):
        """Train linear regression.

        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

        Output:
            self. """
        x, y = models[0]
        optimizer, cost = models[1]
        weights = models[2]
        _, loss, params = session.run([optimizer, cost, weights], feed_dict={x: self.x, y: self.y})
        dict_params['w'] = params[0][0]
        dict_params['b'] = params[1]
        dict_params['loss'] = loss

        return self

    @model()
    def logistic_model():
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
        x = tf.placeholder(name='input', dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1, 1), dtype=tf.float32)

        predict = tf.sigmoid(tf.add(tf.multiply(w, x, name='output'), b))
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))

        optimize = tf.train.AdamOptimizer(learning_rate=0.005)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w, b]]

    @action(model='logistic_model')
    def train_logistic_model(self, models, session, dict_params):
        """ Train logistic regression.
        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

            result: result of prediction.

            test: data to predict.

        Output:
            self. """
        x, y = models[0]
        optimizer, cost, predict = models[1]
        weight = models[2]
        _, loss, params = session.run([optimizer, cost, weight], feed_dict={x: self.x, y: self.y})
        dict_params['w'] = params[0][0]
        dict_params['b'] = params[1]
        dict_params['loss'] = loss
        dict_params['result'] = session.run([predict], feed_dict={x: dict_params['test']})[0]

        return self

    @model()
    def poisson_model():
        """ Function with graph of poisson regression.

        Output:
            array with shape = (3,2(3))
            x: data.
            y: answers to data.
            train: function - optimizer.
            loss: quality of model.
            predict: model prediction.
            w: array of weights."""
        x = tf.placeholder(name='input', shape=[None, 3], dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=3).reshape(3, 1), name='weight', dtype=tf.float32)

        predict = tf.matmul(x, w)
        loss = tf.reduce_sum(tf.nn.log_poisson_loss(y, predict))

        optimize = tf.train.AdamOptimizer(learning_rate=0.005)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w]]

    @action(model='poisson_model')
    def train_poisson_model(self, models, session, dict_params):
        """ Train poisson regression.
        Args:
            model: fit funtion. In this case it's linear_model.

            session: tensorflow session.

        Output
            self. """
        x, y = models[0]
        optimizer, cost, predict = models[1]
        weight = models[2]
        _, loss, params = session.run([optimizer, cost, weight], feed_dict={x: self.x, y: self.y})
        dict_params['w'] = params[0]
        dict_params['loss'] = loss
        dict_params['result'] = session.run([predict], feed_dict={x: dict_params['test']})

        return self
