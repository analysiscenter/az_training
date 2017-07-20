"""MyBatch"""
import sys
sys.path.append('..')
import numpy as np
import tensorflow as tf
from dataset import Batch, action, model

def generate_linear_data(lenght=10):
    """
    Generation of data for fit linear regression.

    lenght - lenght of data.

    return:
    x - array [0..lenght]
    y - array [0..lenght] with some random noize
    """
    y = np.linspace(0, 10, lenght)
    x = y + np.random.random(lenght) - 0.5

    return x, y

def generate_logistic_data(lenght=10):
    """
    Generation of data for fit logistic regression.

    lenght - lenght of data.

    return:
        x - random numbers from the range of -10 to 10
    y - array of 1 or 0. if x[i] < 0 y[i] = 0 else y[i] = 1
    """
    x = np.array(np.random.randint(-10, 10, lenght), dtype=np.float32)
    y = np.array([1. if i > 0 else 0. for i in x])

    return x, y

def generate_poisson_data(lambd, lenght=10):
    """
    Generation of data for fit poisson regression.

    lenght - lenght of data.

    lambd - Poisson distribution parameter.

    return:
    y - array of poisson distribution numbers
    x - matrix with shape(lenght,3) with random numbers of uniform distribution
    """
    y = np.random.poisson(lambd, size=lenght)
    x = np.random.uniform(0, np.exp(-lambd), lenght)
    for _ in range(2):
        x = np.vstack((x, np.random.uniform(0, np.exp(-lambd), lenght)))

    return x.T, y

class MyBatch(Batch):
    """
    Main class
    """
    def __init__(self, index, *args, **kwargs):
        """
        Initialization of variable from parent class - Batch.
        """
        super().__init__(index, *args, **kwargs)
        self.x = None
        self.y = None
        self.w = None
        self.b = None
        self.loss = None

    @property
    def components(self):
        """
        Define componentis.
        """
        return 'x', 'y', 'w', 'b', 'loss'

    @action
    def generate(self, lenght=10, ttype="linear"):
        """
        Create batch by self.indices by rewrite self.x and self.y.
        lenght - size all data.

        ttype - name of using algorithm:
            * 'linear' for linear regression. (default)
            * 'logistic' for logistic regression.
            * 'poisson' for poisson regression.

        return: self
        """
        if self.x is None or self.y is None:
            self = self.load(lenght, ttype)
        self.x, self.y = self.x[self.indices], self.y[self.indices]

        return self

    @action
    def load(self, lenght=10, ttype='linear', lambd=1):
        """
        Generate data for ttype-algorihm.

        lenght - size all data.

        ttype - name of using algorithm:
            * 'linear' for linear regression. (default)
            * 'logistic' for logistic regression.
            * 'poisson' for poisson regression.

        return: self
        """
        data_dict = {'linear': generate_linear_data(lenght),
                     'logistic': generate_logistic_data(lenght),
                     'poisson': generate_poisson_data(lambd, lenght)}
        self.x, self.y = data_dict['ttype']

        return self

    @model()
    def linear_model():
        """
        Function with graph of linear regression.

        return:
        array with shape = (3,2)
        x - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        w - slope coefficient of straight line.
        b - bias.
        """
        x = tf.placeholder(name='input', dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1, 1), dtype=tf.float32)

        predict = tf.multiply(w, x, name='output') + b
        loss = tf.reduce_mean(tf.square(predict - y))

        optimize = tf.train.GradientDescentOptimizer(learning_rate=0.007)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss], [w, b]]

    @action(model='linear_model')
    def train_linear_model(self, models, session):
        """
        Train linear regression.

        model - fit funtion. In this case it's linear_model.

        session - tensorflow session.

        return self.
        """
        x, y = models[0]
        optimizer, cost = models[1]
        params = models[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={x:self.x, y: self.y})
        self.w = params[0][0]
        self.b = params[1]
        self.loss = loss

        return self

    @model()
    def logistic_model():
        """
        Function with graph of logistic regression.

        return:
        array with shape = (3,2(3))
        x - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        predict - model prediction.
        w - slope coefficient of straight line.
        b - bias.
        """
        x = tf.placeholder(name='input', dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=1), name='weight', dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1, 1), dtype=tf.float32)

        predict = tf.sigmoid(tf.multiply(w, x, name='output') + b)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=predict))

        optimize = tf.train.AdamOptimizer(learning_rate=0.005)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w, b]]

    @action(model='logistic_model')
    def train_logistic_model(self, models, session, result, test):
        """
        Train logistic regression.

        model - fit funtion. In this case it's linear_model.

        session - tensorflow session.

        result - result of prediction.

        test - data to predict.

        return self.
        """
        x, y = models[0]
        optimizer, cost, predict = models[1]
        params = models[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={x:self.x, y: self.y})
        self.w = params[0][0]
        self.b = params[1]
        self.loss = loss
        result[:] = session.run([predict], feed_dict={x:test})[0]

        return self

    @model()
    def poisson_model():
        """
        Function with graph of poisson regression.

        return:
        array with shape = (3,2(3))
        x - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        predict - model prediction.
        w - array of weights.
        """
        x = tf.placeholder(name='input', shape=[None, 3], dtype=tf.float32)
        y = tf.placeholder(name='true_y', dtype=tf.float32)

        w = tf.Variable(np.random.randint(-1, 1, size=3).reshape(3, 1), name='weight', dtype=tf.float32)

        predict = tf.exp(tf.matmul(x, w))
        loss = tf.reduce_sum(tf.nn.log_poisson_loss(y, predict))

        optimize = tf.train.AdamOptimizer(learning_rate=0.005)
        train = optimize.minimize(loss)

        return [[x, y], [train, loss, predict], [w]]

    @action(model='poisson_model')
    def train_poisson_model(self, models, session, result, test):
        """
        Train poisson regression.

        model - fit funtion. In this case it's linear_model.

        session - tensorflow session.

        return self.
        """
        x, y = models[0]
        optimizer, cost, predict = models[1]
        params = models[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={x:self.x, y: self.y})
        self.w = params[0]
        self.loss = loss
        result[:] = session.run([predict], feed_dict={x:test})[0]

        return self
