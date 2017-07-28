''' Logistic regression using dataset and tensor flow '''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, make_blobs
from sklearn.preprocessing import scale
from sklearn.datasets import make_classification

from dataset import Dataset, Batch, action, model

NUM_DIM = 13


class MyBatch(Batch):
    ''' A Batch with logistic regression model '''

    @property
    def components(self):
        ''' Define components '''
        return "features", "labels"

    @action()
    def preprocess_linear_data(self):
        ''' Normalize data '''
        scale(self.features, axis=0, with_mean=True, with_std=True, copy=False)
        return self

    @model()
    def linear_regression():
        ''' Define tf graph for linear regression '''
        learning_rate = 0.01
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.ones([NUM_DIM, 1]))
        bias = tf.Variable(tf.ones([1]))

        y_cup = tf.add(tf.matmul(x_features, weights), bias)
        cost = tf.add(tf.reduce_mean(tf.square(y_target - y_cup)), tf.multiply(tf.reduce_sum(tf.square(weights)), 0.1))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
        return training_step, cost, x_features, y_target, y_cup
    
    def train_any(self, model, sess, my_cost_history):
        ''' Train any regression on the batch '''
        training_step, cost, x_features, y_target = model[:-1]
        sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return my_cost_history

    def predict_any(self, model, sess, y_pred):
        '''Predict target for any model'''
        x_features = model[2]
        y_cup = model[4]
        y_pred[:] = sess.run(y_cup, feed_dict={x_features:self.features})


    @action(model='linear_regression')
    def train_linear(self, model, sess, my_cost_history):
        ''' Train linear regression on the batch '''
        self.train_any(model, sess, my_cost_history)
        return self

    @action(model='linear_regression')
    def test_linear(self, model, sess, y_true, y_pred, mse, x_features):
        ''' Test batch '''
        self.predict_any(model, sess, y_pred)
        mse.append(sess.run(tf.reduce_mean(tf.square(y_pred - self.labels))))
        y_true[:] = self.labels
        x_features[:] = self.features
        return self

    @model()
    def logistic_regression():
        '''Define tf graph for logistic regression model'''
        learning_rate = 0.005
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.zeros([NUM_DIM, 1]))
        bias = tf.Variable(tf.zeros([1]))

        wx_b = tf.add(tf.matmul(x_features, weights), bias)

        prob = tf.sigmoid(wx_b)
        y_cup = tf.sign(wx_b)

        margin = tf.multiply(y_target, wx_b)
        cost = tf.reduce_mean(tf.log(tf.add(tf.ones([1]), tf.exp(-margin))))
        training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        acc = tf.contrib.metrics.accuracy(tf.cast(y_cup, tf.int32), tf.cast(y_target, tf.int32))

        return training_step, cost, x_features, y_target, y_cup, acc

    @action(model='logistic_regression')
    def train_logistic(self, model, sess, my_cost_history, acc_history):
        ''' Train logistic regression on the batch '''
        # self.train_any(model, sess, my_cost_history)
        training_step, cost, x_features, y_target = model[:-2]
        acc = model[-1]
        sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        
        acc_history.append(sess.run(acc, feed_dict={x_features:self.features, y_target:self.labels}))
        return self
    
    @action(model='logistic_regression')
    def test_logistic(self, model, sess, acc):
        ''' Test logistic regression on the batch'''
        y_pred = np.zeros([len(self.labels), 1])
        self.predict_any(model, sess, y_pred)
        y_target_int = tf.cast(self.labels, tf.int32)
        y_pred_int = tf.cast(y_pred, tf.int32)
        acc.append(sess.run(tf.contrib.metrics.accuracy(y_pred_int, y_target_int)))
        return self

    @model()
    def poisson_regression():
        '''Define tf graph for logistic regression model'''
        learning_rate = 0.005
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.zeros([NUM_DIM, 1]))
        bias = tf.Variable(tf.zeros([1]))

        log_input = tf.add(tf.matmul(x_features, weights), bias)
        cost = tf.reduce_mean(tf.add(tf.nn.log_poisson_loss(y_target, log_input, compute_full_loss=False), tf.multiply(tf.reduce_sum(tf.square(weights)), 0.1)))
        # cost = tf.add(tf.reduce_mean(tf.square(y_target - y_cup)), tf.multiply(tf.reduce_sum(tf.square(weights)), 0.1))

        # y_cup = tf.exp(log_input)
            
        training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        return training_step, cost, x_features, y_target, log_input, weights


    @action(model='poisson_regression')
    def train_poisson(self, model, sess, my_cost_history):
        ''' Train poisson regression on the batch '''
        self.train_any(model[:-1], sess, my_cost_history)
        return self


    @action(model='poisson_regression')
    def test_poisson(self, model, sess, y_true, y_pred, weights):
        ''' Test poisson regression on the batch '''
        # y_pred = np.zeros([len(self.labels), 1])
        self.predict_any(model[:-1], sess, y_pred)
        # cost = model[1]
        
        # y_pred_int = sess.run(tf.cast(tf.round(y_pred) + 1, tf.int32))

        # mse = tf.reduce_mean(tf.square(y_pred_int - self.labels))
        weights[:] = sess.run(model[-1], feed_dict={model[2]:self.features})
        y_true[:] = self.labels
        return self


def load_linear_data(num_samples=500):
    ''' load some data '''
    features = np.random.random_sample([num_samples, NUM_DIM])
    weights = np.random.random_sample([NUM_DIM])
    xw = np.dot(features, weights)
    labels = np.random.normal(num_samples) + xw
    labels = np.reshape(labels, [num_samples, 1])
    return features, labels


def load_poisson_data():
    '''generate poisson data'''
    features = np.random.random_sample([500, NUM_DIM])
    weights = np.random.random_sample([NUM_DIM])
    xw = np.exp(np.dot(features, weights))
    labels = np.reshape(np.array([np.random.poisson(lmbd) for lmbd in xw]), [500, 1])
    labels.astype(np.float32)
    weights = np.reshape(weights, [NUM_DIM, 1])
    return weights, features, labels


def load_random_data(n_samples=500, blobs=False):
    ''' load some data '''
    if blobs:
        features, labels = make_blobs(n_samples=n_samples, n_features=13, random_state=1, centers=2)
    else:
        features, labels = make_classification(n_samples=n_samples, n_features=13, random_state=1, n_classes=2, flip_y=0.005)
    labels = np.reshape(labels, [labels.shape[0], 1])
    return features, labels


def load_dataset(n_samples):
    ''' create Dataset with given data '''
    dataset = Dataset(index=np.arange(n_samples), batch_class=MyBatch)    
    dataset.cv_split()
    return dataset



def preprocess_logistic_data(data):
    ''' Change labbel of the second class to '-1' instead of 0'''
    features, labels = data
    labels = 2*data[1] - np.ones((len(data[1]), 1), dtype=np.float32)
    return features, labels


def plot_cost(cost_history):
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, len(cost_history), np.min(cost_history), np.max(cost_history)])
    plt.show()


def plot_test_linear(x_features, y_true, y_pred):
    axis = plt.subplots()[1]
    axis.scatter(y_true, y_pred)
    axis.plot(np.sort(x_features[:, 0], y_pred, 'k--'))
    axis.plot(np.sort(x_features[:, 0], y_true))
    axis.set_xlabel('x')
    axis.set_ylabel('y')
    plt.show()


def plot_log_cost(sess, data, cost_history):
    x_features = tf.placeholder(tf.float32, [None, None])
    y_target = tf.placeholder(tf.float32, [None, 1])
    weights = tf.placeholder(tf.float32, [13, 1])

    log_input = tf.matmul(x_features, weights)
    cost = tf.reduce_mean(tf.nn.log_poisson_loss(y_target, log_input, compute_full_loss=False))

    origin_loss = sess.run(cost, feed_dict={weights:data[0], x_features:data[1], y_target:data[2]})

    plt.plot(range(len(cost_history)), cost_history)
    plt.plot(np.linspace(0, len(cost_history), len(cost_history)), [origin_loss]*len(cost_history), '-', color='g')
    plt.axis([0, len(cost_history), np.min(cost_history), np.max(cost_history)])
    plt.show()


def compute_sample_var(labels):
    return n/(n-1)*np.var(labels)

