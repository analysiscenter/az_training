''' Regression models implementation using dataset and tensorflow '''
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import scale
from sklearn.datasets import make_classification

sys.path.append("../..")
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

    @action()
    def preprocess_binary_data(self):
        ''' Change label of the second class to '-1' instead of 0'''
        self.labels[:] = 2*self.labels - np.ones((len(self.labels), 1), dtype=np.float32)
        return self


def load_linear_data(num_samples=500):
    ''' load some data '''
    features = np.random.random_sample([num_samples, NUM_DIM])
    weights = np.random.random_sample([NUM_DIM])
    x_mult_w = np.dot(features, weights)

    labels = np.random.normal(loc=0.0, scale=0.1, size=num_samples) + x_mult_w
    labels = np.reshape(labels, [num_samples, 1])
    return features, labels


def load_poisson_data():
    '''generate poisson data'''
    features = np.random.random_sample([500, NUM_DIM])
    weights = np.random.random_sample([NUM_DIM])
    bias = np.random.random(1)
    x_mult_w = np.exp(np.dot(features, weights) + bias)
    labels = np.reshape(np.random.poisson(x_mult_w), [500, 1])
    labels.astype(np.float32)
    weights = np.reshape(weights, [NUM_DIM, 1])
    return weights, features, labels


def load_random_data(n_samples=500, blobs=False):
    ''' load some data '''
    if blobs:
        features, labels = make_blobs(n_samples=n_samples, n_features=13, random_state=1, centers=2)
    else:
        features, labels = make_classification \
        (n_samples=n_samples, n_features=13, random_state=1, n_classes=2, flip_y=0.005)
    labels = np.reshape(labels, [labels.shape[0], 1])
    return features, labels


def load_dataset(n_samples):
    ''' create Dataset with given data '''
    dataset = Dataset(index=np.arange(n_samples), batch_class=MyBatch)
    dataset.cv_split()
    return dataset



def plot_cost(cost_history):
    ''' Plot cost history '''
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, len(cost_history), np.min(cost_history), np.max(cost_history)])
    plt.show()


def plot_log_cost(sess, data, cost_history):
    ''' Plot cost history and cost of the target together '''
    x_features = tf.placeholder(tf.float32, [None, None])
    y_target = tf.placeholder(tf.float32, [None, 1])
    weights = tf.placeholder(tf.float32, [13, 1])

    log_input = tf.matmul(x_features, weights)
    cost = tf.reduce_mean(tf.nn.log_poisson_loss(y_target, log_input, compute_full_loss=False))

    origin_loss = sess.run(cost, feed_dict={weights: data[0], x_features: data[1], y_target: data[2]})

    plt.plot(range(len(cost_history)), cost_history)
    plt.plot(np.linspace(0, len(cost_history), len(cost_history)), [origin_loss]*len(cost_history), '-', color='g')
    plt.axis([0, len(cost_history), np.min(cost_history), np.max(cost_history)])
    plt.show()

