"""data generation functions for regression models"""
import numpy as np
from sklearn.datasets import make_blobs
from sklearn.datasets import make_classification

def load_linear_data(num_dim, num_samples=500):
    ''' generate linearly dependent data '''
    features = np.random.random_sample([num_samples, num_dim])
    weights = np.random.random_sample([num_dim])
    x_mult_w = np.dot(features, weights)

    labels = np.random.normal(loc=0.0, scale=0.1, size=num_samples) + x_mult_w
    labels = np.reshape(labels, [num_samples, 1])
    return features, labels


def load_poisson_data(num_dim, num_samples=500):
    '''generate data for poisson regression'''
    features = np.random.random_sample([num_samples, num_dim])
    weights = np.random.random_sample([num_dim])
    bias = np.random.random(1)
    x_mult_w = np.exp(np.dot(features, weights) + bias)
    labels = np.reshape(np.random.poisson(x_mult_w), [num_samples, 1])
    labels.astype(np.float32)
    weights = np.reshape(weights, [num_dim, 1])
    return weights, features, labels


def load_random_data(num_dim, n_samples=500, blobs=False):
    ''' generate data for binary classificaation'''
    if blobs:
        features, labels = make_blobs(n_samples=n_samples, n_features=num_dim,
                                      random_state=1, centers=2)
    else:
        features, labels = make_classification \
        (n_samples=n_samples, n_features=num_dim, random_state=1, n_classes=2, flip_y=0.005)
    labels = np.reshape(labels, [labels.shape[0], 1])
    return features, labels
