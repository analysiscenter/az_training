""" File contains function wicth generate data to regressions"""
import numpy as np


def generate_linear_data(size=10, dist='unif', error_scale=1):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.

    Output:
        x: uniformly or normally distributed array with shape (size, 2)
        y: array [0..size] with some random noize. """
    if dist == 'unif':
        x = np.random.uniform(-1, 1, size * 4).reshape(-1, 2)
        b = np.random.uniform(-1, 1, 1)
    elif dist == 'norm':
        x = np.random.normal(size=size * 6).reshape(-1, 3)
        b = np.random.normal(size=1)

    w = np.random.random(2)
    error = np.random.normal(scale=error_scale * 0.1, size=size * 2)

    xmulw = x * w
    print('trues weights and bias: ', w, ' ', b)

    y_obs = np.array([np.sum(xmulw[i] + error[i] + b) for i in range(len(error))])
    y_true = np.sum(xmulw, axis=1) + b
    print('mse between y with and without noize: ', np.mean((y_obs - y_true) ** 2))
    return x, y_obs

def generate_logistic_data(size=10, first_params=None, second_params=None):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.

    Output:
        x: Coordinates of points in two-dimensional space with shape (size. 2)
        y: labels of dots """
    first = np.random.multivariate_normal(first_params[0], first_params[1], size)
    second = np.random.multivariate_normal(second_params[0], second_params[1], size)

    x = np.vstack((first, second))
    y = np.hstack((np.zeros(size), np.ones(size))).reshape(-1, 1)
    shuffle = np.arange(len(x))
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]

    return x, y

def generate_poisson_data(lam, size=10):
    """ Generation of data for fit poisson regression.

    size: size of data.

    lambd: Poisson distribution parameter.

    Output:
        y: array of poisson distribution numbers.
        x: matrix with shape(size,2) with random numbers of uniform distribution. """
    x = np.random.random(size * 4).reshape(-1, 2)
    y = np.random.poisson(np.exp(np.dot(x, lam)))

    shuffle = np.arange(len(x))
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y[shuffle]

    return x, y
