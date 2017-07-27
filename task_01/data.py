""" File contains function wicth generate data to regressions"""
import numpy as np


NUM_DIM_LIN = 13
def generate_linear_data(size=10, dist='unif', error_scale=1):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.

    Output:
        x: uniformly or normally distributed array with shape (size, 2)
        y: array [0..size] with some random noize. """
    if dist == 'unif':
        x = np.random.uniform(0, 100, size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)
        b = np.random.uniform(0, 10, 1)
    elif dist == 'norm':
        x = np.random.normal(size=size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)
        b = np.random.normal(size=1)

    w = np.random.random(NUM_DIM_LIN)
    error = np.random.normal(scale=error_scale * 0.1, size=size)

    xmulw = x * w

    y_obs = np.array([np.sum(xmulw[i] + error[i] + b) for i in range(len(error))])
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
    x = np.random.random(size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)
    b = np.random.random(1)

    y_obs = np.random.poisson(np.exp(np.dot(x, lam) + b))

    shuffle = np.arange(len(x))
    np.random.shuffle(shuffle)
    x = x[shuffle]
    y = y_obs[shuffle]

    return x, y
