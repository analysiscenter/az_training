""" File contains function witch generate data to regressions """
import numpy as np

NUM_DIM_LIN = 13
def generate_linear_data(size=10, dist='unif'):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.
        dist: sample distribution 'unif' or 'norm'.
    Output:
        x: uniformly or normally distributed array with shape (size, 2)
        y: array [0..size] with some random noize. """
    if dist == 'unif':
        x = np.random.uniform(0, 2, size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)

    elif dist == 'norm':
        x = np.random.normal(size=size * NUM_DIM_LIN).reshape(-1, NUM_DIM_LIN)

    w = np.random.normal(loc=1., size=[NUM_DIM_LIN])
    error = np.random.normal(loc=0., scale=0.1, size=size)

    xmulw = np.dot(x, w)
    y_obs = xmulw + error

    return x, y_obs.reshape(-1, 1)

def generate_logistic_data(size, first_params, second_params):
    """ Generation of data for fit linear regression.
    Args:
        size: length of data.

    Output:
        x: Coordinates of points in two-dimensional space with shape (size, 2)
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

    lam: Poisson distribution parameter.

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
