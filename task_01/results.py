""" File to print all information how all pregressions work. And print some score. """
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, mean_squared_error

def print_info(dict_params, y_true, y_predict, loss):
    """ Print score of models.
    Args:
        dict_params: dict of model parameters like loss and weights.
        y_ture: real answers for data.
        y_predict: model prediction.
        loss: type of loss."""
    w = np.array(dict_params['w'][-1]).reshape(-1)
    b = dict_params['b'][0]
    y_predict = np.array(y_predict).reshape(-1)
    y_true = np.array(y_true).reshape(-1)
    print('models weights and bias: ', w, ' ', b)
    print('size of prediction data: ', len(y_predict))

    if loss == 'mse':
        print('model mse: ', mean_squared_error(y_true, y_predict))
    elif loss == 'acc':
        y_predict = np.array([1 if i >= 0.5 else 0 for i in y_predict])
        print('model acc: ', accuracy_score(y_true, y_predict))

def plot_loss(loss):
    """ Draws a train loss schedule"""
    plt.figure(figsize=(15, 10))
    axis_font = {'fontname':'Arial', 'size':'20'}
    plt.plot(loss)
    plt.xlabel('Iteration', **axis_font)
    plt.ylabel('Loss(mse)', **axis_font)
