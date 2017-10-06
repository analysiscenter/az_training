""" File with some useful functions"""
import numpy as np
import matplotlib.pyplot as plt
from pandas import ewma


plt.style.use('seaborn-poster')
plt.style.use('ggplot')

def draw(fitst, first_label, second=None, second_label=None, type_data='loss', window=50, bound=None):

    """ Draw graphs to compare models. The graph shows a comparison of the average
        values calculated with a window.
    Args:
        fitst: List with loss value first model
        second: List with loss value second model
        tp: type of graph. "loss" or 'accuracy' or etc.
        first_label: label of first model
        second_label: label of second model
        window: average window
        bound: List with bounds to limit graph or None.
            [min x, max x, min y, max y]"""

    firt_ewma = ewma(np.array(fitst), span=window, adjust=False)
    second_ewma = ewma(np.array(second), span=window, adjust=False) if second else None

    plt.plot(firt_ewma, label='{} {}'.format(first_label, type_data))
    if second_label:
        plt.plot(second_ewma, label='{} {}'.format(second_label, type_data))
    plt.xlabel('Iteration', fontsize=16)
    plt.ylabel(type_data, fontsize=16)
    plt.legend(fontsize=14)
    if bound:
        plt.axis(bound)
