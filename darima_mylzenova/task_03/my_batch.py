""" Custom batch class for storing mnist batch and preprocessing
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..")
from dataset import Batch, action, model, inbatch_parallel, ImagesBatch, any_action_failed


class MnistBatch(ImagesBatch):
    """ Mnist batch and models  """
    components = 'images', 'labels'

    # def __init__(self, index, *args, **kwargs):
    #     """ Init func, inherited from base batch
    #     """
    #     super().__init__(index, *args, **kwargs)
    #     self.images = None
    #     self.labels = None


    # @property
    # def components(self):
    #     """ Components of mnist-batch
    #     """
    #     return 'images', 'labels'


    def post_function(self, list_results):
        '''Post function for parallel shift, gathers results of every worker'''
        if any_action_failed(list_results):
            raise ValueError('Some workers have failed')

        result_batch = np.array(list_results)
        self.images = result_batch
        return self

    def init_function(self):
        '''Init function for parallel shift

        Returns
        -------
        list of indices
            each of them will be sent to the worker separately
        '''
        return [{'idx': i}  for i in range(self.images.shape[0])]


    @action
    @inbatch_parallel(init='init_function', post='post_function', target='threads')
    def shift_flattened_pic(self, idx, max_margin=8):
        """ Apply random shift to a flattened pic

        Args:
            idx: index in the self.images of a pic to be flattened
        Return:
            flattened shifted pic
        """
        pic = self.images[idx].reshape(28, 28)
        padded = np.pad(pic, pad_width=[[max_margin, max_margin], [max_margin, max_margin]],
                        mode='constant')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        res = padded[slicing]
        res = res.reshape((28, 28, 1))
        return res

def draw_stats(all_stats, labels, title):
    ''' Draw accuracy/iterations plot '''
    colors = ['r', 'g', 'b', 'plum']
    plt.title(title)
    for i, current in enumerate(all_stats):
        smoothed_current = []
        for j in range(10, len(current) - 10):
            smoothed_current.append(np.mean(current[j-10:j+10]))
        plt.plot(smoothed_current, label=labels[i], c=colors[i])
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()


def draw_digit(pics, y_predict, y_true, probs, answer):
    ''' Draw a random digit '''
    if answer:
        pos = np.where(np.array(y_predict[0]) == np.array(y_true[0]))[0]
    else:
        pos = np.where(np.array(y_predict[0]) != np.array(y_true[0]))[0]
    item = np.random.randint(len(pos) - 1)
    plt.imshow(np.reshape(pics[0][pos[item]], (28, 28)))
    plt.title('Predict: %.0f with prob %.2f, true: %.0f' %(y_predict[0][pos[item]], \
    np.amax(probs[0][pos[item]]), y_true[0][pos[item]]))
    plt.show()
