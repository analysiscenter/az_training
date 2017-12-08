""" Custom batch class for storing mnist batch and models
"""
import sys

import numpy as np

sys.path.append('..')
from dataset.dataset import action, inbatch_parallel, any_action_failed
from dataset.dataset import ImagesBatch

class MnistBatch(ImagesBatch):
    """ Mnist batch and models
    Attributes
    ----------
    images: numpy array
    Array with images

    labels: numpy array
    Array with answers """

    # def __init__(self, index, *args, **kwargs):
    #     _ = args, kwargs
    #     super().__init__(index, *args, **kwargs)
    #     self.images = None
    #     self.labels = None

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def shift_flattened_pic(self, ind, max_margin=8):
        """ Apply random shift to a flattened pic

        Parameters
        ----------
        ind: numpy array
        Array with indices, which need to shift

        max_margin: int
        Constit max value of margin that inamge may
        be shift

        Returns
        -------
        flattened shifted pic """

        squared = self.images[ind].reshape(28, 28)
        padded = np.pad(squared, pad_width=[[max_margin, max_margin], [max_margin, max_margin]],
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        return padded[slicing].reshape(-1, 28, 28, 1)

    def post_func(self, list_of_res):
        """ Concat outputs from shift_flattened_pic
        Parameters
        ----------
        list_of_res: numpy array
        Array with results of shifting """
        if any_action_failed(list_of_res):
            raise Exception("Something bad happend")
        else:
            self.images = np.stack(list_of_res).reshape(-1, 28, 28, 1) # pylint: disable=attribute-defined-outside-init
            return self

    def init_func(self):
        """ Create queue to parallel.
        Resurns
        -------
        Array with parallel indices """
        return [{'ind':i} for i in range(self.images.shape[0])]
