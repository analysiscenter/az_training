""" Custom batch class for storing mnist batch and models
"""
import sys
import os

import blosc
import numpy as np

sys.path.append('..')
from dataset import action, inbatch_parallel, any_action_failed
from dataset.dataset.image import ImagesBatch

class MnistBatch(ImagesBatch):
    """ Mnist batch and models
    Attributes
    ----------
    images: numpy array
    Array with images

    labels: numpy array
    Array with answers """

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.labels = None
        _, _ = args, kwargs

    @property
    def components(self):
        """ Components of mnist-batch """
        return 'images', 'labels'

    @action
    def load(self, src, fmt='blosc', components=None, *args, **kwargs):
        """ Load mnist pics with specified indices

        Parameters
        ----------
        src: str or numpy array
        if fmt='blosc' then src is a path to dir with blosc-packed
                mnist images and labels are stored.
        if fmt='ndarray' - this is a tuple with arrays of images and labels

        fmt: 'blosc' or 'ndarray'
        Format of source

        components: dict, optional
        Not used """
        _, _, _ = args, kwargs, components
        if fmt == 'blosc':
            # read blosc images, labels
            with open(os.path.join(src, 'mnist_pics.blk'), 'rb') as file:
                self.images = blosc.unpack_array(file.read())[self.indices]

            with open(os.path.join(src, 'mnist_labels.blk'), 'rb') as file:
                self.labels = blosc.unpack_array(file.read())[self.indices]
        elif fmt == 'ndarray':
            all_images, all_labels = src
            self.images = all_images[self.indices]
            self.labels = all_labels[self.indices]

        return self

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
        return padded[slicing].reshape(-1)

    def post_func(self, list_of_res):
        """ Concat outputs from shift_flattened_pic
        Parameters
        ----------
        list_of_res: numpy array
        Array with results of shifting """
        if any_action_failed(list_of_res):
            raise Exception("Something bad happend")
        else:
            self.images = np.stack(list_of_res)
            return self

    def init_func(self):
        """ Create queue to parallel.
        Resurns
        -------
        Array with parallel indices """
        return [{'ind':i} for i in range(self.images.shape[0])]
