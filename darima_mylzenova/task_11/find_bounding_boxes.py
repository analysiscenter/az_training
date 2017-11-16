import sys

import numpy as np
import os
import blosc
import time

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.contrib.layers import xavier_initializer_conv2d

from dataset import Batch, action, model, inbatch_parallel, ImagesBatch


class DetectionBatch(ImagesBatch):
    ''' A batch for detection task on MNIST data
    '''
    def __init__(self, index, *args, **kwargs):
        """ Init func, inherited from base batch
        """
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.labels = None
        self.bb_coordinates = None


    @property
    def components(self):
        """ Components of mnist-batch
        """
        return 'images', 'labels', 'bb_coordinates'


    @action
    def load(self, src, fmt='blosc'):
        """ Load mnist pics with specifed indices

        Args:
            fmt: format of source. Can be either 'blosc' or 'ndarray'
            src: if fmt='blosc', then src is a path to dir with blosc-packed
                mnist images and labels are stored.
                if fmt='ndarray' - this is a tuple with arrays of images and labels

        Return:
            self
        """
        if fmt == 'blosc':     
            # read blosc images, labels
            with open('mnist_pics.blk', 'rb') as file:
                self.images = blosc.unpack_array(file.read())[self.indices]
                self.images = np.reshape(self.images, (65000, 28, 28))

            with open('mnist_labels.blk', 'rb') as file:
                self.labels = blosc.unpack_array(file.read())[self.indices]
        elif fmt == 'ndarray':
            all_images, all_labels = src
            self.images = all_images[self.indices]
            self.labels = all_labels[self.indices]

        return self

    def post_function(self, list_results):
        '''Post function for parallel shift, gathers results of every worker'''
        result_batch = np.array(list_results)
        self.images = result_batch
        return self

    def init_function(self):
        '''Init function for parallel shift
        returns list of indices, each of them will be sent to the worker separately
        '''
        return [{'idx': i}  for i in range(self.images.shape[0])]

    def bb_post_function(self, list_results):
        result_bb_batch = np.array(list_results)
        self.bb_coordinates = result_bb_batch

    @action
    @inbatch_parallel(init='init_function', post='bb_post_function', target='threads')
    def find_bounding_box(self, idx, max_margin=8):
        """ Apply random shift to a flattened pic
        
        Args:
            idx: index in the self.images of a pic to be flattened
        Return:
            flattened shifted pic
        """
        
        pic = self.images[idx]
        size = pic.shape[0]
        for i in range(size):
            for j in range(size):
                if pic[i, j] > 0:
                    min_row = i
                    min_col = j

                if pic[size-i, size-j] > 0:
                    max_row = size - i
                    max_col = size - j
        return [min_row, min_col, max_row, max_col]


    @action
    @inbatch_parallel(init='init_function', post='post_function', target='threads')
    def enlarge_data(self, idx, new_size=64):
        pic = self.images[idx]
        size = pic.shape[0]

        padding_size = new_size - size
        left_pad = np.random.randint(padding_size)
        bottom_pad = np.random.randint(padding_size)

        padded_pic = np.pad(pic, pad_width=[[left_pad, padding_size - left_pad], \
                            [bottom_pad, padding_size - bottom_pad]], mode='constant')
        return padded_pic
