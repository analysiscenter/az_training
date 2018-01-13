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
    components = 'images', 'labels', 'coordinates', 'noise', 'other_coordinates', 'other_labels'
    # def __init__(self, index, *args, **kwargs):
    #     """ Init func, inherited from base batch
    #     """
    #     super().__init__(index, *args, **kwargs)
    #     self.images = None
    #     self.labels = None
    #     self.bb_coordinates = None


    # @property
    # def components(self):
    #     """ Components of mnist-batch
    #     """
    #     return 'images', 'labels', 'bb_coordinates'


    # @action
    # def load(self, src, fmt='blosc'):
    #     """ Load mnist pics with specifed indices

    #     Args:
    #         fmt: format of source. Can be either 'blosc' or 'ndarray'
    #         src: if fmt='blosc', then src is a path to dir with blosc-packed
    #             mnist images and labels are stored.
    #             if fmt='ndarray' - this is a tuple with arrays of images and labels

    #     Return:
    #         self
    #     """
    #     if fmt == 'blosc':     
    #         # read blosc images, labels
    #         with open('mnist_pics.blk', 'rb') as file:
    #             self.images = blosc.unpack_array(file.read())[self.indices]
    #             self.images = np.reshape(self.images, (65000, 28, 28))

    #         with open('mnist_labels.blk', 'rb') as file:
    #             self.labels = blosc.unpack_array(file.read())[self.indices]
    #     elif fmt == 'ndarray':
    #         all_images, all_labels = src
    #         self.images = all_images[self.indices]
    #         self.labels = all_labels[self.indices]

    #     return self

    def post_function(self, list_results):
        '''Post function for parallel shift, gathers results of every worker'''
        print(list_results)
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
        self.coordinates = result_bb_batch
        return self

    @action
    @inbatch_parallel(init='images', post='assemble', components='noise')
    def create_noise(self, image, *args):
        """Create noise at MNIST image."""
        # print('1')
        image_size = self.images.shape[1]
        if args[0] == 'random_noise':
            noise = args[1] * np.random.random((image_size, image_size, 1)) * image.max()
        elif args[0] == 'mnist_noise':
            level, n_fragments, size, distr = args[1:]

            ind_for_noise = np.random.choice(len(self.images), n_fragments)
            images = [self.images[i] for i in ind_for_noise]
            coordinates = [self.coordinates[i] for i in ind_for_noise]
            images_for_noise = self.crop_images(images, coordinates)
            fragments = self.create_fragments(images_for_noise, size)
            noise = self.arrange_fragments(image_size, fragments, distr, level)
        else:
            noise = np.zeros_like(image)
        return noise

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def add_noise(self, ind):
        if self.images.shape[-1] != 1:
            return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)
        else:
            return np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0)
    # def enlarge_post_function(self, list_results):
    #     result_bb_batch = np.array(list_results)
    #     self.coordinates = result_bb_batch

        # return self

    # @action
    # @inbatch_parallel(init='images', post='assemble', components=('images', 'coordinates'))
    @action
    @inbatch_parallel(init='init_function', post='bb_post_function', target='threads')
    def find_and_crop(self, idx):
        """ Apply random shift to a flattened pic
        
        Args:
            idx: index in the self.images of a pic to be flattened
        Return:
            flattened shifted pic
        """
        print(idx, 'IDX')
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
        coordinates = [min_row, min_col, max_row, max_col]
        # return crop_images(pic, coordinates), coordinates 
        return coordinates


    # , components=('images', 'coordinates'))
    @action
    @inbatch_parallel(init='images', post='assemble', components=('images', 'coordinates', 'other_coordinates', 'other_labels'))
    def enlarge_data(self, image, num_others=3, new_size=64):
        # pic = self.images[idx]
        pic = image
        size = pic.shape[0]
        # padding_size = new_size - size
        # left_pad = np.random.randint(padding_size)
        # bottom_pad = np.random.randint(padding_size)

        # padded_pic = np.pad(pic, pad_width=[[left_pad, padding_size - left_pad], \
        #                     [bottom_pad, padding_size - bottom_pad]], mode='constant')

        # return padded_pic
        image_size = new_size
        pure_mnist = np.squeeze(image)
        large_mnist = np.zeros((image_size, image_size))

        all_indices = self.images.shape[0]

        other_label= []
        other_coord = []
        for i in range(num_others):
            random_idx = np.random.randint(0, image_size-28)
            random_image = np.squeeze(self.images[random_idx])
            new_x, new_y = np.random.randint(0, image_size-28, 2)
            new_x_2, new_y_2 = new_x + 28, new_y + 28
            large_mnist[new_x:new_x+28, new_y:new_y+28] = random_image
            other_coord.append([new_x, new_y, new_x_2, new_y_2])
            other_label.append(self.labels[random_idx])

        new_x, new_y = np.random.randint(0, image_size-28, 2)
        new_x_2, new_y_2 = new_x + 28, new_y + 28
        large_mnist[new_x:new_x+28, new_y:new_y+28] = pure_mnist
        large_mnist = np.expand_dims(large_mnist, axis=3)
        return large_mnist, [new_x, new_y, new_x_2, new_y_2], other_coord, other_label

    @action
    @inbatch_parallel(init='init_function', post='post_function', target='threads')
    def shift_flattened_pic(self, idx, max_margin=8):
        """ Apply random shift to a flattened pic
        
        Args:
            idx: index in the self.images of a pic to be flattened
        Return:
            flattened shifted pic
        """
        
        pic = self.images[idx]
        padded = np.pad(pic, pad_width=[[max_margin, max_margin], [max_margin, max_margin]], 
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        res = padded[slicing]
        return res


    def crop_images(self, images, coordinates):
        """Crop real 28x28 MNIST from large image."""
        images_for_noise = []
        for image, coord in zip(images, coordinates):
            images_for_noise.append(image[coord[0]:coord[0] + 28, coord[1]:coord[1] + 28])
        return images_for_noise
