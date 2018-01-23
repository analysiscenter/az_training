import sys

import numpy as np
import os
import blosc
import time

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.contrib.layers import xavier_initializer_conv2d

from dataset import Batch, action, inbatch_parallel, ImagesBatch


class DetectionBatch(ImagesBatch):
    ''' A batch for detection task on MNIST data
    '''
    components = 'images', 'labels', 'coordinates', 'noise'
    

    @action
    @inbatch_parallel(init='images', post='assemble', components='noise')
    def create_noise(self, image, *args):
        """Create noise at MNIST image."""
        # image = self.images[idx, :, :, :]
        image_size = self.images.shape[1]
        if args[0] == 'random_noise':
            noise = args[1] * np.random.random((image_size, image_size, 1)) * image.max()
        elif args[0] == 'mnist_noise':
            level, n_fragments, size, distr = args[1:]
            ind_for_noise = np.random.choice(self.images.shape[0], n_fragments)
            images = [self.images[i] for i in ind_for_noise]
            coordinates = [self.coordinates[i][0] for i in ind_for_noise]

            try:
                images_for_noise = self.crop_images(images, coordinates)
            except Exception as e:
                print('crop_images fail')
                return ValueError
            try:
                fragments = self.create_fragments(images_for_noise, size)
#                 print('create_fragments_DONE')
            except Exception as e:
                print('create_fragments fail')
#                 print(len(images_for_noise))
                return ValueError
            try:
                noise = self.arrange_fragments(image_size, fragments, distr, level)
            except Exception as e:
                print('arrange_fragments fail')
                return ValueError
        else:
            noise = np.zeros_like(image)
        return noise
    
    def crop_images(self, images, coordinates):
        """Crop real 28x28 MNIST from large image."""
        images_for_noise = []
        for image, coord in zip(images, coordinates):
#             print(coordinates, 'coordinates')
            try:
                images_for_noise.append(image[coord[0]:coord[2], coord[1]:coord[3]])
#                 if images_for_noise[-1].shape[1] < 4:
#                     print(coord)
            except Exception as e:
                print('HEREEEEE')
                return ValueError
#             print('SALUT', len(images_for_noise), 'SHAPE HERE')
        return images_for_noise

    def create_fragments(self, images, size):
        """Cut fragment from each."""
        fragments = []
        for image in images:
            image = np.squeeze(image)
#             print('ldldl', image.shape)
            
            x = np.random.randint(0, image.shape[0] - size)
            y = np.random.randint(0, image.shape[1] - size)
            try:
                fragment = image[x:x + size, y:y + size]
            except Exception as e:
                print('x={}, y={}, image.shape={}'.format(x, y, image.shape))
            fragments.append(fragment)
#         print(len(fragments), 'FRAGMENTs', len(images), 'IMAGS')
        return fragments

    def arrange_fragments(self, image_size, fragments, distr, level):
        """Put fragments on image."""
        image = np.zeros((image_size, image_size))
        for fragment in fragments:
            size = fragment.shape[0]
            try:
                x_fragment, y_fragment = getattr(self, distr)(image_size, size)
            except Exception as e:
                print('2')
                return ValueError
            try:
                image_to_change = image[x_fragment:x_fragment+size, y_fragment:y_fragment+size]
            except Exception as e:
                print('3')
                return ValueError
            height_to_change, width_to_change = image_to_change.shape
            # print(height_to_change, '+', width_to_change)
#             print(fragment.shape)
            try:
                image_to_change = np.max([level*fragment[:height_to_change, :width_to_change], image_to_change], axis=0)
                # print (image_to_change.shape, 'SHAPE')
            except Exception as e:
                print('5')
                return ValueError
            try:
                image[x_fragment:x_fragment+size, y_fragment:y_fragment+size] = image_to_change
            except Exception as e:
                print('6')
                return ValueError
        return image

    @action
    @inbatch_parallel(init='indices', post='assemble')
    def add_noise(self, ind, margin=4):
        coordinates = self.get(ind, 'coordinates')
        try:
            noise = self.get(ind, 'noise')
            for bbox in enumerate(coordinates):
                x_left, x_right, y_left, y_right = bbox
                print(x_left, x_right, y_left, y_right)
                noise[x_left:x_right, y_left:y_right] = np.zeros((x_right - x_left, y_right - y_left)) 
        except Exception as e:
            print('wrong size')
            return ValueError

        if self.images.shape[-1] != 1:
            return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)
        else:
            
            return np.max([self.get(ind, 'images'), np.expand_dims(noise, axis=-1)], axis=0)
    
    def uniform(self, image_size, fragment_size):
        """Uniform distribution of fragmnents on image."""
        return np.random.randint(0, image_size-fragment_size, 2)

    def normal(self, image_size, fragment_size):
        """Normal distribution of fragmnents on image."""
        return list([int(x) for x in np.random.normal((image_size-fragment_size)/2,
                                                      (image_size-fragment_size)/4, 2)])


    @action
    @inbatch_parallel(init='images', post='assemble', components=('images', 'labels', 'coordinates'))
    def generate_data(self, image, margin=4, num_digits=3, new_size=64):
        ''' Generate image with num_digits random MNIST digits om it

        Parameters
        ----------
        image : np.array
        

        '''
        canvas = np.zeros((new_size, new_size))
        random_indices = np.random.choice(self.images.shape[0], num_digits)
        random_images = [np.squeeze(self.images[i]) for i in random_indices]
        labels = [self.labels[i] for i in random_indices]

        coordinates = []
        for random_image in random_images:
            try:
                random_cropped = self.find_and_crop(random_image)
            except Exception as e:
                print('find_and_crop error')
                return ValueError
            width, height = random_cropped.shape
            left_x = np.random.randint(0, new_size - width)
            left_y = np.random.randint(0, new_size - height)
            right_x, right_y = left_x + width, left_y + height

            canvas[left_x:right_x, left_y:right_y] = random_cropped
            left_x, left_y = max(left_x - margin, 0), max(left_y - margin, 0)
            right_x, right_y = min(right_x + margin, new_size), min(right_y + margin, new_size)

            coordinates.append([left_x, left_y, right_x, right_y])
        canvas = np.expand_dims(canvas, axis=3)
        return canvas, labels, coordinates
    
    @action
    @inbatch_parallel(init='indices', post='assemble', components='labels')
    def one_hot(self, ind):
        """ One hot encoding for labels
        Parameters
        ----------
        ind : numpy.uint8
            index
        Returns
        -------
            One hot labels"""
        label = self.get(ind, 'labels')
        one_hot = np.zeros((len(label), 10))
        one_hot[np.arange(len(label)), label] = 1
        return one_hot.reshape(-1)

    def find_and_crop(self, image):
        """ Find and crop a rectangle with digit        
        Args:
            image: square image with a digit to be cropped
        Return:
            cropped image
        """
        found_left_x = False
        found_left_y = False
        found_right_x = False
        found_right_y = False  
        size = image.shape[0]
        for i in range(size):
            if not found_left_x and np.sum(image[i, :]) > 0:
                left_x = i
                found_left_x = True
            if not found_left_y and np.sum(image[:, i]) > 0:
                left_y = i
                found_left_y = True
            if not found_right_x and np.sum(image[size - 1 - i, :]) > 0:
                right_x = size - 1 - i
                found_right_x = True
            if not found_right_y and np.sum(image[:, size - 1 - i]):
                right_y = size - 1 - i
                found_right_y = True
            if found_right_x and found_right_y and found_left_x and found_left_y:
                break
        cropped_image = image[left_x:right_x, left_y:right_y]
        return cropped_image
