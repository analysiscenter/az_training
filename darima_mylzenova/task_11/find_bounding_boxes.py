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
    components = 'images', 'labels', 'coordinates', 'noise', 'other_coordinates', 'other_labels'
    
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
        image_size = self.images.shape[1]
        if args[0] == 'random_noise':
            noise = args[1] * np.random.random((image_size, image_size, 1)) * image.max()
        elif args[0] == 'mnist_noise':
            level, n_fragments, size, distr = args[1:]
            ind_for_noise = np.random.choice(self.images.shape[0], n_fragments)
            images = [self.images[i] for i in ind_for_noise]
            coordinates = [self.coordinates[i] for i in ind_for_noise]
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
    def add_noise(self, ind):
        if self.images.shape[-1] != 1:
            return np.expand_dims(np.max([self.get(ind, 'images'), self.get(ind, 'noise')], axis=0), axis=-1)
        else:
            return np.max([self.get(ind, 'images'), np.expand_dims(self.get(ind, 'noise'), axis=-1)], axis=0)
    
    def uniform(self, image_size, fragment_size):
        """Uniform distribution of fragmnents on image."""
        return np.random.randint(0, image_size-fragment_size, 2)

    def normal(self, image_size, fragment_size):
        """Normal distribution of fragmnents on image."""
        return list([int(x) for x in np.random.normal((image_size-fragment_size)/2,
                                                      (image_size-fragment_size)/4, 2)])

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
        pure_mnist = np.squeeze(image)
        large_mnist = np.zeros((new_size, new_size))

        all_indices = self.images.shape[0]

        other_label= []
        other_coord = []
        for i in range(num_others):
            random_idx = np.random.randint(0, all_indices)
            random_image = np.squeeze(self.images[random_idx])
            try:
                random_cropped = self.find_and_crop(random_image)
            except Exception as e:
                print('find_and_crop error')
                return ValueError
            width, height = random_cropped.shape
#             height, width = random_cropped.shape
            new_x = np.random.randint(0, new_size - width)
            new_y = np.random.randint(0, new_size - height)
#             new_x_2, new_y_2 = min(new_x + width, new_size), min(new_y + height, new_size)
            new_x_2, new_y_2 = new_x + width, new_y + height
#             print(new_x, ' ', new_y, 'new_x, new_y, ', width, ' ', height, ' width, height')
#             plt.imshow(random_cropped)
#             plt.show()
#             quit()
            large_mnist[new_x:new_x_2, new_y:new_y_2] = random_cropped
            new_x, new_y = max(new_x - 4, 0), max(new_y - 4, 0)
            new_x_2, new_y_2 = min(new_x + width + 4, new_size), min(new_y + height + 4, new_size)

            other_coord.append([new_x, new_y, new_x_2, new_y_2])
            other_label.append(self.labels[random_idx])
        
        cropped = self.find_and_crop(pure_mnist)
        width, height = cropped.shape
#         height, width = cropped.shape
        new_x = np.random.randint(0, new_size - width)
        new_y = np.random.randint(0, new_size - height)

#         new_x, new_y = np.random.randint(0, new_size - size, 2)
        new_x_2, new_y_2 = new_x + width, new_y + height

        large_mnist[new_x:new_x_2, new_y:new_y_2] = cropped
        
        new_x, new_y = max(new_x - 4, 0), max(new_y - 4, 0)
        new_x_2, new_y_2 = min(new_x + width + 4, new_size), min(new_y + height + 4, new_size)

        large_mnist = np.expand_dims(large_mnist, axis=3)
        return large_mnist, [new_x, new_y, new_x_2, new_y_2], other_coord, other_label


#     @action
#     @inbatch_parallel(init='init_function', post='post_function', target='threads')
#     def shift_flattened_pic(self, idx, max_margin=8):
#         """ Apply random shift to a flattened pic
        
#         Args:
#             idx: index in the self.images of a pic to be flattened
#         Return:
#             flattened shifted pic
#         """
        
#         pic = self.images[idx]
#         padded = np.pad(pic, pad_width=[[max_margin, max_margin], [max_margin, max_margin]], 
#                         mode='minimum')
#         left_lower = np.random.randint(2 * max_margin, size=2)
#         slicing = (slice(left_lower[0], left_lower[0] + 28),
#                    slice(left_lower[1], left_lower[1] + 28))
#         res = padded[slicing]
#         return res


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
