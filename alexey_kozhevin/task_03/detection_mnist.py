#pylint:disable=attribute-defined-outside-init

"""
Generate images with MNIST in random positions
"""
import sys
import numpy as np
import scipy.ndimage

sys.path.append('..')

from dataset import action, inbatch_parallel, any_action_failed
from dataset import ImagesBatch

class DetectionMnist(ImagesBatch):
    """Batch class for multiple MNIST."""

    components = ('images', 'labels', 'bboxes')

    @action
    @inbatch_parallel(init='indices', post='post_func_multi')
    def generate_images(self, ind, *args, **kwargs):
        _ = ind, args
        """ Create image with 'image_shape' and put MNIST digits in random locations resized to 'resize_to'. """
        image_shape = kwargs.get('image_shape', (64, 64))
        n_digits = kwargs.get('n_digits', (10, 20))
        resize_to = kwargs.get('resize_to', (28, 28))

        factor = 1. * np.asarray(resize_to) / np.asarray(self.images.shape[1:3])
        if isinstance(n_digits, (list, tuple)):
            n_digits = np.random.randint(*n_digits)
        elif isinstance(n_digits, int):
            digits = n_digits
        else:
            raise TypeError('n_digits must be int, tuple or list but {} was given'.format(type(n_digits).__name__))
        digits = np.random.choice(len(self.images), min([n_digits, len(self.images)]))
        large_image = np.zeros(image_shape)
        bboxes = []
        labels = []

        for i in digits:
            image = np.squeeze(self.images[i])
            image = scipy.ndimage.interpolation.zoom(image, factor, order=3)
            new_x = np.random.randint(0, image_shape[0]-image.shape[0])
            new_y = np.random.randint(0, image_shape[1]-image.shape[1])
            old_region = large_image[new_x:new_x+image.shape[0], new_y:new_y+image.shape[1]]
            large_image[new_x:new_x+image.shape[0], new_y:new_y+image.shape[1]] = np.max([image, old_region], axis=0)
            bboxes.append((new_x, new_y, image.shape[0], image.shape[1]))
            labels.append(self.labels[i])
        return large_image, np.array(bboxes), np.array(labels)

    def post_func_multi(self, list_of_res, *args, **kwargs): # pylint: disable=unused-argument
        """Post function for generate_multimnist_images."""
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, bboxes, labels = list(zip(*list_of_res))
            self.images = np.expand_dims(np.stack(images), axis=-1)
            self.labels = labels
            self.bboxes = bboxes
            return self
