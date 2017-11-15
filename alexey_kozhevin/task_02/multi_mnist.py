import sys

from dataset.opensets import MNIST
import pickle
from dataset import Batch, action, inbatch_parallel, any_action_failed
import numpy as np
from skimage.transform import resize
import scipy.ndimage

class MultiMNIST(Batch):

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.masks = None
        self.images = None
        self.labels = None

    @property
    def components(self):
        """Define components."""
        return ('images', 'labels', 'masks')

    @action
    def load_images(self):
        """Load MNIST images from file."""
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.images = pickle.load(file)[self.indices].reshape(-1, 28, 28)
        with open('../mnist/mnist_labels.pkl', 'rb') as file:
            self.labels = np.argmax(pickle.load(file)[self.indices], axis=-1)
        return self

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def create_multi(self, ind, *args, **kwargs):
        """Create multimnist image
        """
        image_shape = kwargs['image_shape']
        max_digits = kwargs['max_digits']
        n_digits = np.random.randint(1, max_digits+1)
        digits = np.random.choice(len(self.images), min([n_digits, len(self.images)]))
        large_image = np.zeros(image_shape)
        mask = np.zeros(image_shape)

        for i in digits:
            image = self.images[i]	
            shape = [np.random.randint(20, 30)] * 2
            factor = 1. * np.asarray([*shape]) / np.asarray(image.shape[:2])
            image = scipy.ndimage.interpolation.zoom(image, factor, order=3)
            x1 = np.random.randint(0, image_shape[0]-image.shape[0])
            y1 = np.random.randint(0, image_shape[1]-image.shape[1])
            x2 = x1 + image.shape[0]
            y2 = y1 + image.shape[1]
            mask_region = mask[x1:x2, y1:y2]
            mask[x1:x2, y1:y2] = np.max([mask_region, (self.labels[i]+1)*(image > 0.1)], axis=0)
            old_region = large_image[x1:x2, y1:y2]
            large_image[x1:x2, y1:y2] = np.max([image, old_region], axis=0)
        return large_image, mask

    def init_func(self, *args, **kwargs): # pylint: disable=unused-argument
        """Create tasks."""
        return [i for i in range(self.images.shape[0])]

    def post_func(self, list_of_res, *args, **kwargs):
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, masks = list(zip(*list_of_res))
            self.images = np.expand_dims(np.array(images), axis=-1)
            self.masks = np.array(masks) - 1
            self.masks[self.masks == -1] = 10
            return self
