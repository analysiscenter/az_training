"""File contains batch class"""
import numpy as np
from skimage import color

from dataset.dataset import ImagesBatch, action, inbatch_parallel

class TwoMnistBatch(ImagesBatch):
    """ Batch class which create colorize image """

    components = 'images', 'labels', 'first_color', 'second_color', 'first_number', 'second_number', 'indices'

    @action
    def normalize_images(self):
        """Normalize pixel values to (0, 1)."""
        self.images = self.images / 255. # pylint: disable=attribute-defined-outside-init
        return self

    @action
    @inbatch_parallel(init='init_f', post='assemble', components=['images', 'first_color', 'second_color',\
                     'first_number', 'second_number'])
    def colorize_images(self, ind, colors):
        """ colorize images"""
        image = self.get(ind, 'images')
        shape = image.shape[:2]
        image = color.gray2rgb(image).reshape(*shape, 3)
        ind = np.arange(len(colors))
        np.random.shuffle(ind)
        colors = np.array(colors)[ind]

        label = self.get(ind, 'labels')
        if image.shape[1] <= image.shape[0]:
            label = np.hstack(([label], [-1]))
            return [image * colors, *[colors]*2, *label]

        colorized_image = np.hstack((image[:, :int(shape[1]/2)] * colors[0], image[:, int(shape[1]/2):] * colors[1]))
    return [colorized_image, *ind, *label]

    @action
    @inbatch_parallel(init='init_func', post='assemble', components=['images', 'labels'])
    def gluing_of_images(self, ind):
        """gluing images """
        image = self.get(ind, 'images')
        label = self.get(ind, 'labels')

        return [np.hstack((image[0], image[1])), label]

    def init_func(self, components, **kwargs):
        """ Create queue to parallel.
        Resurns
        -------
            Array with parallel indices """
        _ = components, kwargs
        return [{'ind':np.array([i, np.random.choice(self.indices)])} for i in self.indices]

    def init_f(self, components, **kwargs):
        """ Create queue to parallel.
        Resurns
        -------
            Array with parallel indices """
        _ = components, kwargs
        return [{'ind':i} for i in self.indices]
