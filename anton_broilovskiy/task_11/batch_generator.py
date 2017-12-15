"""File contains batch class"""
import numpy as np
from skimage import color

from dataset.dataset import ImagesBatch, action, inbatch_parallel

class TwoMnistBatch(ImagesBatch):
    """ Batch class which create colorize image """

    components = 'images', 'labels', 'first_color', 'second_color', 'first_number', 'second_number', 'indices'

    @action
    def normalize_images(self):
        """ Normalize pixel values to (0, 1). """
        self.images = self.images / 255. # pylint: disable=attribute-defined-outside-init
        return self

    @action
    @inbatch_parallel(init='indices', post='assemble', components=['images', 'first_color', 'second_color',\
                     'first_number', 'second_number'])
    def colorize_images(self, ind, colors):
        """ Colorize input image to given colors

        Parameters
        ----------
        ind : numpy.unit8
            index

        colors : np.array of npo.arrays
            new colors can be shape = (2, 3) or (1, 3)

        Returns
        -------
        colorized_image : np.array
            new image

        index : list
            sequence of new colors. (1, 0) or (0, 1)

        label : list
            labels to image
        """

        image = self.get(ind, 'images')
        label = self.get(ind, 'labels')

        shape = image.shape[:2]
        image = color.gray2rgb(image).reshape(*shape, 3)
        index = np.arange(len(colors))
        np.random.shuffle(index)
        colors = np.array(colors)[index]
        if isinstance(label, np.uint8) or len(index) < 2:
            label = np.hstack([label, -1])
            index = np.hstack([index, 0])

        colorized_image = np.hstack((image[:, :int(shape[1]/2)] * colors[0], image[:, int(shape[1]/2):] * colors[-1]))
        return [colorized_image, *index, *label]

    @action
    @inbatch_parallel(init='init_func', post='assemble', components=['images', 'labels'])
    def gluing_of_images(self, ind):
        """ Gluing two image by y axis

        Parameters
        ----------
        ind : numpy.uint8
            index

        Returns
        -------
        image : np.array
            new image

        label : list
            list len = 2 with answers to new image"""
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
