"""File contains batch class"""
import numpy as np
from skimage import color

from dataset.dataset import ImagesBatch, action, inbatch_parallel

class TwoMnistBatch(ImagesBatch):
    """ Batch class which create colorize image """

    components = 'images', 'labels', 'color', 'first_number', 'second_number', 'indices'

    @action
    def normalize_images(self):
        """Normalize pixel values to (0, 1)."""
        self.images = self.images / 255. # pylint: disable=attribute-defined-outside-init
        return self

    @action
    @inbatch_parallel(init='init_func', post='assemble', components=['images', 'color', 'first_number',
                                                                     'second_number'])
    def concat_and_colorize_images(self, ind):
        """ From a black and white image makes either a red or blue image
        with probability = percent_blue.
        Parameters
        ----------
        image : np.array
            input image
        Returns
        -------
            colorized image
        """
        image = self.get(ind, 'images')
        label = self.get(ind, 'labels')
        shape = image.shape[1:3]
        image = color.gray2rgb(image).reshape(-1, *shape, 3)

        indices = np.array([0, 1])
        np.random.shuffle(indices)
        col = np.array([[0., 0., 1.], [1., 0., 0.]])[indices]

        return [np.hstack((image[0] * col[0], image[1] * col[1])), indices[0], *label]

    def init_func(self, components):
        """ Create queue to parallel.
        Resurns
        -------
            Array with parallel indices """
        return [{'ind':[i, np.random.choice(self.indices)]} for i in self.indices]