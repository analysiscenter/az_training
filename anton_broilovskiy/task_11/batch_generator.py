"""File contains batch class"""
import numpy as np
from skimage import color

from dataset.dataset import ImagesBatch, action, inbatch_parallel

class TwoMnistBatch(ImagesBatch):
    """ Batch class which create colorize image """
    components = 'images', 'labels', 'indices'

    @action
    @inbatch_parallel(init='images', post='assemble')
    def create_color(self, image, percent_blue=0.5):
        """ From a black and white image makes either a red or blue image
        with probability = percent_blue.

        Parameters
        ----------

        image : np.array
            input image

        percent_blue : int, optional
            create blue picture with percentage = percent_blue. Default = 0.5


        """
        if np.max(image) > 1.:
            image = image / 255.

        if image.shape[-1] < 3:
            image = color.gray2rgb(image).reshape(28, 28, 3)

        ind_of_color = np.random.choice([0, 1], p=[percent_blue, 1 - percent_blue])
        col = np.array([[0., 0., 1.], [1., 0., 0.]])[ind_of_color]
        return image * col

    @action
    @inbatch_parallel(init='init_func', post='assemble')
    def concatenate_images(self, ind):
        """ Joining two pictures by horizontal axis

        Parameters
        ----------
        ind : list with size 2
            indices

        Returns
        -------
            One concat image
        """
        images = self.get(ind, 'images')
        return np.hstack((images[0], images[1]))


    def init_func(self):
        """ Create queue to parallel.
        Resurns
        -------
            Array with parallel indices """
        return [{'ind':[i, np.random.choice(self.indices)]} for i in self.indices]
