"""batch generator file"""
import sys
import numpy as np
sys.path.append('..')
sys.path.append('../..')
from task_11.batch_generator import TwoMnistBatch
from dataset.dataset import action, inbatch_parallel

class SixNumbersBatch(TwoMnistBatch):
    """class with something """
    components = 'images', 'labels', 'first', 'second', 'third', 'fourth', 'fifth', 'indices'

    @action
    @inbatch_parallel(init='init_func', post='assemble', \
                      components=['images', 'first', 'second', 'third', 'fourth', 'fifth', 'labels'])
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

        return [np.hstack((image[0], image[1], image[2], image[3], image[4])), label, label]

    def init_func(self, components, **kwargs):
        """ Create queue to parallel.
        Resurns
        -------
            Array with parallel indices """
        _ = components, kwargs
        return [{'ind':np.array([i, np.random.choice(self.indices, 4)])} for i in self.indices]
