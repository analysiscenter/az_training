"""Batch class for water meter task"""
import sys
import re

import scipy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from dataset.dataset import ImagesBatch, action, inbatch_parallel

class WaterBatch(ImagesBatch):
    """Class to create batch with water meter"""
    components = 'images', 'labels', 'coordinates', 'indices', 'numbers'

    def _init_component(self, *args, **kwargs):
        """Create and preallocate a new attribute with the name ``dst`` if it
        does not exist and return batch indices."""
        _ = args, kwargs
        dst = kwargs.get('dst')
        if dst is None:
            raise KeyError('dst argument must be specified')
        if not hasattr(self, dst):
            setattr(self, dst, np.array([None] * len(self.index)))
        return self.indices

    @action
    @inbatch_parallel(init='_init_component', src='images', dst='cropped', target='threads')
    def crop_to_bbox(self, index, *args, src='images', dst='cropped', **kwargs):
        """Create cropped attr with crop image use ``coordinates``"""
        _ = args, kwargs
        image = self.get(index, 'images')
        x, y, x1, y1 = self.get(index, 'coordinates')
        i = self.get_pos(None, 'images', index)
        dst_data = image[y:y+y1, x:x+x1]
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init='_init_component', src='cropped', dst='sepcrop', target='threads')
    def crop_to_numbers(self, index, *args, shape=(64, 32), src='cropped', dst='sepcrop', **kwargs):
        """Crop image with 8 number to 8 images with one number"""
        def _resize(img, shape):
            factor = 1. * np.asarray([*shape]) / np.asarray(img.shape[:2])
            if len(img.shape) > 2:
                factor = np.concatenate((factor, [1.] * len(img.shape[2:])))
            new_image = scipy.ndimage.interpolation.zoom(img, factor, order=3)
            return new_image

        _ = args, kwargs
        i = self.get_pos(None, 'cropped', index)
        image = getattr(self, 'cropped')[i]
        step = round(image.shape[1]/8)
        numbers = np.array([_resize(image[:, i:i+step], shape) for i in range(0, image.shape[1], step)] + \
                           [None])[:-1]
        if len(numbers) > 8:
            numbers = numbers[:-1]
        getattr(self, dst)[i] = numbers

    @action
    @inbatch_parallel(init='_init_component', src='labels', dst='labels', target='threads')
    def crop_labels(self, index, *args, src='labels', dst='labels', **kwargs):
        """Create labels"""
        _ = args, kwargs
        i = self.get_pos(None, 'labels', index)
        label = getattr(self, 'labels')[i]

        more_label = np.array([int(i) for i in label.replace('.', '')] + [None])[:-1]

        getattr(self, dst)[i] = more_label

    @inbatch_parallel(init='indices', post='assemble')
    def _load_jpg(self, ind, src, components=None):
        _ = components, self
        images = plt.imread(src + ind + '.jpg')
        return images

    def _load_csv(self, src, components=None, *args, **kwargs):
        _ = args, kwargs
        if src[-4:] != '.csv':
            src += '.csv'
        _data = pd.read_csv(src, *args, **kwargs)

        if 'file_name' in _data.columns:
            _data = [_data[_data['file_name'] == ind]['counter_value'].values[0] for ind in self.indices]

        else:
            indices = [int(ind[1:3]) for ind in self.indices]
            coord = []

            for ind in indices:
                string = _data.loc[ind].values[0][36:-7]
                coord.append(list([int(i) for i in re.sub('\\D+', ' ', string).split(' ')[1:]]))
            _data = np.array(coord)
        setattr(self, components, _data)

    @action
    def load(self, src, fmt=None, components=None, *args, **kwargs):
        """
        Parameters
        ----------
        src :
            a source (e.g. an array or a file name)

        fmt : str
            a source format, one of 'jpg' or 'csv'

        components : None or str or tuple of str
            components to load

        *args :
            other parameters are passed to format-specific loaders
        **kwargs :
            other parameters are passed to format-specific loaders
        """
        if fmt == 'jpg':
            self._load_jpg(src, components)
        elif fmt == 'csv':
            self._load_csv(src, components, *args, **kwargs)
        else:
            super().load(src, fmt, components, *args, **kwargs)
        return self

    @action
    def normalize_images(self):
        """ Normalize pixel values to (0, 1). """
        self.images = self.images / 255. # pylint: disable=attribute-defined-outside-init
        return self
