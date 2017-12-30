"""Batch class for water meter task"""
import sys
import re

import scipy
import dill
import blosc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from dataset.dataset import ImagesBatch, action, inbatch_parallel

class WaterBatch(ImagesBatch):
    """Class to create batch with water meter"""
    components = 'images', 'labels', 'coordinates', 'indices'

    @action
    @inbatch_parallel(init='indices', src='images', post='assemble')
    def normalize_images(self, ind, src='images'):
        """ Normalize pixel values to (0, 1). """
        image = self.get(ind, src)
        normalize_image = image / 255.
        return normalize_image

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
    def crop_to_bbox(self, ind, *args, src='images', dst='cropped', **kwargs):
        """Create cropped attr with crop image use ``coordinates``

        Parameter
        ----------
        ind : str or int
        dataset index

        src : str
        the name of the placeholder with data

        dst : str
        the name of the placeholder in witch the result will be recorded"""
        _ = args, kwargs
        image = self.get(ind, src)
        x, y, width, height = self.get(ind, 'coordinates')
        i = self.get_pos(None, src, ind)
        dst_data = image[y:y+height, x:x+width]
        getattr(self, dst)[i] = dst_data

    @action
    @inbatch_parallel(init='_init_component', src='cropped', dst='sepcrop', target='threads')
    def crop_to_numbers(self, ind, *args, shape=(64, 32), src='cropped', dst='sepcrop', **kwargs):
        """Crop image with 8 number to 8 images with one number

        Parameters
        ----------
        ind : str or int
        dataset index

        shape : tuple or list
        shape of output image

        src : str
        the name of the placeholder with data

        dst : str
        the name of the placeholder in witch the result will be recorded"""

        def _resize(img, shape):
            factor = 1. * np.asarray([*shape]) / np.asarray(img.shape[:2])
            if len(img.shape) > 2:
                factor = np.concatenate((factor, [1.] * len(img.shape[2:])))
            new_image = scipy.ndimage.interpolation.zoom(img, factor, order=3)
            return new_image

        _ = args, kwargs
        i = self.get_pos(None, src, ind)
        image = getattr(self, src)[i]
        numbers = np.array([_resize(img, shape) for img in np.array_split(image, 8, axis=1)] + [None])[:-1]

        getattr(self, dst)[i] = numbers

    @inbatch_parallel(init='_init_component', src='labels', dst='labels', target='threads')
    def _crop_labels(self, ind, *args, src='labels', dst='labels', **kwargs):
        _ = args, kwargs
        i = self.get_pos(None, src, ind)
        label = getattr(self, src)[i]
        more_label = np.array([int(i) for i in label.replace('.', '')] + [None])[:-1]
        getattr(self, dst)[i] = more_label

    @inbatch_parallel(init='indices', post='assemble')
    def _load_jpg(self, ind, src, components=None, *args, **kwargs):
        _ = components, self
        images = plt.imread(src + ind + '.jpg', *args, **kwargs)
        return images

    def _load_csv(self, src, components=None, *args, **kwargs):
        _ = args
        crop_labels = kwargs.pop('crop_labels') if 'crop_labels' in kwargs.keys() else False
        if src[-4:] != '.csv':
            src += '.csv'
        _data = pd.read_csv(src, *args, **kwargs)
        if 'file_name' in _data.columns: # pylint: disable=no-member
            _data = [_data[_data['file_name'] == ind]['counter_value'].values[0] for ind in self.indices]

        else:
            indices = [int(ind[1:3]) for ind in self.indices]
            coord = []

            for ind in indices:
                string = _data.loc[ind].values[0][36:-7] # pylint: disable=no-member
                coord.append(list([int(i) for i in re.sub('\\D+', ' ', string).split(' ')[1:]]))
            _data = np.array(coord)
        setattr(self, components, _data)
        if crop_labels:
            self._crop_labels(self.indices)

    @inbatch_parallel(init='indices', post='assemble')
    def _load_blosc(self, ind, src=None, components=None, *args, **kwargs):
        _ = args, kwargs, components
        file_name = self._get_file_name(ind, src, 'blosc')
        with open(file_name, 'rb') as file:
            data = dill.loads(blosc.decompress(file.read()))
        return data

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
            self._load_jpg(src, components, *args, **kwargs)
        elif fmt == 'csv':
            self._load_csv(src, components, *args, **kwargs)
        elif fmt == 'blosc':
            self._load_blosc(src, components, *args, **kwargs)
        else:
            raise ValueError("Unknown format " + fmt)
        return self
