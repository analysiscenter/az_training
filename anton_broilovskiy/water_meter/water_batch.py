"""Batch class for water meter task"""
import sys
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append('..')
from dataset import ImagesBatch, action, inbatch_parallel

class WaterBatch(ImagesBatch):
    """Class to create batch with water meter"""
    components = 'images', 'labels', 'coordinates', 'indices'

    @inbatch_parallel('indices', post='assemble')
    def _load_jpg(self, ind, src, components=None):
        _ = components, self
        images = plt.imread(src + ind + '.jpg')
        return images

    def _load_csv(self, src, components=None, *args, **kwargs):
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
            raise ValueError("Unknown format " + fmt)
        return self

    @action
    def normalize_images(self):
        """ Normalize pixel values to (0, 1). """
        self.images = self.images / 255. # pylint: disable=attribute-defined-outside-init
        return self
