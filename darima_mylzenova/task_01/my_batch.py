''' Regression models implementation using dataset and tensorflow '''
import sys
import numpy as np
sys.path.append("../..")
from dataset.dataset import Batch, action

from sklearn.preprocessing import scale


class MyBatch(Batch):
    ''' A Batch with logistic regression model '''

    @property
    def components(self):
        ''' Define components '''
        return "features", "labels"

    @action()
    def preprocess_linear_data(self):
        ''' Normalize data '''
        scale(self.features, axis=0, with_mean=True, with_std=True, copy=False)
        return self

    @action()
    def preprocess_binary_data(self):
        ''' Change label of the second class to '-1' instead of 0'''
        self.labels[:] = 2*self.labels - np.ones((len(self.labels), 1), dtype=np.float64)
        return self