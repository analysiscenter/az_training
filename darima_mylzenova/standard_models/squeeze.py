''' SqueezeNet Model '''

import sys

import tensorflow as tf

sys.path.append("..")

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf.layers.pooling import global_average_pooling

class SqueezeNet(TFModel):
    def _build(self, *args, **kwargs):
    