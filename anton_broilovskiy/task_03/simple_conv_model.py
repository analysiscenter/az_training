""" File with convolution network """

import sys

import tensorflow as tf

sys.path.append('../../dataset')
from dataset.models.tf import TFModel
from dataset.models.tf.layers import conv_block


class ConvModel(TFModel):
    """ Class to build conv model """

    @classmethod
    def default_config(cls):
        """ Default parameters for conv model.

         Returns
        -------
        config : dict
            default parameters to network
        """
        config = TFModel.default_config()
        config['body'].update(dict(filters=[16, 32, 64], kernel_size=[7, 5, 3],
                              pool_size=2, pool_strides=[3, 3, 2], layout='cpna'))
        config['head'].update(dict(layout='fafa'))
        return config

    @classmethod
    def body(cls, inputs, layout, filters, name='body', **kwargs):
        """ Base layers.

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        layout : str
            a sequence of blocks
        filters: list
            number of filters in convolutions
        arch : dict
            parameters for each block
        name : str
            scope name

        Returns
        -------
        tf.Tensor
        """
        with tf.variable_scope(name):
            x = conv_block(inputs, layout*3, filters, **kwargs)
        return x