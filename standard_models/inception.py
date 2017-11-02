""" Contains GoogLeNet network: https://arxiv.org/abs/1409.4842 """
import sys

import tensorflow as tf
sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf.layers.pooling import max_pooling, global_average_pooling

class InceptionV1(TFModel):
    """ implementation GoogLeNet model

    Parameters:
    -----------
    data_format: str {'channels_last', 'channels_first'}

    dim : int
    spacial dimension of input without the number of channels
    """

    def _build(self, *args, **kwargs):
        _ = args, kwargs

        data_format = self.get_from_config('data_format')
        dim = self.get_from_config('dim')

        names = ['images', 'labels']
        _, transformed_placeholders = self._make_inputs(names)

        with tf.variable_scope('inception'):
            net = conv_block(dim=dim, input_tensor=transformed_placeholders['images'], filters=64, kernel_size=7,\
                             strides=2, layout='cp', data_format=data_format, pool_size=3, pool_strides=2)
            net = conv_block(dim=dim, input_tensor=net, filters=64, kernel_size=3, layout='c',\
                             data_format=data_format)
            net = conv_block(dim=dim, input_tensor=net, filters=192, kernel_size=3, layout='cp',\
                             data_format=data_format, pool_size=3, strides=2)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[64, 96, 128, 16, 32, 32],\
                                  data_format=data_format, name='3a', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[128, 128, 192, 32, 96, 64],\
                                  data_format=data_format, name='3b', is_training=self.is_training)
            net = max_pooling(dim=dim, inputs=net, pool_size=3, strides=2, padding='same', data_format=data_format)

            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[192, 96, 208, 16, 48, 64],\
                                  data_format=data_format, name='4a', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[160, 112, 224, 24, 64, 64],\
                                  data_format=data_format, name='4b', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[128, 128, 256, 24, 64, 64],\
                                  data_format=data_format, name='4c', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[112, 144, 288, 32, 64, 64],\
                                  data_format=data_format, name='4d', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[256, 160, 320, 32, 128, 128],\
                                  data_format=data_format, name='4e', is_training=self.is_training)
            net = max_pooling(dim=dim, inputs=net, pool_size=3, strides=2, padding='same', data_format=data_format)

            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[256, 160, 320, 32, 128, 128],\
                                  data_format=data_format, name='5a', is_training=self.is_training)
            net = self.googlenet_block(dim=dim, input_tensor=net, filters=[384, 192, 384, 48, 128, 128],\
                                  data_format=data_format, name='5b', is_training=self.is_training)
            net = global_average_pooling(dim=dim, inputs=net, data_format=data_format)
            net = tf.layers.dropout(net, 0.4, training=self.is_training)
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10)

        self.statistic(tf.identity(net, name='predictions'), transformed_placeholders['labels'])



    def statistic(self, net, targets):
        """Added to graph some useful funstion like accuracy or preidctions
        Parameters:
        -----------
        net: tf.Tensor
        Network output

        targets: tf.Tensor
        Answers on the data
        """
        prob = tf.nn.softmax(net, name='prob_predictions')

        labels_hat = tf.cast(tf.argmax(prob, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(targets, axis=1), tf.float32, name='labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

    @staticmethod
    def googlenet_block(dim, input_tensor, filters, data_format, name, batch_norm=False, is_training=True):
        """ Function contains building block from googlenet achitecture

        Parameters:
        -----------
        dim: int
        spacial dimension of input without the number of channels

        input_tensor: tf.Tensor

        filters: list len 6
        [number of filters in one conv 1x1,
        -//- in conv 1x1 before conv 3x3,
        -//- in conv 3x3,
        -//- in conv 1x1 before conv 5x5,
        -//- in conv 5x5,
        -//- in conv 1x1 after max_pool]

        data_format: str {'channels_last', 'channels_first'}

        name: str
        name of block

        batch_norm: bool
        Use batch norm or not

        Returns:
        --------
        tf.Tensor - output tf.Tensor
        """
        if batch_norm:
            layout = 'cn'
        else:
            layout = 'c'

        with tf.variable_scope("block_" + name):
            block_1 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[0], kernel_size=1,\
                                 layout=layout, name='conv_1', data_format=data_format, is_training=is_training)

            block_1_3 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[1], kernel_size=1,\
                                   layout=layout, name='conv_1_3', data_format=data_format, is_training=is_training)
            block_3 = conv_block(dim=dim, input_tensor=block_1_3, filters=filters[2], kernel_size=1,\
                                 layout=layout, name='conv_3', data_format=data_format, is_training=is_training)

            block_1_5 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[3], kernel_size=1,\
                                   layout=layout, name='conv_1_5', data_format=data_format, is_training=is_training)
            block_5 = conv_block(dim=dim, input_tensor=block_1_5, filters=filters[4], kernel_size=1,\
                                 layout=layout, name='conv_5', data_format=data_format, is_training=is_training)

            conv_pool = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[5], kernel_size=1, \
                                   layout='p'+layout, name='conv_pool', data_format=data_format, pool_size=1,\
                                   pool_strides=1, is_training=is_training)

            if data_format == 'channels_last':
                return tf.concat([block_1, block_3, block_5, conv_pool], -1, name='output')
            elif data_format == 'channels_first':
                return tf.concat([block_1, block_3, block_5, conv_pool], 0, name='output')
            else:
                raise ValueError("data_format can be 'last' or 'first' not %d"% data_format)
