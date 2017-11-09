Skip to content
This repository
Search
Pull requests
Issues
Marketplace
Explore
 @dmylzenova
 Sign out
 Unwatch 6
  Unstar 2
 Fork 0 analysiscenter/az_training Private
 Code  Issues 0  Pull requests 0  Projects 0  Wiki  Insights
Branch: anton_broilovs… Find file Copy pathaz_training/task_08/resnet_model.py
2874a9d  11 hours ago
@anton-br anton-br Add functions to resnet model
1 contributor
RawBlameHistory     
276 lines (234 sloc)  12 KB
""" ResNetModel class
"""
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
import sys

import tensorflow as tf
import numpy as np

sys.path.append("..")

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf.layers.pooling import global_average_pooling


class ResNetModel(TFModel):
    ''' Universal Resnet model constructor
    '''
    def _build(self, *args, **kwargs):
        '''
        Builds a ResNet model.
        Parameters are taken from the config
        ----------
        input_config: a dict containing
            dim_shape: list, mandatory
                Shape of the input tenror including number of channels.
            data_format: str, either "channels_last" or "channels_first"
                 It specifies which data format convention will follow. Default value is "channels_last".
            n_classes: int
                Number of classes.
            length_factor: list of length 4 with int elements
                Specifies how many Reidual blocks will be of the same feature maps dimension.
                Recall that ResNet can have [64, 128, 256, 512] output feature maps thefore there are 4 types of
                sequences of ResNet blocks whith the same n_filters parameter.
                So the length_factor = [1, 2, 3, 4] will make one block with 16 feature maps, 2 blocks with 32,
                3 blocks with 64, 4 with 128.
            layout: str - a sequence of layers
                c - convolution
                n - batch normalization
                a - activation
                p - max pooling
                Default is 'cna'.
            bottleneck: bool
                If True all residual blocks will have 1x1, 3x3, 1x1 convolutional layers.
            max_pool: bool
                Whether to do maxpulling with stride 2 after first convolutional layer.
            conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
            downsampling_keys: list of length 4 with boolean elements
                Defines whether to do first downsampling convolution with stride 2 in every sequence
                of ResNet blocks.
                For example, [True, False, False, False] will make downsampling convoultion with stride 2 only
                in the beginning of the network and other expansions of number of feature maps will be performed
                without downsampling.
            dropout_rate: float in [0, 1]. Default is 0.
        Returns
        -------
        '''
        _ = args, kwargs

        names = ['images', 'labels']
        _, transformed_placeholders = self._make_inputs(names)

        input_config = self.get_from_config('inputs', None)
        if input_config is None:
            raise ValueError('you should specify configuration of input data')

        data_format = input_config[names[0]].get('data_format', 'channels_last')
        n_classes = input_config[names[1]].get('shape', 2)

        dim = self.get_from_config('dim', None)
        if dim is None:
            raise ValueError('dim should be customized in config')
        filters = self.get_from_config('filters', [64, 128, 256, 512])
        length_factor = self.get_from_config('length_factor', [1, 1, 1, 1])
        strides = self.get_from_config('strides', [2, 1, 1, 1])

        layout = self.get_from_config('layout', 'cna')

        bottleneck = self.get_from_config('bottleneck', False)
        skip = self.get_from_config('skip', True)
        stochastic = self.get_from_config('stochastic', False)
        se_block = self.get_from_config('se_block', False)
        if se_block:
            if isinstance(se_block, dict):
                if 'C' not in se_block:
                    se_block['C'] = 128
                if 'r' not in se_block:
                    se_block['r'] = 8
            elif isinstance(se_block, bool):
                se_block = dict(C=128, r=8)
            else:
                raise ValueError('se_block must be dict or bool not {}'.format(type(se_block)))

        bottelneck_factor = self.get_from_config('bottelneck_factor', 4)
        max_pool = self.get_from_config('max_pool', False)
        conv_params = self.get_from_config('conv_params', {'conv': {}})
        dropout_rate = self.get_from_config('dropout_rate', 0.)

        if max_pool:
            first_layout = layout + 'p'
        else:
            first_layout = layout

        threshold = np.linspace(1, 0.5, sum(length_factor))

        net = conv_block(dim, transformed_placeholders['images'], filters[0], (7, 7), first_layout,\
                         name='first_convolution', strides=2,\
                         is_training=self.is_training, pool_size=3, pool_strides=2)

        for index, block_length in enumerate(length_factor):
            for block_number in range(block_length):
                net = self.conv_block(dim, net, filters[index], layout, str(index), block_number, \
                                       conv_params['conv'], strides=strides[index], is_training=self.is_training, \
                                       data_format=data_format, bottleneck=bottleneck, \
                                       bottelneck_factor=bottelneck_factor, dropout_rate=dropout_rate, skip=skip, \
                                       stochastic=stochastic, threshold=threshold[index+block_number], \
                                       se_block=se_block)

        net = tf.identity(net, name='conv_output')

        net = global_average_pooling(dim, net, data_format)
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, n_classes)

        predictions = tf.identity(net, name='predictions')

        tf.nn.softmax(net, name='predicted_prob')
        labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(transformed_placeholders['labels'], axis=1), tf.float32, 'true_labels')

        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

    #@staticmethod
    def conv_block(self, dim, input_tensor, filters, layout, name, block_number, conv_params, strides, \
                   is_training=True, data_format='channels_last', bottleneck=False, bottelneck_factor=4, \
                   dropout_rate=0., skip=False, stochastic=False, threshold=0., se_block=False, \
                   activation=tf.nn.relu):
        """ Building block of ResNet architecture
        """
        # rewrite functions to functools partial
        # stochastic doesn't work
        if block_number != 0:
            strides = 1

        name = name + '/' + str(block_number)
        with tf.variable_scope(name):

            if bottleneck:
                output_filters = filters * bottelneck_factor
                x = self.bottleneck_block(dim, input_tensor, filters, output_filters, layout, conv_params, strides,\
                                          is_training, data_format, activation)
            else:
                output_filters = filters
                x = self.normal_block(dim, input_tensor, filters, layout, conv_params, strides, is_training, \
                                 data_format, dropout_rate, activation)
            if se_block:
                x = self.se_block(x, se_block)


            shortcut = self.add_shortcut(dim, input_tensor, output_filters, strides, data_format, skip, \
                                         is_training, conv_params)

            off = 1.
            if stochastic:
                off = tf.cond(is_training, \
                          lambda: tf.where(tf.random_uniform([1, ], 0, 1) > (1 - threshold),\
                          tf.ones([1, ]), tf.zeros([1, ])), lambda: tf.ones([1, ]) * threshold)[0]

            x = tf.add(shortcut, tf.cond(tf.less_equal(off, 0), lambda: tf.zeros_like(x, tf.float32), lambda: x)) * off
            x = tf.nn.relu(x, name='output')
        return x

    def bottleneck_block(self, dim, input_tensor, filters, output_filters, layout, conv_params, strides, \
                         is_training=True, data_format='channels_last', activation=tf.nn.relu):
        """ create one bottleneck block """
        filters = [filters, filters, output_filters]
        x = conv_block(dim, input_tensor, filters, [1, 3, 1], layout*3, 'bottle_block', strides=[strides, 1, 1], \
                       padding='same', data_format=data_format, activation=activation, \
                       is_training=is_training, conv=conv_params)
        return x

    def normal_block(self, dim, input_tensor, filters, layout, conv_params, strides, \
                         is_training=True, data_format='channels_last', dropout_rate=0., activation=tf.nn.relu):
        """ create normal block """
        x = conv_block(dim, input_tensor, [filters]*2, 3, layout*2, 'normal_block', strides=[strides, 1],\
                       padding='same', data_format=data_format, activation=activation, is_training=is_training,\
                       dropout_rate=dropout_rate, conv=conv_params)
        return x

    def se_block(self, x, se_block):
        """ create se block """
        r = se_block['r']
        C = se_block['C']
        full = tf.reduce_mean(x, [1, 2])
        full = tf.reshape(full, [-1, 1, 1, C])
        full = tf.layers.dense(full, int(C/r), activation=tf.nn.relu, \
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='first_dense_se_block')
        full = tf.layers.dense(full, C, activation=tf.nn.sigmoid, \
            kernel_initializer=tf.contrib.layers.xavier_initializer(), name='second_dense_se_block')
        return x * full

    def add_shortcut(self, dim, input_tensor, output_filters, strides, data_format, skip, \
                     is_training, conv_params):
        """ create shortcut connetion """
        if data_format == 'channels_last':
            input_filters = input_tensor.get_shape()[-1]
        else:
            input_filters = input_tensor.get_shape()[0]
        if skip:
            return 0.
        elif input_filters != output_filters:
            return conv_block(dim, input_tensor, output_filters, 1, 'c', strides=strides, padding='same', \
                                  is_training=is_training, conv=conv_params, name='shortcut')
        else:
            return input_tensor

class ResNet152(ResNetModel):
    ''' An original ResNet-152 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 8, 36, 3]
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet101(ResNetModel):
    ''' An original ResNet-101 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 23, 3]
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet50(ResNetModel):
    ''' An original ResNet-50 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet34(ResNetModel):
    ''' An original ResNet-34 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()



class ResNet18(ResNetModel):
    ''' An original ResNet-18 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [2, 2, 2, 2]
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()
© 2017 GitHub, Inc.
Terms
Privacy
Security
Status
Help
Contact GitHub
API
Training
Shop
Blog
About