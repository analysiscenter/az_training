""" ResNetModel class
"""
import sys

import tensorflow as tf

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
                 It specifies which data format convention will follow.
                 Default value is "channels_last".
            n_classes: int
                Number of classes.
            length_factor: list of length 4 with int elements
                Specifies how many Reidual blocks
                will be of the same feature maps dimension.
                Recall that ResNet can have [64, 128, 256, 512] output feature maps so
                there are 4 types of sequences of ResNet blocks with the same n_filters parameter.
                So the length_factor = [1, 2, 3, 4] will make one block with 16 feature maps,
                2 blocks with 32,
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
                For example, [True, False, False, False] will make downsampling convoultion with
                stride 2 only in the beginning of the network and other expansions of number of
                feature maps will be performed without downsampling.
            dropout_rate: float in [0, 1]. Default is 0.

        Returns
        -------
        '''
        
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('labels')
        dim = self.spatial_dim('images')
        data_format = self.data_format('images')




        # input_config = self.get_from_config('input', None)
        # if input_config == None:
        #     raise ValueError('you should specify configuration of input data')
        
        # dim_shape = input_config.get('shape', None)
        # data_format = input_config.get('data_format', 'channels_last')
        
        # n_classes = input_config.get('n_classes', 2)
        # dim = input_config.get('dim', None)
        # if dim == None:
        #     raise ValueError('dim should be customized in config')

        # if dim_shape == None:
        #     raise ValueError('dim_shape should be customized in config')


        filters = self.get_from_config('filters', [64, 128, 256, 512])
        length_factor = self.get_from_config('length_factor', [1, 1, 1, 1])
        strides = self.get_from_config('strides', [2, 1, 1, 1])

        layout = self.get_from_config('layout', 'cna')

        bottleneck = self.get_from_config('bottleneck', False)
        bottelneck_factor = self.get_from_config('bottelneck_factor', 4)

        max_pool = self.get_from_config('max_pool', False)
        conv_params = self.get_from_config('conv_params', {'conv': {}})


        dropout_rate = self.get_from_config('dropout_rate', 0.)
        kernel_size = self.get_from_config('kernel_size', 3)

        is_training = self.is_training


        # x = tf.placeholder(tf.float32, name='input_images')

        # x_reshaped = tf.reshape(x, shape=[-1] + dim_shape)

        if max_pool:
            first_layout = layout + 'p'
        else:
            first_layout = layout

        net = conv_block(dim, inputs['images'], filters[0], 7, first_layout, name='0', strides=2, \
                         is_training=is_training, pool_size=3, pool_strides=2)

        for index, block_length in enumerate(length_factor):
            for block_number in range(block_length):
                net = self.conv_block(dim, net, kernel_size, filters[index], layout, str(index), \
                                      block_number, conv_params['conv'], strides=strides[index], \
                                      is_training=is_training, data_format=data_format, \
                                      bottleneck=bottleneck, bottelneck_factor=bottelneck_factor, \
                                      dropout_rate=dropout_rate)

        net = tf.identity(net, name='conv_output')

        net = global_average_pooling(dim, net, data_format)
        net = tf.contrib.layers.flatten(net)


        net = tf.layers.dense(net, n_classes)
        predictions = tf.identity(net, name='predictions')

        probs = tf.nn.softmax(net, name='predicted_prob')
        # y_ = tf.placeholder(tf.float32, [None, n_classes], name='targets')
        labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(inputs['labels'], axis=1), tf.float32, 'true_labels')

        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), \
                                  tf.float32), name='accuracy')


    @staticmethod
    def conv_block(dim, input_tensor, kernel_size, filters, layout, name, block_number, conv_params,
                   strides, is_training=True, data_format='channels_last', bottleneck=False,
                   bottelneck_factor=4, dropout_rate=0.):

        if block_number != 0:
            strides = 1

        name = name + '/' + str(block_number)
        with tf.variable_scope(name):

            if bottleneck:
                output_filters = filters * bottelneck_factor

                x = conv_block(dim, input_tensor, [filters, filters, output_filters], [1, 3, 1], \
                               layout+layout+layout, '1', [strides, 1, 1], padding='same', \
                               data_format=data_format, is_training=is_training, conv=conv_params)

                # x = conv_block(dim, input_tensor, filters, 1, layout, '1', strides=strides, \
                #                padding='same', data_format=data_format, activation=tf.nn.relu, \
                #                is_training=is_training, conv=conv_params)

                # x = conv_block(dim, x, filters, kernel_size, layout, '2', padding='same', \
                #                data_format=data_format, activation=tf.nn.relu, \
                #                is_training=is_training, conv=conv_params)


                # x = conv_block(dim, x, output_filters, 1, layout, '3', padding='same', \
                #                data_format=data_format, activation=tf.nn.relu, \
                #                is_training=is_training, conv=conv_params)

            else:
                output_filters = filters

                x = conv_block(dim, input_tensor, [filters, filters], [3, 3], layout+'d'+layout, \
                               '1', strides=[strides, 1], padding='same', data_format=data_format, \
                               activation=tf.nn.relu, is_training=is_training, \
                               dropout_rate=dropout_rate, conv=conv_params)

            if block_number == 0:
                shortcut = conv_block(dim, input_tensor, output_filters, 1, 'c', \
                                      strides=strides, padding='same', is_training=is_training, \
                                      conv=conv_params)
            else:
                shortcut = input_tensor

            x = tf.add(x, shortcut)
            x = tf.nn.relu(x, name='output')
        return x


class ResNet152(ResNetModel):
    ''' An original ResNet-101 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 8, 36, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()

class ResNet101(ResNetModel):
    ''' An original ResNet-101 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 23, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()

class ResNet50(ResNetModel):
    ''' An original ResNet-50 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()

class ResNet34(ResNetModel):
    ''' An original ResNet-34 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()


class ResNet18(ResNetModel):
    ''' An original ResNet-18 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [2, 2, 2, 2]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()
