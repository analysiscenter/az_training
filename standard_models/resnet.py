""" ResNetModel class
"""
import sys

import tensorflow as tf

sys.path.append("..")

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf.layers.pooling import global_average_pooling


class ResNet(TFModel):
    ''' Universal Resnet model constructor
    '''
    '''
    Configuration:
        layout: str - a sequence of layers
            c - convolution
            n - batch normalization
            a - activation
            p - max pooling
            Default is 'cna'.
        filters: list. Default is [64, 128, 256, 512].
            Defines number of filters in the Reidual blocks output.
        length_factor: list of length 4 with int elements
            Specifies how many Reidual blocks
            will be of the same feature maps dimension.
            Recall that ResNet can have [64, 128, 256, 512] output feature maps so
            there are 4 types of sequences of ResNet blocks with the same n_filters parameter.
            So the length_factor = [1, 2, 3, 4] will make one block with 16 feature maps,
            2 blocks with 32,
            3 blocks with 64, 4 with 128.
        bottleneck: bool
            If True all residual blocks will have 1x1, 3x3, 1x1 convolutional layers.
        bottelneck_factor: int
            A multiplicative factor for restored dimension in bottleneck block.
            Default is 4, e.g., for block with 64 input filters, there will be 256 filters 
            in the output tensor.
        max_pool: bool
            Whether to do maxpulling with stride 2 after first convolutional layer.
        conv : dict - parameters for convolution layers, like initializers, regularalizers, etc
        strides: list of length 4 with int elements
            Default is [2, 1, 1, 1]
        dropout_rate: float in [0, 1]. Default is 0.
        '''
    def _build(self, *args, **kwargs):
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_classes('labels')
        data_format = self.data_format('images')

        dim = self.get_from_config('dim', 2)

        filters = self.get_from_config('filters', [64, 128, 256, 512])
        length_factor = self.get_from_config('length_factor', [1, 1, 1, 1]) # or int
        strides = self.get_from_config('strides', [2, 2, 2, 2])
        # check equal lengts else fail 

        layout = self.get_from_config('layout', 'cna')

        bottleneck = self.get_from_config('bottleneck', False) # make a list or one int 
        bottelneck_factor = self.get_from_config('bottelneck_factor', 4) # make a list
        
        # убрать skip = self.get_from_config('skip', True)
        
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

        max_pool = self.get_from_config('max_pool', False)
        conv_params = self.get_from_config('conv_params', {'conv': {}})
        activation = self.get_from_config('activation', tf.nn.relu)

        dropout_rate = self.get_from_config('dropout_rate', 0.)

        is_training = self.is_training

        with tf.variable_scope('resnet'):
            if max_pool:
                first_layout = layout + 'p'
            else:
                first_layout = layout

            net = conv_block(dim, inputs['images'], filters[0], 7, first_layout, name='0', strides=2, \
                             is_training=is_training, pool_size=3, pool_strides=2)

            kwargs = {'conv': conv_params['conv'], 'is_training': is_training, 'data_format': data_format, 
                      'dropout_rate': dropout_rate, 'activation': activation}

            for index, block_length in enumerate(length_factor):
                for block_number in range(block_length):
                    net = self.block(dim, net, filters[index], layout, str(index), block_number, \
                                          strides[index], bottleneck, bottelneck_factor, skip, \
                                          se_block, **kwargs)
                    
            net = tf.identity(net, name='conv_output')

            net = global_average_pooling(dim, net, data_format)
            net = tf.contrib.layers.flatten(net)


            net = tf.layers.dense(net, n_classes)
            predictions = tf.identity(net, name='predictions')

            probs = tf.nn.softmax(net, name='predicted_prob')
            labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
            labels = tf.cast(tf.argmax(inputs['labels'], axis=1), tf.float32, 'true_labels')

            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), \
                                      tf.float32), name='accuracy')


    @staticmethod
    def block(dim, input_tensor, filters, layout, name, block_number, strides, 
                   bottleneck=False, bottelneck_factor=4, skip=False, se_block=False, 
                   **kwargs):

        if block_number != 0:
            strides = 1

        name = name + '/' + str(block_number)

        with tf.variable_scope(name):
            if bottleneck:
                output_filters = filters * bottelneck_factor
                x = ResNet.bottleneck_conv(dim, input_tensor, filters, layout, strides, bottelneck_factor
                                           **kwargs)
            else:
                output_filters = filters
                x = ResNet.original_conv(dim, input_tensor, filters, layout, strides, **kwargs)

            if se_block:
                x = ResNet.se_block(x, se_block)

            shortcut = input_tensor
            if block_number == 0:
                shortcut = conv_block(dim, input_tensor, output_filters, 1, 'c', strides=strides, 
                                      **kwargs)

            x = tf.add(x, shortcut)
            x = tf.nn.relu(x, name='output')
        return x


    @staticmethod
    def bottleneck_conv(dim, input_tensor, filters, layout, name, conv, strides, is_training,
                        data_format, bottelneck_factor, activation):
        # output_filters = filters * bottelneck_factor
        x = conv_block(dim, input_tensor, [filters, filters, output_filters], [1, 3, 1], \
                       layout*3, strides=[strides, 1, 1], data_format=data_format, activation=activation, \
                       is_training=is_training, conv=conv)
        return x


    @staticmethod
    def original_conv(dim, input_tensor, filters, layout, strides, **kwargs):
        x = conv_block(dim, input_tensor, filters=[filters, filters], kernel_size=[3, 3], layout=layout+'d'+layout, \
                               strides=[strides, 1], **kwargs)
        return x


    @staticmethod
    def se_block(self,dim, x, data_format, se_block):
            """ create se block """
            r = se_block['r']
            C = se_block['C']
            full = global_average_pooling(dim=dim, inputs=x, data_format=data_format)
            if data_format == 'channels_last':
                shape = [-1] + [1] * dim + [C]
            else:
                shape = [C] + [-1] + [1] * dim
            full = tf.reshape(full, shape)
            full = tf.layers.dense(full, int(C/r), activation=tf.nn.relu, \
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='first_dense_se_block')
            full = tf.layers.dense(full, C, activation=tf.nn.sigmoid, \
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='second_dense_se_block')
            return x * full

    
    @staticmethod
    def add_shortcut(dim, input_tensor, output_filters, strides, skip,
                     **kwargs):
            """ create shortcut connetion """
            if kwargs['data_format'] == 'channels_last':
                input_filters = input_tensor.get_shape()[-1]
            else:
                input_filters = input_tensor.get_shape()[0]
            
            if input_filters != output_filters:
                return conv_block(dim, input_tensor, output_filters, 1, 'c', strides=strides, 
                                  **kwargs)
            else:
                return input_tensor

    def input_block
        None - ничего 
        kernel , filter, stride
        pool_size, pool_stride

class ResNet152(ResNet):
    ''' An original ResNet-101 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 8, 36, 3]
        super()._build()

class ResNet101(ResNet):
    ''' An original ResNet-101 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 23, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()

class ResNet50(ResNet):
    ''' An original ResNet-50 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()

class ResNet34(ResNet):
    ''' An original ResNet-34 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()


class ResNet18(ResNet):
    ''' An original ResNet-18 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [2, 2, 2, 2]
        self.config['layout'] = 'cna'
        self.config['max_pool'] = True
        super()._build()
