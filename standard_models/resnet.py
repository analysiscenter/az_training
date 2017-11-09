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
        d : int {1, 2, 3}
            number of dimensions

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
        if not isinstance(dim, int) or dim < 1 or dim > 3:
            raise ValueError("Number of dimensions should be 1, 2 or 3, but given %d" % dim)

        layout = self.get_from_config('layout', 'cna')
        head_type = self.get_from_config('head_type', 'dense')
        conv_params = self.get_from_config('conv_params', {'conv': {}})
        activation = self.get_from_config('activation', tf.nn.relu)
        dropout_rate = self.get_from_config('dropout_rate', 0.)
        input_block_config = self.get_from_config('input_block_config', 
                                                 {'layout': 'cnap', 'filters': 64, 'kernel_size': 7, 
                                                 'strides': 2, 'pool_size': 3, 'pool_strides': 2})
        filters = self.get_from_config('filters', [64, 128, 256, 512])
        length_factor = self.get_from_config('length_factor', [1, 1, 1, 1])
        if isinstance(length_factor, int):
            length_factor = length_factor * len(filtlers)
        elif len(length_factor) != len(filters):
            raise ValueError("length_factor should be int or list of the same length as list\
                            of filters, but given length is %d" % len(length_factor))

        strides = self.get_from_config('strides', [2, 2, 2, 2])
        if isinstance(strides, int):
            strides = strides * len(filtlers)
        elif len(strides) != len(filters):
            raise ValueError("strides should be int or list of the same length as list\
                            of filters, but given length is %d" % len(strides))

        bottleneck = self.get_from_config('bottleneck', [False, False, False, False])
        if isinstance(bottleneck, bool):
            bottleneck = bottleneck * len(filtlers)
        elif len(bottleneck) != len(filters):
            raise ValueError("bottleneck should be bool or list of the same length as list\
                            of filters, but given length is %d" % len(bottleneck))

        bottelneck_factor = self.get_from_config('bottelneck_factor', [4, 4, 4, 4])
        if isinstance(bottelneck_factor, int):
            bottelneck_factor = bottelneck_factor * len(filtlers)
        elif len(bottelneck_factor) != len(filters):
            raise ValueError("bottelneck_factor should be int or list of the same length as list\
                            of filters, but given length is %d" % len(bottelneck_factor))

        se_block = self.get_from_config('se_block', [0, 0, 0, 0])
        if isinstance(se_block, int):
            se_block = se_block * len(filtlers)
        elif len(se_block) != len(filters):
            raise ValueError("se_block should be int or list of the same length as list\
                            of filters, but given length is %d" % len(se_block))

        is_training = self.is_training
        kwargs = {'conv': conv_params['conv'], 'is_training': is_training, 'data_format': data_format, 
                      'dropout_rate': dropout_rate, 'activation': activation}

        with tf.variable_scope('resnet'):
            net = ResNet.body(dim, inputs['images'], filters, length_factor, strides, layout, 
                              se_block, bottleneck, bottelneck_factor, input_block_config, **kwargs)
            net = ResNet.head(dim, net, n_classes, head_type, data_format, is_training)

        predictions = tf.identity(net, name='predictions')
        probs = tf.nn.softmax(net, name='predicted_prob')
        labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(inputs['labels'], axis=1), tf.float32, 'true_labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), \
                                  tf.float32), name='accuracy')


    @staticmethod
    def body(dim, inputs, filters, length_factor, strides, layout, se_block, bottleneck, bottelneck_factor, input_block_config, **kwargs):
        
        with tf.variable_scope('body'):
            net = ResNet.input_block(dim, inputs, input_block_config=input_block_config)
            for index, block_length in enumerate(length_factor):
                print('index ', index)
                for block_number in range(block_length):
                    print('block_number ', block_number)
                    net = ResNet.block(dim, net, filters[index], layout, 'block-'+str(index), block_number, \
                                       strides[index], bottleneck[index], bottelneck_factor[index], \
                                       se_block[index], **kwargs)
            net = tf.identity(net, name='conv_output')
        return net
    

    @staticmethod
    def head(dim, inputs, n_outputs, head_type='dense', data_format='channels_last', is_training=True):
        """ Head for classification
        """
        with tf.variable_scope('head'):
            if head_type == 'dense':
                net = global_average_pooling(dim=dim, inputs=inputs, data_format=data_format)
                net = tf.layers.dense(net, n_outputs)

            elif head_type == 'conv':
                net = conv_block(dim=dim, input_tensor=inputs, filters=n_outputs, kernel_size=1,\
                                     layout='c', name='con v_1', data_format=data_format)
                net = global_average_pooling(dim=dim, inputs=net, data_format=data_format)
            else:
                raise ValueError("Head_type should be dense or conv, but given %d" % head_type)
        return net


    def input_block(dim, inputs, input_block_config, name='block-'+'input'):
        with tf.variable_scope(name):
            net = conv_block(dim, inputs, **input_block_config)
        return net


    @staticmethod
    def block(dim, inputs, filters, layout, name, block_number, strides, 
                   bottleneck=False, bottelneck_factor=4, se_block=0, 
                   **kwargs):
        ''' 
        Residual block 
        Parameters:
        ----------

        dim: int
            spatial dimension of input without the number of channels

        inputs : tf.Tensor

        filters: int
            number of filters in the ouput tensor

        layout: str
            a sequence of layers:
                c - convolution
                n - batch normalization
                a - activation
                Default is 'cna'

        name: sstr
            name of the block that will be used as a scope.

        block_number: int
            index number of the block in the sequence of residual 
            blocks of the same number of filters. 
        
        strides: int
            if block_number != 0 strides is equal to 1 for in residual blocks 
            downsampling is performed only paired with an enlargement 
            of number of filters' size.

        bottleneck: bool. Default is False.
            whether to do bottleneck. In original ResNet models bottleneck is used
            for networks with more than 50 layers.

        bottelneck_factor: int. Default is 4
            enlargement factor for the last layer's number of filters in 
            the bottleneck block. Recall that 1x1 convolutions are responsible 
            for reducing and then restoring filters' dimension.

        se_block: int.
            squezing factor for the Squeeze and excitation (se) block.
            Se block will be applyed if se_block > 0.
        
        Returns
        -------
        
        tf. tensor
        '''

        strides = 1 if block_number != 0 else strides

        name = name + '-' + str(block_number)

        with tf.variable_scope(name):
            if bottleneck:
                output_filters = filters * bottelneck_factor
                x = ResNet.bottleneck_conv(dim, inputs, filters, output_filters, layout, strides, 
                                           **kwargs)
            else:
                output_filters = filters
                x = ResNet.original_conv(dim, inputs, filters, layout, strides, **kwargs)

            if se_block > 0:
                x = ResNet.se_block(dim, x, se_block, **kwargs)

            shortcut = inputs
            if block_number == 0:
                shortcut = conv_block(dim, inputs, output_filters, 1, 'c', 'shortcut', strides=strides, 
                                      **kwargs)

            x = tf.add(x, shortcut)
            x = tf.nn.relu(x, name='output')
        return x


    @staticmethod
    def bottleneck_conv(dim, inputs, filters, output_filters, layout, strides, **kwargs):
        ''' A stack of 3 convolutions for residual block with bottleneck'''
        x = conv_block(dim, inputs, [filters, filters, output_filters], [1, 3, 1], \
                       layout*3, name='bottleneck_conv', strides=[strides, 1, 1], **kwargs)
        return x


    @staticmethod
    def original_conv(dim, inputs, filters, layout, strides, **kwargs):
        ''' A stack of 2 convolutions for residual block '''
        x = conv_block(dim, inputs, filters=[filters, filters], kernel_size=[3, 3], layout=layout+'d'+layout, \
                               strides=[strides, 1], **kwargs)
        return x


    @staticmethod
    def se_block(dim, inputs, se_block, **kwargs):
            """ Squeeze and excitation block """
            data_format = kwargs['data_format']
            full = global_average_pooling(dim=dim, inputs=inputs, data_format=data_format)
            if data_format == 'channels_last':
                original_filters = inputs.get_shape().as_list()[-1]
                shape = [-1] + [1] * dim + [original_filters]
            else:
                original_filters = inputs.get_shape().as_list()[1]
                shape = [original_filters] + [-1] + [1] * dim
                
            full = tf.reshape(full, shape)
            full = tf.layers.dense(full, int(original_filters/se_block), activation=tf.nn.relu, \
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='first_dense_se_block')
            full = tf.layers.dense(full, original_filters, activation=tf.nn.sigmoid, \
                kernel_initializer=tf.contrib.layers.xavier_initializer(), name='second_dense_se_block')
            return inputs * full


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
        super()._build()


class ResNet50(ResNet):
    ''' An original ResNet-50 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        super()._build()


class ResNet34(ResNet):
    ''' An original ResNet-34 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [3, 4, 6, 3]
        super()._build()


class ResNet18(ResNet):
    ''' An original ResNet-18 architecture for ImageNet
    '''
    def _build(self, *args, **kwargs):
        self.config['length_factor'] = [2, 2, 2, 2]
        super()._build()
