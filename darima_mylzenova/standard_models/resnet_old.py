""" ResNetModel class
"""
import sys

import tensorflow as tf

sys.path.append("..")

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block


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
                Recall that ResNet can have [16, 32, 64, 128] output feature maps thefore there are 4 types of
                sequences of ResNet blocks whith the same n_filters parameter.
                So the length_factor = [1, 2, 3, 4] will make one block with 16 feature maps, 2 blocks with 32,
                3 blocks with 64, 4 with 128.
            widening_factor: int. Default is 4.
                Myltiplies default [1, 2, 3, 4] feature maps sizes.
                For example, widening_factor = 4 will make [64, 128, 256, 512] feature maps sizes.
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

        input_config = self.get_from_config('input', None)
        if input_config == None:
            raise ValueError('you should specify configuration of input data')
        dim_shape = input_config.get('shape', None)
        data_format = input_config.get('data_format', 'channels_last')
        n_classes = input_config.get('n_classes', 2)

        length_factor = self.get_from_config('length_factor', [1]) 
        widening_factor = self.get_from_config('widenning_factor', 4)
        layout = self.get_from_config('layout', 'cna')
        bottleneck = self.get_from_config('bottleneck', False)
        max_pool = self.get_from_config('max_pool', False)
        conv_params = self.get_from_config('conv_params', {})
        downsampling_keys = self.get_from_config('downsampling_keys', [True, True, True, True])
        dropout_rate = self.get_from_config('dropout_rate', 0.)

        is_training = self.is_training

        if dim_shape == None:
            raise ValueError('dim_shape should be customized in config')

        x = tf.placeholder(tf.float32, name='input_images')

        if data_format == 'channels_first':
            dim_shape = dim_shape[1:] + dim_shape[0]
        dim = len(dim_shape) - 1
        
        x_reshaped = tf.reshape(x, shape=[-1] + dim_shape)


        n_filters = 16 * widening_factor
        if max_pool:
            first_layout = layout + 'p'
        else:
            first_layout = layout

        net = conv_block(dim, x_reshaped, n_filters, (7, 7), first_layout, '0', strides=2, is_training=is_training, pool_size=3, pool_strides=2)

        for block_number in range(length_factor[0]):
            if block_number == 0 and downsampling_keys[0]:
                    net = downsampling_block(dim, net, 3, [16, 16], layout,'1' + str(block_number), conv_params['conv'], strides=2, w_factor=widening_factor, is_training=is_training, \
                                     bottleneck=bottleneck, dropout_rate=dropout_rate)
            else:
                net  = identity_block(dim, net, 3, [16, 16], layout, '1_' + str(block_number), conv_params['conv'], w_factor=widening_factor, \
                                      is_training=is_training, bottleneck=bottleneck, dropout_rate=dropout_rate)

        for block_number in range(length_factor[1]):
            if block_number == 0 and downsampling_keys[0]:
                net = downsampling_block(dim, net, 3, [32, 32], layout, '2' + str(block_number), conv_params['conv'], strides=2, w_factor=widening_factor, is_training=is_training, \
                                        bottleneck=bottleneck, dropout_rate=dropout_rate)
            else:
                net  = identity_block(dim, net, 3, [32, 32], layout, '2_' + str(block_number), conv_params['conv'], w_factor=widening_factor, \
                                      is_training=is_training, bottleneck=bottleneck, dropout_rate=dropout_rate)

        for block_number in range(length_factor[2]):
            if block_number == 0 and downsampling_keys[0]:
                net = downsampling_block(dim, net, 3, [64, 64], layout, '3' + str(block_number), conv_params['conv'], strides=2, w_factor=widening_factor, is_training=is_training, \
                                         bottleneck=bottleneck, dropout_rate=dropout_rate)
            else:
                net  = identity_block(dim, net, 3, [64, 64], layout, '3_' + str(block_number), conv_params['conv'], w_factor=widening_factor, \
                                      is_training=is_training, bottleneck=bottleneck, dropout_rate=dropout_rate)


        for block_number in range(length_factor[2]):
            if block_number == 0 and downsampling_keys[0]:
                net = downsampling_block(dim, net, 3, [128, 128], layout, '4' + str(block_number), conv_params['conv'], strides=2, w_factor=widening_factor, is_training=is_training, \
                                         bottleneck=bottleneck, dropout_rate=dropout_rate)
            else:
                net = identity_block(dim, net, 3, [128, 128], layout, '4_' + str(block_number), conv_params['conv'], w_factor=widening_factor, \
                                      is_training=is_training, bottleneck=bottleneck, dropout_rate=dropout_rate)

        net = tf.identity(net, name = 'conv_output')

        net = tf.layers.average_pooling2d(net, (7, 7), strides=(1, 1))
        net = tf.contrib.layers.flatten(net)


        net = tf.layers.dense(net, n_classes)ф
        predictions = tf.identity(net, name='predictions')

        probs = tf.nn.softmax(net, name='predicted_prob')
        y_ = tf.placeholder(tf.float32, [None, n_classes], name='targets')

        labels_hat = tf.cast(tf.argmax(predictions, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, 'true_labels')
        # true_labels = tf.identity(labels, name='true_labels')

        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), \
                                  tf.float32), name='accuracy')




def downsampling_block(dim, input_tensor, kernel_size, filters, layout, name, conv_params,
                       strides, w_factor, is_training=True, bottleneck=False, dropout_rate=0.):
    """ This function creates downsampling Residual block with skipconnection and conv layer at shortcut
    If bottleneck is True it has 3 convolutional layers
    Otherwise it has 2 3x3 convolutions
    Parameters
    ----------
         dim : int {1, 2, 3} - number of one feature map's dimensions
        input_tensor: input tensorflow layer.
        kernel_size: int or tuple(int, int) of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of the block that will be used as a scope.
        initializer: weights initializer.
        strides: typle of strides in convolution layer.
        w_factor: widenning factor which increases width of the network by multiplying number of filters.
        is_training: bool or tf.Tensor.
        dropout_rate: in [0, 1]. E.g 0.1 would drop out 10% of input units.

    Returns
    -------

    output tensor: tf.Tensor
    """

    filters = [int(filt * w_factor) for filt in filters]

    if bottleneck:
        BOTTLENECK_FACTOR = 4

        filters1, filters2 = filters
        filters3 = filters1 * BOTTLENECK_FACTOR
        
        x = conv_block(dim, input_tensor, filters1, (1, 1), layout, name=name + '_b1_11', strides=strides, padding='same', \
                       activation=tf.nn.relu, is_training=is_training, conv=conv_params)

        x = conv_block(dim, x, filters2, kernel_size, layout, name=name + '_b2_33', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)


        x = conv_block(dim, x, filters3, (1, 1), layout, name=name + '_b2_11', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)

   
    else:
        filters1, filters2 = filters
        filters3 = filters2

        x = conv_block(dim, input_tensor, filters1, kernel_size, layout + 'd', name=name + '_b1_33', strides=strides, padding='same', \
                       activation=tf.nn.relu, is_training=is_training, dropout_rate=dropout_rate)

        x = conv_block(dim, x, filters2, kernel_size, layout, name=name + '_b2_33', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)


    shortcut = conv_block(dim, input_tensor, filters3, (1, 1), 'c', name=name + '_shortuct', strides=strides, padding='same', 
                                       is_training=is_training, conv=conv_params)

    x = tf.add(x, shortcut)
    x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate, training)
    return x

def identity_block(dim, input_tensor, kernel_size, filters, layout, name, conv_params, w_factor, is_training, 
                    bottleneck=False, dropout_rate=0.):
    """ This function creates Residual block with skipconnection and has no conv layer at shortcut
    If bottleneck is True it has 3 convolutional layers
    Otherwise it has 2 3x3 convolutions
    Parameters
    ----------
         dim : int {1, 2, 3} - number of one feature map's dimensions
        input_tensor: input tensorflow layer.
        kernel_size: int or tuple(int, int) of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of the block that will be used as a scope.
        initializer: weights initializer.
        strides: typle of strides in convolution layer.
        w_factor: widenning factor which increases width of the network by multiplying number of filters.
        is_training: bool or tf.Tensor.
        dropout_rate: in [0, 1]. E.g 0.1 would drop out 10% of input units.

    Returns
    -------

    output tensor: tf.Tensor
    """

    filters = [int(filt * w_factor) for filt in filters]
    # print(filters1, filters2)
    if bottleneck:
        BOTTLENECK_FACTOR = 4

        filters1, filters2 = filters
        filters3 = filters1 * BOTTLENECK_FACTOR

        x = conv_block(dim, input_tensor, filters1, (1, 1), layout, name=name + '_b1_11', padding='same', \
                       activation=tf.nn.relu, is_training=is_training, conv=conv_params)

        x = conv_block(dim, x, filters2, kernel_size, layout, name=name + '_b2_33', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)

        x = conv_block(dim, x, filters3, (1, 1), layout, name=name + '_b2_11', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)
   
    else:
        filters1, filters2 = filters

        x = conv_block(dim, input_tensor, filters1, kernel_size, layout + 'd', name=name + '_b1_33', padding='same', \
                       activation=tf.nn.relu, is_training=is_training, dropout_rate=dropout_rate, conv=conv_params)

        x = conv_block(dim, x, filters2, kernel_size, layout, name=name + '_b2_33', padding='same', \
                        activation=tf.nn.relu, is_training=is_training, conv=conv_params)


    x = tf.add(x, input_tensor)
    x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate, training)
    return x


class ResNet152(ResNetModel):
    def _build(self, *args, **kwargs):
        ''' An original ResNet-101 architecture for ImageNet
        '''
        self.config['length_factor'] = [3, 8, 36, 3]
        self.config['widenning_factor'] = 4
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet101(ResNetModel):
    def _build(self, *args, **kwargs):
        ''' An original ResNet-101 architecture for ImageNet
        '''
        self.config['length_factor'] = [3, 4, 23, 3]
        self.config['widenning_factor'] = 4
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet50(ResNetModel):
    def _build(self, *args, **kwargs):
        ''' An original ResNet-50 architecture for ImageNet
        '''
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['widenning_factor'] = 4
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()

class ResNet34(ResNetModel):
    def _build(self, *args, **kwargs):
        ''' An original ResNet-34 architecture for ImageNet
        '''
        self.config['length_factor'] = [3, 4, 6, 3]
        self.config['widenning_factor'] = 4
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()


 
class ResNet18(ResNetModel):
    def _build(self, *args, **kwargs):
        ''' An original ResNet-18 architecture for ImageNet
        '''
        self.config['length_factor'] = [2, 2, 2, 2]
        self.config['widenning_factor'] = 4
        self.config['dim_shape'] = [None, 224, 224, 3]
        self.config['layout'] = 'cna'
        self.config['n_classes'] = 1000
        self.config['max_pool'] = True
        super()._build()
