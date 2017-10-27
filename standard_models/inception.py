import sys

import tensorflow as tf
sys.path.append('..')

from pooling import max_pooling, average_pooling
from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv_block

class GoogLeNet(TFModel):
    def _build(self, *args, **kwargs):
        _ = args, kwargs

        data_format = self.get_from_config('data_format')

        placeholders = self._make_inputs()
        input_image = placeholders['images']
        targets = placeholders['targets']

        with tf.variable_scope('googlenet'):
            net = conv_block(dim=dim, input_tensor=input_image, filters=64, kernel_size=7, strides=2, layout='cm',\
                             data_format=data_format, pool_size=3, pool_strides=2)
            net = conv_block(dim=dim, input_tensor=net, filters=64, kernel_size=3, layout='c',\
                             data_format=data_format)
            net = conv_block(dim=dim, input_tensor=net, filters=192, kernel_size=3, layout='cm',\
                             data_format=data_format, pool_size=3, pool_strides=2)
            net = googlenet_block(dim=dim, input_tensor=net, filters=[64, 96, 128, 16, 32, 32], data_format=data_format, name='3a')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[128, 128, 192, 32, 96, 64], data_format=data_format, name='3b')
            net = max_pooling(dim=dim, input=net, pool_size=3, pool_strides=2, padding='same', data_format=data_format)
            
            net = googlenet_block(dim=dim, input_tensor=net, filters=[192, 96, 208, 16, 48, 64], data_format=data_format, name='4a')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[160, 112, 224, 24, 64, 64], data_format=data_format, name='4b')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[128, 128, 256, 24, 64, 64], data_format=data_format, name='4c')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[112, 144, 288, 32, 64, 64], data_format=data_format, name='4d')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[256, 160, 320, 32, 128, 128], data_format=data_format, name='4d')
            net = max_pooling(dim=dim, input=net, pool_size=3, pool_strides=2, padding='same', data_format=data_format)

            net = googlenet_block(dim=dim, input_tensor=net, filters=[256, 160, 320, 32, 128, 128], data_format=data_format, name='5a')
            net = googlenet_block(dim=dim, input_tensor=net, filters=[384, 192, 384, 48, 128, 128], data_format=data_format, name='5b')
            pool_size = net.shape[1:3]
            net = average_pooling(dim=dim, input=net, pool_size=pool_size, pool_strides=1, padding='same', data_format=data_format)
            net = tf.layers.dropout(net, 0.4, training=self.is_training)
            net = tf.layers.dense(net, 10)

            tf.identity(net, name='predictions')
        return statistic(net)

    def statistic(self, net):
        prob = tf.nn.softmax(net, name='prob_predictions')

        labels_hat = tf.cast(tf.argmax(prob, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(targets, axis=1), tf.float32, name='labels')
        tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

    def googlenet_block(self, dim, input_tensor, filters, data_format, name, batch_norm=False):
        
        if batch_norm:
            layout = 'cn'
        else:
            layout = 'c'

        with tf.variable_scope("block_" + name):
            block_1 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[0], kernel_size=1, layout=layout, name='conv_1',\
                                 data_format=data_format)

            block_1_3 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[1], kernel_size=1, layout=layout, name='conv_1_3',\
                                 data_format=data_format)
            block_3 = conv_block(dim=dim, input_tensor=block_1_3, filters=filters[2], kernel_size=1, layout=layout, name='conv_3',\
                                 data_format=data_format)

            block_1_5 = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[3], kernel_size=1, layout=layout, name='conv_1_5',\
                                 data_format=data_format)
            block_5 = conv_block(dim=dim, input_tensor=block_1_5, filters=filters[4], kernel_size=1, layout=layout, name='conv_5',\
                                 data_format=data_format)

            conv_pool = conv_block(dim=dim, input_tensor=input_tensor, filters=filters[5], kernel_size=1, layout='p'+layout, name='conv_pool',\
                                 data_format=data_format, pool_size=1, pool_strides=1)

            if data_format == 'channels_last':
                return tf.concat([block_1, block_3, block_5, conv_pool], -1, name='output')
            elif data_format == 'channels_first':
                return tf.concat([block_1, block_3, block_5, conv_pool], 0, name='output')
            else:
                raise ValueError("data_format can be 'last' or 'first' not %d"% data_format)
