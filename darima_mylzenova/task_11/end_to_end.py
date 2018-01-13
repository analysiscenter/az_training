""" Custom class for detection CNN
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..//..")
from dataset.models.tf.resnet import ResNet
# , MobileNet
from dataset import Batch, action, model, inbatch_parallel
from dataset import ImagesBatch
from dataset.dataset.models.tf import TFModel

from dataset.dataset.models.tf.layers import conv_block




class DetectionResNet(ResNet):
    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = 4 + 10
        return config

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        filters = 16
        config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))

           # number of filters in the first block
        config['body']['num_blocks'] = [2, 2, 2]
        config['body']['filters'] = 2 ** np.arange(len(config['body']['num_blocks'])) * filters
        config['body']['block']['bottleneck'] = True
        return config

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ Last network layers which produce output from the network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

            MyModel.head(2, network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits::

            MyModel.head(2, network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """

        kwargs = cls.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        true_coordinates = tf.get_default_graph().get_tensor_by_name("DetectionResNet/inputs/coordinates:0")
        print('true_coordinates ', true_coordinates)

        
        x_true = true_coordinates[:, 0:1]
        y_true = true_coordinates[:, 1:2]
        x_1_true = true_coordinates[:, 2:3]
        y_1_true = true_coordinates[:, 3:4]
        width_true = x_1_true - x_true
        height_true = y_1_true - y_true
        norm_x = tf.round(x_true / (width_true - 1))
        norm_y = tf.round(y_true / (height_true - 1))
        norm_x_1 = tf.round(x_1_true / (width_true - 1))
        norm_y_1 = tf.round(y_1_true / (height_true - 1))
        print (norm_x.get_shape().as_list(), 'dsd', x_true.get_shape().as_list())
        true_normalized = tf.concat((norm_x, norm_y, norm_x_1, norm_y_1), axis=1)
        mse_loss =  tf.reduce_mean(tf.square(true_normalized - outputs[:, 10:]), name='mse')
        tf.losses.add_loss(mse_loss)
        tf.identity(outputs[:, 10:], name='predicted_bb')
        tf.identity(mse_loss, name='mse_loss')


        # print(outputs[:, 0:5], 'outputs[:, :]')
        # # x, y, x_1, y_1 = outputs[:, 0:5]
        # x = outputs[:, 0]
        # y = outputs[:, 1]
        # x_1 = outputs[:, 2]
        # y_1 = outputs[:, 3]

        # target_width = x_1 - x
        # target_height = y_1 - y
        # # offset_height
        # print(' inputs.get_shape().as_list()[1:]',  inputs.get_shape().as_list()[1:3])
        # image_width, image_height = inputs.get_shape().as_list()[1:3]
        # # image_width = inputs.shape[1]
        # # image_height = inputs.shape[2]
        # norm_y = tf.round(y / (image_height - 1))
        # norm_x = tf.round(x / (image_width - 1))
        # norm_x1 = tf.round(x_1 / (image_width - 1))
        # norm_y = tf.round(y_1 / (image_height - 1))




        true_classes = tf.get_default_graph().get_tensor_by_name('DetectionResNet/inputs/targets:0')
        ce_loss = tf.losses.softmax_cross_entropy(true_classes, outputs[:, :10])

        tf.losses.add_loss(ce_loss)

        tf.identity(ce_loss, name='ce_loss')

        predicted_labels = tf.cast(tf.argmax(outputs[:, :10], axis=1), tf.int64, name='labels_hat')
        accy = tf.contrib.metrics.accuracy(tf.argmax(true_classes, axis=1), predicted_labels, name='accuracy_0')
        tf.identity(accy, name='accuracy')

        # tf.reduce_mean(tf.square(true_coordinates[:, :] - outputs[:, :]), name='mse')

        # print(tf.get_default_graph().get_operations())

        # predicted_probs = outputs
        return outputs


























class DetectionResNet2(ResNet):
    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = 4
        return config

    @classmethod
    def default_config(cls):
        config = ResNet.default_config()
        filters = 16
        config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
                                          pool_size=3, pool_strides=2))

           # number of filters in the first block
        config['body']['num_blocks'] = [2, 2, 2]
        config['body']['filters'] = 2 ** np.arange(len(config['body']['num_blocks'])) * filters
        config['body']['block']['bottleneck'] = True
        return config

    # @classmethod
    def head(self, inputs, name='head', **kwargs):
        """ Last network layers which produce output from the network embedding

        Parameters
        ----------
        inputs : tf.Tensor
            input tensor
        name : str
            scope name

        Notes
        -----
        For other parameters see :func:`.conv_block`.

        Returns
        -------
        tf.Tensor

        Examples
        --------
        A fully convolutional head with 3x3 and 1x1 convolutions and global max pooling:

            MyModel.head(2, network_embedding, layout='cacaP', filters=[128, num_classes], kernel_size=[3, 1])

        A fully connected head with dropouts, a dense layer with 1000 units and final dense layer with class logits::

            MyModel.head(2, network_embedding, layout='dfadf', units=[1000, num_classes], dropout_rate=.15)
        """

        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        true_coordinates = tf.get_default_graph().get_tensor_by_name("DetectionResNet2/inputs/coordinates:0")
        print('true_coordinates ', true_coordinates)

        
        x_true = true_coordinates[:, 0:1]
        y_true = true_coordinates[:, 1:2]
        x_1_true = true_coordinates[:, 2:3]
        y_1_true = true_coordinates[:, 3:4]
        width_true = x_1_true - x_true
        height_true = y_1_true - y_true
        norm_x = tf.round(x_true / (width_true - 1))
        norm_y = tf.round(y_true / (height_true - 1))
        norm_x_1 = tf.round(x_1_true / (width_true - 1))
        norm_y_1 = tf.round(y_1_true / (height_true - 1))
        print (norm_x.get_shape().as_list(), 'dsd', x_true.get_shape().as_list())
        true_normalized = tf.concat((norm_x, norm_y, norm_x_1, norm_y_1), axis=1)


        mse_loss =  tf.reduce_mean(tf.square(true_normalized - outputs[:, :]), name='mse')
        tf.losses.add_loss(mse_loss)
        
        tf.identity(outputs[:, :], name='predicted_bb')
        tf.identity(mse_loss, name='mse_loss')

        # other_coordinates = tf.get_default_graph().get_tensor_by_name('DetectionResNet2/inputs/other_coordinates:0')
        # for i, coord in enumerate(other_coordinates):
        #         x_true = other_coordinates[:, i, 0:1]
        #         y_true = other_coordinates[:, i, 1:2]
        #         x_1_true = other_coordinates[:, i,  2:3]
        #         y_1_true = other_coordinates[:, i, 3:4]
        #         width_true = x_1_true - x_true
        #         height_true = y_1_true - y_true
        #         norm_x = tf.round(x_true / (width_true - 1))
        #         norm_y = tf.round(y_true / (height_true - 1))
        #         norm_x_1 = tf.round(x_1_true / (width_true - 1))
        #         norm_y_1 = tf.round(y_1_true / (height_true - 1))
        #         print (norm_x.get_shape().as_list(), 'dsd', x_true.get_shape().as_list())
        #         true_normalized = tf.concat((norm_x, norm_y, norm_x_1, norm_y_1), axis=1)
        #         current_mse_loss =  tf.reduce_mean(tf.square(true_normalized - outputs[:, :]), name='mse')
        #         if current_mse_loss < mse_loss:
        #             mse_looss 



        # print(outputs[:, 0:5], 'outputs[:, :]')
        # # x, y, x_1, y_1 = outputs[:, 0:5]
        # x = outputs[:, 0]
        # y = outputs[:, 1]
        # x_1 = outputs[:, 2]
        # y_1 = outputs[:, 3]

        # target_width = x_1 - x
        # target_height = y_1 - y
        # # offset_height
        # print(' inputs.get_shape().as_list()[1:]',  inputs.get_shape().as_list()[1:3])
        # image_width, image_height = inputs.get_shape().as_list()[1:3]
        # # image_width = inputs.shape[1]
        # # image_height = inputs.shape[2]
        # norm_y = tf.round(y / (image_height - 1))
        # norm_x = tf.round(x / (image_width - 1))
        # norm_x1 = tf.round(x_1 / (image_width - 1))
        # norm_y = tf.round(y_1 / (image_height - 1))


        # tf.image.crop_and_resize()

        # print(tf.get_default_graph().get_operations())
        print(tf.range(100), 'tf range')
        new_inputs = tf.image.crop_and_resize(self.inputs['images'], outputs[:, :], tf.range(100), \
                                 tf.constant([28, 28], dtype=tf.int32))

        x = self.block(new_inputs, filters=16, bottleneck=False, data_format='channels_last', name='new0')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new1')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new2')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new3')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new4')

        print('x shape ', x.get_shape().as_list())
        second_outputs = conv_block(inputs, name='new5', layout='Vf', units=10)


        true_classes = tf.get_default_graph().get_tensor_by_name('DetectionResNet2/inputs/targets:0')


        ce_loss = tf.losses.softmax_cross_entropy(true_classes, second_outputs[:, :10])
        tf.losses.add_loss(ce_loss)
        tf.identity(ce_loss, name='ce_loss')

        predicted_labels = tf.cast(tf.argmax(second_outputs[:, :10], axis=1), tf.int64)
        tf.identity(predicted_labels, name='labels_hat')
        
        accy = tf.contrib.metrics.accuracy(predicted_labels, tf.argmax(true_classes, axis=1), name='accuracy_0')
        accy = tf.expand_dims(accy, -1)
        # accy = tf.expand_dims(accy, -1)

        print('accy ', accy.get_shape().as_list())
        other_labels = tf.get_default_graph().get_tensor_by_name('DetectionResNet2/inputs/other_labels:0')
        other_labels_len = other_labels.get_shape().as_list()[1]

        max_accy_index = 0
        for i in range(other_labels_len):
            print('other_labels[:, i] ', other_labels[:, i])
            current_accy = tf.contrib.metrics.accuracy(predicted_labels, tf.cast(other_labels[:, i], tf.int64), name='accuracy_'+ str(i))
            current_accy = tf.expand_dims(current_accy, -1)
            # current_accy = tf.expand_dims(current_accy, -1)

            print('current_accy ', current_accy.get_shape().as_list())

            # if (current_accy > accy) is not None:
            #     max_accy_index = i
            accy = tf.concat([accy, current_accy], axis=0)
        # max_accy_name = 'DetectionResNet2/inputs/accuracy_' + str(max_accy_index) + ':0'
        max_accy =  tf.reduce_max(accy, axis=-1)
        # tf.identity(tf.get_default_graph().get_tensor_by_name(max_accy_name), name='accuracy')
        tf.identity(max_accy, name='accuracy')
        # tf.reduce_mean(tf.square(true_coordinates[:, :] - outputs[:, :]), name='mse')


        # predicted_probs = outputs
        return outputs

    # def block(self, inputs, strides=1, double_filters=False, width_factor=1, name=None, **kwargs):
    #     """ A network building block consisting of a separable depthwise convolution and 1x1 pointwise covolution.

    #     Parameters
    #     ----------
    #     inputs : tf.Tensor
    #         input tensor
    #     strides : int
    #         strides in separable convolution
    #     double_filters : bool
    #         if True number of filters in 1x1 covolution will be doubled
    #     width_factor : float
    #         multiplier for the number of filters
    #     name : str
    #         scope name

    #     Returns
    #     -------
    #     tf.Tensor
    #     """
    #     data_format = kwargs.get('data_format')
    #     num_filters = cls.num_channels(inputs, data_format) * width_factor
    #     filters = [num_filters, num_filters*2] if double_filters else num_filters
    #     return conv_block(inputs, 'sna cna', filters, [3, 1], name=name, strides=[strides, 1], **kwargs)