""" Custom class for detection CNN
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append("..//..")
from dataset.models.tf.resnet import ResNet, ResNet18
# , MobileNet
from dataset import Batch, action, inbatch_parallel
from dataset import ImagesBatch
from dataset.dataset.models.tf import TFModel

from dataset.dataset.models.tf.layers import conv_block




class DetectionResNet(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        config['head']['units'] = 4 + 10
        return config

    # @classmethod
    # def default_config(cls):
    #     config = ResNet.default_config()
    #     filters = 16
    #     config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
    #                                       pool_size=3, pool_strides=2))

    #        # number of filters in the first block
    #     config['body']['num_blocks'] = [2, 2, 2]
    #     config['body']['filters'] = 2 ** np.arange(len(config['body']['num_blocks'])) * filters
    #     config['body']['block']['bottleneck'] = True
    #     return config

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

        true_coordinates = tf.get_default_graph().get_tensor_by_name("DetectionResNet/inputs/coordinates:0")
        # print('true_coordinates ', true_coordinates)

        x_true = true_coordinates[:, 0:1]
        y_true = true_coordinates[:, 1:2]
        x_1_true = true_coordinates[:, 2:3]
        y_1_true = true_coordinates[:, 3:4]
        width_true = self.images.get_shape().as_list()[1]
        height_true = self.images.get_shape().as_list()[2]
        # width = 
        # width_true = x_1_true - x_true
        # height_true = y_1_true - y_true
        norm_x = (x_true / (width_true - 1))
        norm_y = (y_true / (height_true - 1))
        norm_x_1 = (x_1_true / (width_true - 1))
        norm_y_1 = (y_1_true / (height_true - 1))
        # print (norm_x.get_shape().as_list(), 'dsd', x_true.get_shape().as_list())
        true_normalized = tf.concat((norm_x, norm_y, norm_x_1, norm_y_1), axis=1)
        tf.identity(true_normalized, name='true_normalized')

        # mse_loss =  tf.reduce_mean(tf.square(true_normalized - outputs[:, 10:]), name='mse')
        huber_loss = tf.losses.huber_loss(true_normalized, outputs[:, 10:])
        tf.losses.add_loss(huber_loss)
        tf.identity(huber_loss, name='huber')

        tf.identity(outputs[:, 10:], name='predicted_bb')
        # tf.identity(mse_loss, name='mse_loss')

        true_classes = tf.get_default_graph().get_tensor_by_name('DetectionResNet/inputs/targets:0')
        ce_loss = tf.losses.softmax_cross_entropy(true_classes, outputs[:, :10])

        tf.losses.add_loss(ce_loss)

        tf.identity(ce_loss, name='ce_loss')

        # predicted_labels = tf.cast(tf.argmax(outputs[:, :10], axis=1), tf.int64, name='labels_hat')
        # accy = tf.contrib.metrics.accuracy(tf.argmax(true_classes, axis=1), predicted_labels, name='accuracy_0')
        # tf.identity(accy, name='accuracy')



        predicted_labels = tf.cast(tf.argmax(outputs[:, :10], axis=1), tf.int64)
        tf.identity(predicted_labels, name='labels_hat')
        
        accy = tf.contrib.metrics.accuracy(predicted_labels, tf.argmax(true_classes, axis=1), name='accuracy_0')
        accy = tf.expand_dims(accy, -1)
        # accy = tf.expand_dims(accy, -1)

        # print('accy ', accy.get_shape().as_list())
        other_labels = tf.get_default_graph().get_tensor_by_name('DetectionResNet/inputs/other_labels:0')
        other_labels_len = other_labels.get_shape().as_list()[1]

        max_accy_index = 0
        for i in range(other_labels_len):
            print('other_labels[:, i] ', other_labels[:, i])
            current_accy = tf.contrib.metrics.accuracy(predicted_labels, tf.cast(other_labels[:, i], tf.int64), name='accuracy_'+ str(i))
            current_accy = tf.expand_dims(current_accy, -1)
            # current_accy = tf.expand_dims(current_accy, -1)

            # print('current_accy ', current_accy.get_shape().as_list())

            # if (current_accy > accy) is not None:
            #     max_accy_index = i
            accy = tf.concat([accy, current_accy], axis=0)
        tf.identity(accy, name='all_acuracies')
        # max_accy_name = 'DetectionResNet2/inputs/accuracy_' + str(max_accy_index) + ':0'
        max_accy =  tf.reduce_max(accy, axis=-1)
        # tf.identity(tf.get_default_graph().get_tensor_by_name(max_accy_name), name='accuracy')
        tf.identity(max_accy, name='accuracy')


        # tf.reduce_mean(tf.square(true_coordinates[:, :] - outputs[:, :]), name='mse')

        # print(tf.get_default_graph().get_operations())

        # predicted_probs = outputs
        return outputs














class SimpleDetection(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        num_digits = config.pop('num_digits')
        config['head']['num_digits'] = num_digits
        config['head']['units'] = 4 * num_digits
        return config

    def head(self, inputs, num_digits, name='head', **kwargs):
        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        all_coordinates = tf.get_default_graph().get_tensor_by_name("SimpleDetection/inputs/coordinates:0")
        width = self.images.get_shape().as_list()[1]
        height = self.images.get_shape().as_list()[2]

        for index in range(num_digits):
            current_normalized = normalize_bbox(width, height, all_coordinates[index])
            current_predictions = outputs[:, 4 * index : 4 * (index + 1)]
            current_huber = tf.losses.huber_loss(current_normalized, current_predictions)
            tf.losses.add_loss(current_huber)

        tf.identity(current_huber, name='huber')


        tf.identity(outputs[:, :], name='predicted_bb')
        tf.zeros([1], tf.int32, name='accuracy')
        tf.zeros([1], tf.int32, name='ce_loss')
        tf.zeros([1], tf.int32, name='mse_loss_history')
        tf.zeros([1], tf.int32, name='labels_hat')
        tf.zeros([1], tf.int32, name='all_acuracies')
        # print(tf.get_default_graph().get_operations())
        return outputs


class NearestDetection(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        num_digits = config.pop('num_digits')
        config['head']['num_digits'] = num_digits
        config['head']['units'] = 4 * num_digits
        return config

    def head(self, inputs, num_digits, name='head', **kwargs):
        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        all_coordinates = tf.get_default_graph().get_tensor_by_name("NearestDetection/inputs/coordinates:0")
        print('all_coordinates', all_coordinates.get_shape().as_list())
        # all_coordinates = self.coordinates
        width = self.images.get_shape().as_list()[1]
        height = self.images.get_shape().as_list()[2]

        all_distances_list = []
        true_normalized = []
        for index in range(num_digits):
            current_normalized = normalize_bbox(width, height, all_coordinates[:, index, :])
            print('current_normalized ', current_normalized.get_shape().as_list())
            true_normalized.append(current_normalized)
            print('outputs ', outputs.get_shape().as_list())
            all_dists = []
            for j in range(num_digits):
                all_dists.append(tf.reduce_mean(tf.square(current_normalized - outputs[:, 4 * j : 4 * (j + 1)]), axis=1))
            all_distances_list.append(tf.stack(all_dists, axis=1))

        true_normalized = tf.stack(true_normalized, axis=1)
        tf.identity(true_normalized, name='true_normalized')
        all_distances = tf.stack(all_distances_list, axis=2)
        print('all_distances ', all_distances.get_shape().as_list())

        min_distances = tf.reduce_min(all_distances, axis=1)

        print('min_distances ', min_distances.get_shape().as_list())

        tf.identity(min_distances, name='min_distances')
        
        average_min_distances = tf.reduce_mean(min_distances)
        print('min_distances after', average_min_distances.get_shape().as_list())
        tf.losses.add_loss(average_min_distances)
        # current_huber = tf.losses.huber_loss(current_normalized, current_predictions)
        # tf.losses.add_loss(current_huber)

        tf.identity(min_distances, name='min_distances')


        tf.identity(outputs[:, :], name='predicted_bb')
        tf.zeros([1], tf.int32, name='accuracy')
        tf.zeros([1], tf.int32, name='ce_loss')
        tf.zeros([1], tf.int32, name='mse_loss_history')
        tf.zeros([1], tf.int32, name='labels_hat')
        tf.zeros([1], tf.int32, name='all_acuracies')
        # print(tf.get_default_graph().get_operations())
        return outputs




class CropDetection(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        num_digits = config.pop('num_digits')
        config['head']['num_digits'] = num_digits
        config['head']['units'] = 4 * num_digits
        return config

    def head(self, inputs, num_digits, loss_type='mse', name='head', **kwargs):
        kwargs = self.fill_params('head', **kwargs)
        outputs = conv_block(inputs, name=name, **kwargs)
        print('outputs.shape ', outputs.shape)

        all_coordinates = tf.get_default_graph().get_tensor_by_name("CropDetection/inputs/coordinates:0")
        print('all_coordinates', all_coordinates.get_shape().as_list())
        width = self.images.get_shape().as_list()[1]
        height = self.images.get_shape().as_list()[2]

        all_distances_list = []
        true_normalized = []
        for index in range(num_digits):
            current_normalized = normalize_bbox(width, height, all_coordinates[:, index, :])
            true_normalized.append(current_normalized)
            all_dists = []
            for j in range(num_digits):
                if loss_type == 'mse':
                    all_dists.append(tf.reduce_mean(tf.square(current_normalized - outputs[:, 4 * j : 4 * (j + 1)]), axis=1))
                else:
                    all_dists.append(tf.reduce_mean(tf.losses.huber_loss(current_normalized, outputs[:, 4 * j : 4 * (j + 1)], 
                                                                         reduction=None)), axis=1)
            all_distances_list.append(tf.stack(all_dists, axis=1))
        
        true_normalized = tf.stack(true_normalized, axis=1)
        tf.identity(true_normalized, name='true_normalized')
        
        all_distances = tf.stack(all_distances_list, axis=2)
        print('all_distances shape ', all_distances.get_shape().as_list())

        min_distances = tf.reduce_min(all_distances, axis=1)
        tf.identity(min_distances, name='min_distances')

        average_min_distances = tf.reduce_mean(min_distances)
        tf.losses.add_loss(average_min_distances)

        arg_mins = tf.argmin(all_distances, axis=1)
        print(arg_mins.get_shape().as_list(), 'ARGMINS')
        print(arg_mins[:, 0].get_shape().as_list(), 'ARGMINS 0 ')

        cropped_images = []
        all_labels = tf.get_default_graph().get_tensor_by_name("CropDetection/inputs/labels:0")
        labels_list = []
        for index in range(num_digits):
            min_index = arg_mins[:, index][1]
            x_predicted = outputs[:, 4 * (min_index + 1):4 * (min_index + 1) + 1]
            y_predicted = outputs[:, 4 * (min_index + 1) + 1:4 * (min_index + 1) + 2]
            x_1_predicted = outputs[:, 4 * (min_index + 1) + 2:4 * (min_index + 1) + 3]
            y_1_predicted = outputs[:, 4 * (min_index + 1) + 3:4 * (min_index + 2)]
            bb_to_crop = tf.concat((y_predicted, x_predicted, y_1_predicted, x_1_predicted), axis=1)
            print(bb_to_crop.get_shape().as_list())

            new_inputs = tf.image.crop_and_resize(self.inputs['images'], bb_to_crop,
                                                  tf.range(100), tf.constant([28, 28], dtype=tf.int32))
            cropped_images.append(new_inputs)
            labels_list.append(all_labels[:, index])
            print(new_inputs.shape)

        cropped_images = tf.stack(cropped_images, axis=0)
        labels_extended = tf.stack(labels_list, axis=0)

        x = self.block(new_inputs, filters=16, bottleneck=False, data_format='channels_last', name='new0')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new1')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new2')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new3')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new4')
        second_outputs = conv_block(inputs, name='new5', layout='Vf', units=10)
        

        ce_loss = tf.losses.softmax_cross_entropy(labels_extended, second_outputs)
        tf.losses.add_loss(ce_loss)
        tf.identity(ce_loss, name='ce_loss')


        predicted_labels = tf.cast(tf.argmax(second_outputs, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(labels_extended, axis=1), tf.float32, name='labels')

        accy = tf.reduce_mean(tf.cast(tf.equal(predicted_labels, labels), tf.float32))
        tf.identity(accy, name='accuracy')

        tf.identity(outputs[:, :], name='predicted_bb')
        tf.zeros([1], tf.int32, name='mse_loss_history')
        tf.zeros([1], tf.int32, name='all_acuracies')
        return second_outputs







class DetectionResNet2(ResNet18):
    def build_config(self, names=None):
        config = super().build_config(names)
        # config['head']['units'] = 4
        num_others = config.pop('num_others')
        config['head']['num_others'] = num_others
        config['head']['units'] = 4 * (num_others + 1)
        return config

    # @classmethod
    # def default_config(cls):
    #     config = ResNet.default_config()
    #     filters = 16
    #     config['input_block'].update(dict(layout='cnap', filters=filters, kernel_size=7, strides=2,
    #                                       pool_size=3, pool_strides=2))

    #        # number of filters in the first block
    #     config['body']['num_blocks'] = [2, 2, 2]
    #     config['body']['filters'] = 2 ** np.arange(len(config['body']['num_blocks'])) * filters
    #     config['body']['block']['bottleneck'] = True
    #     return config

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
        # print('true_coordinates ', true_coordinates)
        width = self.images.get_shape().as_list()[1]
        height = self.images.get_shape().as_list()[2]

        true_normalized = normalize_bbox(width, height, true_coordinates)
        tf.identity(true_normalized, name='true_normalized')

        huber_loss = tf.losses.huber_loss(true_normalized, outputs[:, :4])
        tf.losses.add_loss(huber_loss)
        tf.identity(huber_loss, name='huber')

        width = self.images.get_shape().as_list()[1]
        height = self.images.get_shape().as_list()[2]
        x_predicted = outputs[:, 0:1]
        y_predicted = outputs[:, 1:2]
        x_1_predicted = outputs[:, 2:3]
        y_1_predicted = outputs[:, 3:4]
        bb_to_crop = tf.concat((y_predicted, x_predicted, y_1_predicted, x_1_predicted), axis=1)
        new_inputs = tf.image.crop_and_resize(self.inputs['images'], bb_to_crop,
                                                  tf.range(100), tf.constant([28, 28], dtype=tf.int32))
    
        for i in range(num_others):
            other_normalized = normalize_bbox(width, height, self.other_coordinates[:, i])
            huber_loss_i = tf.losses.huber_loss(other_normalized, outputs[:, 4 * (i + 1):4 * (i + 2)])
            tf.losses.add_loss(huber_loss_i)
        
            x_predicted = outputs[:, 4 * (i + 1):4 * (i + 1) + 1]
            y_predicted = outputs[:, 4 * (i + 1) + 1:4 * (i + 1) + 2]
            x_1_predicted = outputs[:, 4 * (i + 1) + 2:4 * (i + 1) + 3]
            y_1_predicted = outputs[:, 4 * (i + 1) + 3:4 * (i + 2)]
            bb_to_crop = tf.concat((y_predicted, x_predicted, y_1_predicted, x_1_predicted), axis=1)
            new_inputs = tf.concat(new_inputs, tf.image.crop_and_resize(self.inputs['images'], bb_to_crop,
                                                  tf.range(100), tf.constant([28, 28], dtype=tf.int32)), axis=1)

        print('new_inputs shape ', new_inputs.get_shape().as_list())

        # mse_loss =  tf.reduce_mean(tf.square(true_normalized - outputs[:, :]), name='mse')
        # tf.losses.add_loss(mse_loss)

        
        tf.identity(outputs[:, :], name='predicted_bb')



        # tf.identity(mse_loss, name='mse_loss')

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
        # print(tf.range(100), 'tf range')

        x = self.block(new_inputs, filters=16, bottleneck=False, data_format='channels_last', name='new0')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new1')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new2')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new3')
        x = self.block(x, filters=16, bottleneck=False, data_format='channels_last', name='new4')

        # print('x shape ', x.get_shape().as_list())
        second_outputs = conv_block(inputs, name='new5', layout='Vf', units=10 * (num_others + 1))
        
        ce_loss = tf.losses.softmax_cross_entropy(true_classes, second_outputs[:, :10])
        tf.losses.add_loss(ce_loss)
        # tf.identity(ce_loss, name='ce_loss')

        true_classes = tf.get_default_graph().get_tensor_by_name('DetectionResNet2/inputs/targets:0')
        other_labels = tf.get_default_graph().get_tensor_by_name('DetectionResNet2/inputs/other_labels:0')
        other_labels_len = other_labels.get_shape().as_list()[1]

        

        predicted_labels = tf.cast(tf.argmax(second_outputs[:, :10], axis=1), tf.int64)
        tf.identity(predicted_labels, name='labels_hat')
        
        accy = tf.contrib.metrics.accuracy(predicted_labels, tf.argmax(true_classes, axis=1), name='accuracy_0')
        accy = tf.expand_dims(accy, -1)
        # accy = tf.expand_dims(accy, -1)

        # print('accy ', accy.get_shape().as_list())
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
        tf.identity(accy, name='all_acuracies')
        # max_accy_name = 'DetectionResNet2/inputs/accuracy_' + str(max_accy_index) + ':0'
        max_accy =  tf.reduce_max(accy, axis=-1)
        # tf.identity(tf.get_default_graph().get_tensor_by_name(max_accy_name), name='accuracy')
        tf.identity(max_accy, name='accuracy')
        # tf.reduce_mean(tf.square(true_coordinates[:, :] - outputs[:, :]), name='mse')


        # predicted_probs = outputs
        return outputs


def normalize_bbox(width, height, coordinates):
    x_true = coordinates[:, 0:1]
    y_true = coordinates[:, 1:2]
    x_1_true = coordinates[:, 2:3]
    y_1_true = coordinates[:, 3:4]
    norm_x = (x_true / (width - 1))
    norm_y = (y_true / (height - 1))
    norm_x_1 = (x_1_true / (width - 1))
    norm_y_1 = (y_1_true / (height - 1))
    normalized_coordinates = tf.concat((norm_x, norm_y, norm_x_1, norm_y_1), axis=1)
    return normalized_coordinates
