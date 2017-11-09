""" Custom batch class for storing mnist batch and ensemble models
"""
import sys

import numpy as np
import os
import blosc
import time

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.contrib.layers import xavier_initializer_conv2d

from dataset import Batch, action, model, inbatch_parallel
from dataset.dataset.image import ImagesBatch


class MnistBatch_new(ImagesBatch):
    """ Mnist batch and models
    """
    def __init__(self, index, *args, **kwargs):
        """ Init func, inherited from base batch
        """
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.labels = None
    
    @property
    def components(self):
        """ Components of mnist-batch
        """
        return 'images', 'labels'


    @model(mode='dynamic')
    def resnet(self, config):
        """ Simple implementation of Resnet.
        Args:
            self
        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Plcaeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_lables - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.
            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """

        conv_initializer = self.pipeline.config['resnet']['conv_initializer']
        dense_initializer = self.pipeline.config['resnet']['dense_initializer']

        # conv_initializer = xavier_initializer_conv2d()
        # dense_initializer = tf.contrib.layers.xavier_initializer()
        widening_factor = self.pipeline.config['resnet']['factor']
        length_factor = self.pipeline.config['resnet']['length_factor'] - 1

        additional_blocks = self.pipeline.config['resnet']['add_blocks']

        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1])
            all_data = tf.placeholder(tf.float32, shape=[None, 28, 28])
            training_mode = tf.placeholder(tf.bool, shape=[])
            drop_rate = tf.placeholder(tf.float32, shape=[])

            x_a = tf.gather_nd(all_data, indices)

            x1_to_tens = tf.reshape(x_a, shape=[-1, 28, 28, 1])

            net1 = tf.layers.conv2d(x1_to_tens, 16, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                kernel_initializer=conv_initializer, name='11')

            # net1 = tf.layers.max_pooling2d(net1, (2, 2), (2, 2))

            net1 = conv_block(net1, 3, [16, 16], name='22', strides=(2, 2), initializer=conv_initializer, \
                w_factor=widening_factor, training=training_mode, rate=drop_rate)
            
            for block_length in range(length_factor):
                net1  = identity_block(net1, 3, [16, 16], name=str(block_length) +'n1', initializer=conv_initializer, \
                    w_factor=widening_factor, training=training_mode, rate=drop_rate)

            net1 = conv_block(net1, 3, [32, 32], name='33', strides=(1, 1), initializer=conv_initializer, \
                w_factor=widening_factor, training=training_mode, rate=drop_rate)
            
            for block_length in range(length_factor):
                net1  = identity_block(net1, 3, [32, 32], name=str(block_length) +'n2', initializer=conv_initializer, \
                    w_factor=widening_factor, training=training_mode, rate=drop_rate)

            # net1 = conv_block(net1, 3, [64, 64], name='53', strides=(1, 1), initializer=conv_initializer, w_factor=widening_factor)
            net1 = conv_block(net1, 3, [64, 64], name='63', strides=(1, 1), initializer=conv_initializer, \
                w_factor=widening_factor, training=training_mode, rate=drop_rate)

            # net1 = conv_block(net1, 3, [64, 64], name='63', strides=(2, 2), initializer=conv_initializer, w_factor=widening_factor)
        
            for block_length in range(length_factor):
                net1  = identity_block(net1, 3, [64, 64], name=str(block_length) +'n3', initializer=conv_initializer, \
                    w_factor=widening_factor, training=training_mode, rate=drop_rate)

            for add_block in range(additional_blocks):
                # net1 = conv_block(net1, 3, [64, 64], name='73_'+ str(add_block), strides=(1, 1), initializer=conv_initializer, w_factor=widening_factor)
                net1  = identity_block(net1, 3, [64, 64], name=str(add_block) +'id_n3', initializer=conv_initializer, \
                    w_factor=widening_factor, training=training_mode, rate=drop_rate)



            net1 = tf.layers.average_pooling2d(net1, (7, 7), strides=(1, 1))
            net1 = tf.contrib.layers.flatten(net1)

            # with tf.variable_scope('dense3'):
            net1 = tf.layers.dense(net1, 10, kernel_initializer=dense_initializer)


            prob1 = tf.nn.softmax(net1)
            all_lables = tf.placeholder(tf.float32, [None, 10])

            y_ = tf.gather_nd(all_lables, indices)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net1, labels=y_), name='loss3')            

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train1 = (tf.train.AdamOptimizer().minimize(loss))



            labels_hat = tf.cast(tf.argmax(net1, axis=1), tf.float32, name='labels_hat')
            labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')

            accuracy1 = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32, name='a3ccuracy'))
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        return [[indices, all_data, all_lables, loss, train1, prob1], [accuracy1, session], [training_mode, drop_rate]]


    @action(model='resnet')
    def train_res(self, models, train_loss, accs, data, lables, time_list, rate):
        """ Function for traning ResNet.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.
        Output:
            self """

        # config = self.pipeline.config
        accuracy, sess = models[1]
        indices, all_data, all_lables, loss, train, _ = models[0]
        training, drop_rate = models[2]

        start_time = time.clock()
        sess.run(train, feed_dict={indices:self.indices.reshape(-1, 1), all_lables:lables, all_data:data, training:True, drop_rate:rate})
        measured_time = time.clock() - start_time
        time_list.append(measured_time)

        train_loss.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
            all_lables:lables, all_data:data, training:True, drop_rate:rate}))

        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1),\
            all_lables:lables, all_data:data, training:True, drop_rate:rate}))

        return self



    # @action(model='convy')
    # def predict(self, model, sess, pics, y_true, y_predict, probabilities):
    #     ''' Predict labels '''
    #     sess = model[5][0]
    #     x, y_, _, _, training, keep_prob = model[0]
    #     labels, labels_hat, _ = model[1]
    #     probs = model[2][0]
    #     probabilities.append(sess.run(probs, feed_dict={x:self.images, training: False, keep_prob: 1.0}))
    #     y_predict.append(sess.run(labels_hat, feed_dict={x:self.images, training: False, keep_prob: 1.0}))
    #     y_true.append(sess.run(labels, feed_dict={y_:self.labels}))
    #     pics.append(self.images)
    #     return self

    
    @action(model='resnet')
    def update_stats(self, models, test_loss, accs, data, lables, rate):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        
        # _, _, accuracy = model[1]
        # x, y_, loss, _, training, keep_prob = model[0]
        # loss_history.append(sess.run(loss, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))

        # accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))
        

        # config = self.pipeline.config
        training, drop_rate = models[2]
        accuracy, sess = models[1]
        indices, all_data, all_lables, loss, train, _ = models[0]

        test_loss.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
            all_lables:lables, all_data:data, training:False, drop_rate:rate}))

        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1),\
            all_lables:lables, all_data:data, training:False, drop_rate:rate}))

        return self


def conv_block(input_tensor, kernel, filters, name, initializer, strides, w_factor, training, rate):
    """ Function to create block of ResNet network which incluce
    three convolution layers and one skip-connetion layer.
    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.
        strides: typle of strides in convolution layer.
    Output:
        x: Block output layer """
    filters = [int(filt * w_factor) for filt in filters]
    filters1, filters2 = filters
    print(filters1, filters2)

    # x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='convo' + name, activation=tf.nn.relu,\
    #                      kernel_initializer=initializer)

    x = tf.layers.conv2d(input_tensor, filters1, kernel, strides, name='01' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=initializer)


    x = tf.layers.dropout(x, rate, training)


    x = tf.layers.conv2d(x, filters2, kernel, name='02' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=initializer)

    # x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name,\
    #                      kernel_initializer=initializer)
    print(x.get_shape().as_list())
    shortcut = tf.layers.conv2d(input_tensor, filters2, (1, 1), strides, name='short' + name, \
                             kernel_initializer=initializer)
    print(shortcut.get_shape().as_list())
    x = tf.add(x, shortcut)
    x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate, training)

    return x

def identity_block(input_tensor, kernel, filters, name, initializer, w_factor, training, rate):
    """ Function to create block of ResNet network which include
    three convolution layers.
    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.
    Output:
        x: Block output layer """

    filters = [int(filt * w_factor) for filt in filters]

    filters1, filters2 = filters
    print(filters1, filters2)
    # x = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='convo' + name, activation=tf.nn.relu,\
    #                      kernel_initializer=initializer)

    x = tf.layers.conv2d(input_tensor, filters2, kernel, name='01' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=initializer)

    x = tf.layers.dropout(x, rate, training)

    x = tf.layers.conv2d(x, filters2, kernel, name='02' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=initializer)

    # x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name,\
    #                      kernel_initializer=initializer)


    x = tf.add(x, input_tensor)
    x = tf.nn.relu(x)
    # x = tf.layers.dropout(x, rate, training)

    return x


def draw_stats(stats, title):
    plt.title(title)
    plt.plot(stats)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()