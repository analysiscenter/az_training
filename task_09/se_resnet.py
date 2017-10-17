""" File with se-resnet """
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier

sys.path.append('..')
from dataset import action, model, Batch

def bottle_conv_block(input_tensor, kernel, filters, name=None, \
 strides=(2, 2)):
    """ Function to create block bottleneck of ResNet network which incluce
    three convolution layers and one skip-connetion layer.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.
        strides: typle of strides in convolution layer.

    Output:
        X: Block output layer """
    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3 = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
        kernel_initializer=Xavier(uniform=True))

    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, \
               name='shortcut_conv_1X1_of_' + name, kernel_initializer=Xavier(uniform=True))

    X = tf.add(X, shortcut)
    X = tf.nn.relu(X)
    return X

def bottle_identity_block(input_tensor, kernel, filters, name=None):
    """ Function to create bottleneck-identity block of ResNet network.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.

    Output:
        X: Block output layer """
    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3 = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
                         kernel_initializer=Xavier(uniform=True))

    X = tf.add(X, input_tensor)
    X = tf.nn.relu(X)
    return X

def se_conv_block(input_tensor, kernel, filters, name=None, \
 strides=(2, 2)):
    """ Function to create block bottleneck of ResNet network which incluce
    three convolution layers and one skip-connetion layer.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.
        strides: typle of strides in convolution layer.

    Output:
        X: Block output layer """

    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3, r, C = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
        kernel_initializer=Xavier(uniform=True))

    full = tf.reduce_mean(X, [1, 2])
    full = tf.reshape(full, [-1, 1, 1, C])
    full = tf.layers.dense(full, int(C/r), activation=tf.nn.relu, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    full = tf.layers.dense(full, C, activation=tf.nn.sigmoid, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    X = X * full

    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, \
               name='shortcut_conv_1X1_of_' + name, kernel_initializer=Xavier(uniform=True))

    X = tf.add(X, shortcut)
    X = tf.nn.relu(X)
    return X

def se_identity_block(input_tensor, kernel, filters, name=None):
    """ Function to create bottleneck-identity block of ResNet network.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.

    Output:
        X: Block output layer """
    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3, r, C = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
                         kernel_initializer=Xavier(uniform=True))

    full = tf.reduce_mean(X, [1, 2])
    full = tf.reshape(full, [-1, 1, 1, C])
    full = tf.layers.dense(full, int(C/r), activation=tf.nn.relu, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    full = tf.layers.dense(full, C, activation=tf.nn.sigmoid, \
        kernel_initializer=tf.contrib.layers.xavier_initializer())
    X = X * full

    X = tf.add(X, input_tensor)
    X = tf.nn.relu(X)
    return X

def accum_accuracy(ind, models, data):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
    Output:
        acc: accuracy after one iteration.
        loss: loss after one iteration. """
    indices, all_images, all_labels, loss, _ = models[0]
    acc, sess = models[1]
    loss, acc = sess.run([loss, acc], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1]})
    return acc, loss

def training(ind, models, data, all_time):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
        all_time: list with len=1, calculate time for training model.
    Output:
        loss: loss after one iteration. """
    indices, all_images, all_labels, loss, train = models[0]
    sess = model[1][1]
    timer = time.time()
    _, loss = sess.run([train, loss], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1]})
    all_time[0] += time.time() - timer
    return loss

def get_maps(ind, models, data, maps):
    """ function to get filters from model """
    indices, all_images, all_labels = models[0][:3]
    sess = models[1][1]
    filters = models[2][:-1]
    filters = sess.run(filters, {indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1]})
    maps['avgpool'].append(filters[0])
    maps['answers'].append(filters[1])
    return maps
class Seresnet(Batch):
    """ Class to compare results of training ResNet with Squeese and excitation and simple ResNet"""
    def __init__(self, index, *args, **kwargs):
        """ Init function """
        super().__init__(index, *args, **kwargs)

    @model(mode='dynamic')
    def seresnet(self):
        """create se res net model"""
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')

            all_images = tf.gather_nd(all_images, indices, name='all_images')

            input_batch = tf.reshape(all_images, shape=[-1, 28, 28, 1], name='X_to_tens')

            net = tf.layers.conv2d(input_batch, 64, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')


            filters = np.array([32, 32, 128, 8, 128]) * 2
            net = se_conv_block(net, 3, filters, strides=(1, 1))
            net = se_identity_block(net, 3, filters)
            net = se_identity_block(net, 3, filters)

            net = se_conv_block(net, 3, filters * 2, strides=(2, 2))
            net = se_identity_block(net, 3, filters * 2)
            net = se_identity_block(net, 3, filters * 2)
            net = se_identity_block(net, 3, filters * 2)

            net = se_conv_block(net, 3, filters * 4, strides=(2, 2))
            net = se_identity_block(net, 3, filters * 4)
            net = se_identity_block(net, 3, filters * 4)
            net = se_identity_block(net, 3, filters * 4)
            net = se_identity_block(net, 3, filters * 4)
            net = se_identity_block(net, 3, filters * 4)

            net = se_conv_block(net, 3, filters * 8, strides=(2, 2))
            net = se_identity_block(net, 3, filters * 8)
            net = se_identity_block(net, 3, filters * 8)

            avgpool = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(avgpool)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())

            prob = tf.nn.softmax(net, name='soft')

            all_labels = tf.placeholder(tf.float32, [None, 10], name='all_labels')
            y = tf.gather_nd(all_labels, indices, name='y')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y), name='loss')

            train_step = tf.train.AdamOptimizer().minimize(loss)
            labels_predict = tf.cast(tf.argmax(net, axis=1), tf.float32)
            labels_true = tf.cast(tf.argmax(y, axis=1), tf.float32)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_predict, labels_true), tf.float32))

            session = tf.Session()
            session.run(tf.global_variables_initializer())

        return [[indices, all_images, all_labels, loss, train_step], [accuracy, session], [avgpool, y, prob]]

    @action(model='seresnet')
    def train_se(self, models, data, list_loss, all_time):
        """Function to train SE model.
        Args:
            data: all dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        list_loss.append(training(self.indices, models, data, all_time))
        return self

    @action(model='seresnet')
    def get_maps_se(self, models, data, maps):
        """ function to get filters from model """
        maps = get_maps(self.indices, models, data, maps)
        return self

    @action(model='seresnet')
    def accuracy_se(self, models, data, acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """
        acc.append(accum_accuracy(self.indices, models, data))
        return self


    @model(mode='dynamic')
    def bottlenet(self):
        """ Model with ResNet-50, using in tochastic_resnet.ipynb """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')

            all_images = tf.gather_nd(all_images, indices, name='all_images')

            input_batch = tf.reshape(all_images, shape=[-1, 28, 28, 1], name='X_to_tens')
            net = tf.layers.conv2d(input_batch, 64, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')

            #net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), name='maX_pool')

            filters = np.array([32, 32, 128]) * 2
            net = bottle_conv_block(net, 3, filters, strides=(1, 1))
            net = bottle_identity_block(net, 3, filters)
            net = bottle_identity_block(net, 3, filters)

            net = bottle_conv_block(net, 3, filters * 2, strides=(2, 2))
            net = bottle_identity_block(net, 3, filters * 2)
            net = bottle_identity_block(net, 3, filters * 2)
            net = bottle_identity_block(net, 3, filters * 2)

            net = bottle_conv_block(net, 3, filters * 4, strides=(2, 2))
            net = bottle_identity_block(net, 3, filters * 4)
            net = bottle_identity_block(net, 3, filters * 4)
            net = bottle_identity_block(net, 3, filters * 4)
            net = bottle_identity_block(net, 3, filters * 4)
            net = bottle_identity_block(net, 3, filters * 4)

            net = bottle_conv_block(net, 3, filters * 8, strides=(2, 2))
            net = bottle_identity_block(net, 3, filters * 8)
            net = bottle_identity_block(net, 3, filters * 8)
            avgpool = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())

            prob = tf.nn.softmax(net, name='soft')

            all_labels = tf.placeholder(tf.float32, [None, 10], name='all_labels')
            y = tf.gather_nd(all_labels, indices, name='y')

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y), name='loss')

            train_step = tf.train.AdamOptimizer().minimize(loss)
            labels_predict = tf.cast(tf.argmax(net, axis=1), tf.float32)
            labels_true = tf.cast(tf.argmax(y, axis=1), tf.float32)

            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_predict, labels_true), tf.float32))

            session = tf.Session()
            session.run(tf.global_variables_initializer())

        return [[indices, all_images, all_labels, loss, train_step], [accuracy, session], [avgpool, y, prob]]

    @action(model='bottlenet')
    def train_bottle(self, models, data, train_loss, all_time):
        """Function to train bottlenet model.
        Args:
            data: training dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        train_loss.append(training(self.indices, models, data, all_time))
        return self


    @action(model='bottlenet')
    def accuracy_bottle(self, models, data, bottle_acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """
        bottle_acc.append(accum_accuracy(self.indices, models, data))
        return self

    @action(model='bottlenet')
    def get_maps_bo(self, models, data, maps):
        """ function to get filters from model """
        maps = get_maps(self.indices, models, data, maps)
        return self
