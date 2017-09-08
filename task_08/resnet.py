""" File with many variants of ResNet's """
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier

sys.path.append('..')
from dataset import action, model, Batch

def bottle_conv_block(input_tensor, kernel, filters, trainable=None, name=None, \
 strides=(2, 2), prob_on=1.):
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
    if name == None:
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
    
    if trainable is not None:
        off = tf.cond(trainable, \
              lambda: tf.where(tf.random_uniform([1, ], 0, 1)> (1 - prob_on), tf.ones([1, ]), \
              tf.zeros([1, ])), lambda: tf.ones([1, ]) * prob_on)[0]
        X = X * off

    X = tf.add(X, shortcut)
    X = tf.nn.relu(X)
    return X

def bottle_identity_block(input_tensor, kernel, filters, trainable=None, name=None, prob_on=1.):
    """ Function to create bottleneck-identity block of ResNet network.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.

    Output:
        X: Block output layer """
    if name == None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3 = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
                         kernel_initializer=Xavier(uniform=True))
   
    if trainable is not None:
        off = tf.cond(trainable, \
              lambda: tf.where(tf.random_uniform([1, ], 0, 1)> (1 - prob_on), tf.ones([1, ]), \
              tf.zeros([1, ])), lambda: tf.ones([1, ]) * prob_on)[0]
        X = X * off

    X = tf.add(X, input_tensor)
    X = tf.nn.relu(X)
    return X

def conv_block(input_tensor, kernel, filters, name, strides=(2, 2)):
    """ Function to create block of ResNet network which incluce
    two convolution layers.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: int defining nums filters in convolution layers.
        name: name of block.
        strides: typle of strides in convolution layer.

    Output:
        X: Block output layer """
    X = tf.layers.conv2d(input_tensor, filters, kernel, strides, name='first_conv_3X3_of_' + name, \
        activation=tf.nn.relu, padding='SAME',kernel_initializer=Xavier(uniform=True))
    X = tf.layers.conv2d(input_tensor, filters, kernel, name='second_conv_3X3_of_' + name, \
        activation=tf.nn.relu, padding='SAME', kernel_initializer=Xavier(uniform=True))
    return X

def identity_block(input_tensor, kernel, filters, name):
    """"Function to create block of ResNet network which incluce
    two convolution layers and skipconnection.

    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: int defining nums filters in convolution layers.
        name: name of block.
        strides: typle of strides in convolution layer.

    Output:
        X: Block output layer """
    X = tf.layers.conv2d(input_tensor, filters, kernel, name='first_conv_3X3_of_' + name, \
        activation=tf.nn.relu, padding='SAME', kernel_initializer=Xavier(uniform=True))
    X = tf.layers.conv2d(input_tensor, filters, kernel, name='second_conv_3X3_of_' + name, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))
    
    X = tf.add(X, input_tensor)
    X = tf.nn.relu(X)
    return X

def accuracy(ind, model, data):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
    Output:
        acc: accuracy after one iteration.
        loss: loss after one iteration. """
    indices, all_images, all_labels, trainable, loss, _ = model[0]
    acc, sess = model[1]
    loss, acc = sess.run([loss, acc], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1], trainable: False})
    return acc, loss

def train(ind, model, data, all_time):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
        all_time: list with len=1, calculate time for training model.
    Output:
        loss: loss after one iteration. """
    indices, all_images, all_labels, trainable, loss, train = model[0]
    sess = model[1][1]
    t = time.time()
    _, loss = sess.run([train, loss], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1], trainable: True})
    all_time[0] += time.time() - t
    return loss

class ResBottleBatch(Batch):
    """ Class to compare results of training ResNet with bottleneck and without bottleneck """
    def __init__(self, indeX, *args, **kwargs):
        """ Init function """
        super().__init__(indeX, *args, **kwargs)

    @model(mode='dynamic')
    def bottlenet(self):
        """ Model with bottleneck blocks. Using in ResNet_vs_BottleResNet.ipynb"""
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')
            trainable = tf.placeholder(tf.bool, shape=[])

            X_a = tf.gather_nd(all_images, indices, name='X_a')

            X_f_to_tens = tf.reshape(X_a, shape=[-1, 28, 28, 1], name='X_to_tens')

            net = tf.layers.conv2d(X_f_to_tens, 64, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')

            filters = np.array([32, 32, 128])
            net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), name='max_pool')

            net = bottle_identity_block(net, 3, [16, 16, 64], name='first_identity_block')
           
            net = bottle_conv_block(net, 3, filters, name='first_conv_block', strides=(2, 2))
            net = bottle_identity_block(net, 3, filters, name='second_identity_block')

            net = bottle_conv_block(net, 3, filters * 2, name='second_conv_block', strides=(2, 2))
            net = bottle_identity_block(net, 3, filters * 2, name='third_identity_block')

            net = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')

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

        return [[indices, all_images, all_labels, trainable, loss, train_step], [accuracy, session], [labels_predict]]

    @action(model='bottlenet')
    def train_bottle(self, model, data, train_loss, all_time):
        """Function to train bottlenet model.
        Args:
            data: all dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        train_loss.append(train(self.indices, model, data, all_time))
        return self


    @action(model='bottlenet')
    def accuracy_bottle(self, model, data, bottle_acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """
        bottle_acc.append(accuracy(self.indices, model, data))
        return self

    @model(mode='dynamic')
    def resnet(self):
        """ Model with simple ResNet blocks. Usinge in ResNet_vs_BottleResNet.ipynb"""
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')
            X_a = tf.gather_nd(all_images, indices, name='X_a')
            trainable = tf.placeholder(tf.bool, shape=[])

            X_f_to_tens = tf.reshape(X_a, shape=[-1, 28, 28, 1], name='X_to_tens')

            net = tf.layers.conv2d(X_f_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')

            net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), name='max_pool')

            net = identity_block(net, 3, 32, name='first_identity_block')
            
            net = conv_block(net, 3, 64, name='first_conv_block', strides= (2, 2))
            net = identity_block(net, 3, 64, name='second_identity_block')
            
            net = conv_block(net, 3, 128, name='second_conv_block', strides= (2, 2))
            net = identity_block(net, 3, 128, name='third_identity_block')

            net = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='dense')

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

        return [[indices, all_images, all_labels, trainable, loss, train_step], [accuracy, session]]

    @action(model='resnet')
    def train_resnet(self, model, data, train_loss, all_time):
        """Function to train bottlenet model.
        Args:
            data: training dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        train_loss.append(train(self.indices, model, data, all_time))
        return self

    @action(model='resnet')
    def accuracy_res(self, model, data, resnet_acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """

        resnet_acc.append(accuracy(self.indices, model, data))
        return self

    @model(mode='dynamic')
    def stochasticnet(self):
        """ Model with stochastic ResNet-50, unsing in tochastic_resnet.ipynb """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')
            trainable = tf.placeholder(tf.bool, shape=[], name='trainable')

            X_a = tf.gather_nd(all_images, indices, name='X_a')

            X_f_to_tens = tf.reshape(X_a, shape=[-1, 28, 28, 1], name='X_to_tens')

            threshold = np.linspace(1, 0.5, 17)
            

            net = tf.layers.conv2d(X_f_to_tens, 64, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')
            net = tf.layers.maX_pooling2d(net, (2, 2), (2, 2), name='maX_pool')

            filters = np.array([32, 32, 128])
            net = bottle_conv_block(net, 3, filters, trainable, strides=(1, 1), prob_on=threshold[0])
            net = bottle_identity_block(net, 3, filters, trainable, prob_on=threshold[1])
            net = bottle_identity_block(net, 3, filters, trainable, prob_on=threshold[2])

            net = bottle_conv_block(net, 3, filters * 2, trainable, strides=(2, 2), prob_on=threshold[3])
            net = bottle_identity_block(net, 3, filters * 2, trainable, prob_on=threshold[4])
            net = bottle_identity_block(net, 3, filters * 2, trainable, prob_on=threshold[5])
            net = bottle_identity_block(net, 3, filters * 2, trainable, prob_on=threshold[6])

            net = bottle_conv_block(net, 3, filters * 4, trainable, strides=(2, 2), prob_on=threshold[7])
            net = bottle_identity_block(net, 3, filters * 4, trainable, prob_on=threshold[8])
            net = bottle_identity_block(net, 3, filters * 4, trainable, prob_on=threshold[9])
            net = bottle_identity_block(net, 3, filters * 4, trainable, prob_on=threshold[10])
            net = bottle_identity_block(net, 3, filters * 4, trainable, prob_on=threshold[11])
            net = bottle_identity_block(net, 3, filters * 4, trainable, prob_on=threshold[12])

            net = bottle_conv_block(net, 3, filters * 8, trainable, strides=(1, 1), prob_on=threshold[12])
            net = bottle_identity_block(net, 3, filters * 8, trainable, prob_on=threshold[13])
            net = bottle_identity_block(net, 3, filters * 8, trainable, prob_on=threshold[14])
            net = bottle_identity_block(net, 3, filters * 8, trainable, prob_on=threshold[15])
            net = bottle_identity_block(net, 3, filters * 8, trainable, prob_on=threshold[16])

            net = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.Xavier_initializer(), name='dense')

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

        return [[indices, all_images, all_labels, trainable, loss, train_step], [accuracy, session]]

    @action(model='stochasticnet')
    def train_stochastic(self, model, data, train_loss, all_time):
        """Function to train bottlenet model.
        Args:
            data: training dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        train_loss.append(train(self.indices, model, data, all_time))
        return self

    @action(model='stochasticnet')
    def accuracy_stochastic(self, model, data, stochastic_acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """
        stochastic_acc.append(accuracy(self.indices, model, data))
        return self

    @model(mode='dynamic')
    def bigbottlenet(self):
        """ Model with ResNet-50, using in tochastic_resnet.ipynb """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_images = tf.placeholder(tf.float32, shape=[65000, 28, 28], name='all_data')
            trainable = tf.placeholder(tf.bool, shape=[])

            X_a = tf.gather_nd(all_images, indices, name='X_a')

            X_f_to_tens = tf.reshape(X_a, shape=[-1, 28, 28, 1], name='X_to_tens')

            net = tf.layers.conv2d(X_f_to_tens, 64, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=Xavier(), name='first_convolution')

            net = tf.layers.maX_pooling2d(net, (2, 2), (2, 2), name='maX_pool')

           
            filters = np.array([32, 32, 128])
            net = bottle_conv_block(net, 3, filters , strides=(1, 1))
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

            net = bottle_conv_block(net, 3, filters * 8, strides=(1, 1))
            net = bottle_identity_block(net, 3, filters * 8)
            net = bottle_identity_block(net, 3, filters * 8)
            net = bottle_identity_block(net, 3, filters * 8)
            net = bottle_identity_block(net, 3, filters * 8)

            net = tf.layers.average_pooling2d(net, (2, 2), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)
            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.Xavier_initializer(), name='dense')

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

        return [[indices, all_images, all_labels, trainable, loss, train_step], [accuracy, session], [labels_predict]]

    @action(model='bigbottlenet')
    def train_bigbottle(self, model, data, train_loss, all_time):
        """Function to train bottlenet model.
        Args:
            data: training dataset.
            train_loss: list with train loss values.
            all_time: time of model work.

        Output:
            self"""
        train_loss.append(train(self.indices, model, data, all_time))
        return self


    @action(model='bigbottlenet')
    def accuracy_bigbottle(self, model, data, bottle_acc):
        """ Function to calculate accuracy.
        Args:
            data: all dataset.
            bottle_acc: list with accuracy values.

        Output:
            self """
        bottle_acc.append(accuracy(self.indices, model, data))
        return self
