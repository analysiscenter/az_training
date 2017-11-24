""" Custom batch class for storing mnist batch and ensemble models
"""
import sys

import numpy as np
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier

from dataset import Batch, action, model, ImagesBatch


class MnistBatch(ImagesBatch):
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

    @action
    def load(self, src, fmt='blosc'):
        """ Load mnist pics with specifed indices
        Args:
            fmt: format of source. Can be either 'blosc' or 'ndarray'
            src: if fmt='blosc', then src is a path to dir with blosc-packed
                mnist images and labels are stored.
                if fmt='ndarray' - this is a tuple with arrays of images and labels
        Return:
            self
        """
        if fmt == 'blosc':
            # read blosc images, labels
            with open('mnist_pics.blk', 'rb') as file:
                self.images = blosc.unpack_array(file.read())[self.indices]
                self.images = np.reshape(self.images, (65000, 28, 28))

            with open('mnist_labels.blk', 'rb') as file:
                self.labels = blosc.unpack_array(file.read())[self.indices]
        elif fmt == 'ndarray':
            all_images, all_labels = src
            self.images = all_images[self.indices]
            self.labels = all_labels[self.indices]

        return self

    @model(mode='dynamic')
    def resnet(self, config):
        """ Simple implementation of Resnet.
        Args:
            self
        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Plcaeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_labels - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.
            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')

            all_data = tf.placeholder(tf.float32, shape=[None, 28, 28], name='all_data')

            x_a = tf.gather_nd(all_data, indices)

            x1_to_tens = tf.reshape(x_a, shape=[-1, 28, 28, 1])

            net1 = tf.layers.conv2d(x1_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                    kernel_initializer=xavier(), name='11')
            net1 = tf.layers.max_pooling2d(net1, (2, 2), (2, 2))

            net1 = conv_block(net1, 3, [32, 32, 128], name='22', strides=(1, 1))



            net1 = conv_block(net1, 3, [32, 32, 128], name='in_33', strides=(1, 1))

            net1 = identity_block(net1, 3, [32, 32, 128], name='33')

            net1 = conv_block(net1, 3, [64, 64, 256], name='53', strides=(1, 1))
            net1 = identity_block(net1, 3, [64, 64, 256], name='63')

            net1 = tf.layers.average_pooling2d(net1, (7, 7), strides=(1, 1))
            net1 = tf.contrib.layers.flatten(net1)
            net1 = tf.layers.dense(net1, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())

            prob1 = tf.nn.softmax(net1, name='softmax_output')
            all_labels = tf.placeholder(tf.float32, [None, 10])

            y_ = tf.gather_nd(all_labels, indices)

            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net1, labels=y_), name='loss')

            global_step = tf.Variable(0, trainable=False)
            learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')

            train1 = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))
            labels_hat = tf.cast(tf.argmax(net1, axis=1), tf.float32, name='labels_hat')
            labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')

            accuracy1 = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32, name='accuracy'))
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        return [[indices, all_data, all_labels, loss, train1, prob1], [accuracy1, session], [learning_rate, global_step]]


    @action(model='resnet')
    def train_res(self, models, train_loss, accs, data, lables):
        """ Function for traning ResNet.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.
        Output:
            self """
        config = self.pipeline.config
        sess = models[1][1]
        grph = sess.graph
        with grph.as_default():
            accuracy = models[1][0]
            indices, all_data, all_labels, loss, train, _ = models[0]
            learning_rate, global_step = models[2]

            period = tf.cast(config['resnet']['period'], tf.float32)
            n_iterations = tf.cast(config['resnet']['n_iterations'], tf.float32)
            alpha = tf.cast(config['resnet']['alpha'], tf.float32)

            cyclic_learning_rate = (alpha / tf.cast(2.0, tf.float32) *
                                    (tf.cos(tf.cast(np.pi, tf.float32) *
                                            ((tf.cast(global_step, tf.float32) - 1) %
                                             (period)) / (period)) + 1))

            cyclic_learning_rate = sess.run(cyclic_learning_rate)

            sess.run(train, feed_dict={indices:self.indices.reshape(-1, 1), \
                     all_labels:lables, all_data:data, learning_rate:cyclic_learning_rate})

            period = sess.run(period)
            global_step = sess.run(global_step)

            if (global_step) % period == 0:
                if global_step == 0:
                    pass
                else:
                    saver = tf.train.Saver(name=str(global_step))
                    address = 'trained' + '+' + str(global_step) + '/model'
                    saver.save(sess, address, global_step=global_step)

            train_loss.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
                all_labels:lables, all_data:data}))

            accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1),\
             all_labels:lables, all_data:data}))

        return self


    @model(mode='dynamic')
    def ensemble(self):
        ''' Classifier which averages prediction from m models loaded from the disk
            Args:
            __
            Returns:
        '''

        config = self.pipeline.config
        period = config['resnet']['period']
        n_iterations = config['resnet']['n_iterations']

        n_cycles = n_iterations // period
        results = []
        ensemble_data = defaultdict(list)

        for i in range(1, n_cycles + 1):
            folder = 'trained+' + str(i*period) + '/'
            address = folder + 'model' + '-' + str(i*period) + '.meta'
            print('currently loading', address)

            grapphy_2 = tf.Graph()
            with grapphy_2.as_default():
                new_sess = tf.Session()

                new_saver = tf.train.import_meta_graph(address)
                new_saver.restore(new_sess, tf.train.latest_checkpoint(folder))
                indices = grapphy_2.get_tensor_by_name('indices:0')
                all_data = grapphy_2.get_tensor_by_name('all_data:0')
                softmax_output = grapphy_2.get_tensor_by_name('softmax_output:0')
                ensemble_data['sess'].append(new_sess)
                ensemble_data['graph'].append(grapphy_2)
                ensemble_data['indices'].append(indices)
                ensemble_data['all_data'].append(all_data)
                ensemble_data['softmax'].append(softmax_output)
        return ensemble_data

    @action(model='ensemble')
    def update_stats_ensemble(self, model, config, accs, loss_history, all_data, all_labels):
        ensemble_data = model
        results = []
        for i, sess in enumerate(ensemble_data['sess']):
            if i == 0:
                continue
            indices = ensemble_data['indices'][i]
            all_data_placehoder = ensemble_data['all_data'][i]
            softmax_output = ensemble_data['softmax'][i]
            results.append(sess.run(softmax_output, feed_dict={indices:self.indices.reshape(-1, 1), \
                                    all_data_placehoder:all_data}))
        avg_logits = np.mean(np.array(results), axis=0)
        small_graph = tf.Graph()
        with small_graph.as_default():
            y_ = tf.placeholder(tf.float32, [None, 10], name='y_')
            labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')

            logits = tf.placeholder(tf.float32, [None, 10], name='logits')
            labels_hat = tf.cast(tf.argmax(logits, axis=1), tf.float32, name='labels_hat')

            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_, name='loss'))

            small_sess = tf.Session()
            small_sess.run(tf.global_variables_initializer())
            labels = all_labels[self.indices]
            accs.append(small_sess.run(accuracy, feed_dict={y_:labels, logits:avg_logits}))
            loss_history.append(small_sess.run(loss, feed_dict={y_:labels, logits:avg_logits}))
        return self

    @model(mode='dynamic')
    def standard_resnet(self, config):
        """ Simple implementation of Resnet.
        Args:
            self
        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Plcaeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_labels - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.
            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """
        with tf.Graph().as_default():
            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_data = tf.placeholder(tf.float32, shape=[None, 28, 28], name='all_data')
            x_a = tf.gather_nd(all_data, indices)
            x1_to_tens = tf.reshape(x_a, shape=[-1, 28, 28, 1])
            net1 = tf.layers.conv2d(x1_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', \
                                    activation=tf.nn.relu, kernel_initializer=xavier(), name='11')
            net1 = tf.layers.max_pooling2d(net1, (2, 2), (2, 2))
            net1 = conv_block(net1, 3, [32, 32, 128], name='22', strides=(1, 1))
            net1 = conv_block(net1, 3, [32, 32, 128], name='in_33', strides=(1, 1))
            net1 = identity_block(net1, 3, [32, 32, 128], name='33')
            net1 = conv_block(net1, 3, [64, 64, 256], name='53', strides=(1, 1))
            net1 = identity_block(net1, 3, [64, 64, 256], name='63')
            net1 = tf.layers.average_pooling2d(net1, (7, 7), strides=(1, 1))
            net1 = tf.contrib.layers.flatten(net1)
            net1 = tf.layers.dense(net1, 10, kernel_initializer=tf.contrib.layers.xavier_initializer())
            prob1 = tf.nn.softmax(net1, name='softmax_output')
            all_labels = tf.placeholder(tf.float32, [None, 10])
            y_ = tf.gather_nd(all_labels, indices)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net1, labels=y_), name='loss')
            train1 = (tf.train.AdamOptimizer().minimize(loss))
            labels_hat = tf.cast(tf.argmax(net1, axis=1), tf.float32, name='labels_hat')
            labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
            accuracy1 = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32, name='accuracy'))
            session = tf.Session()
            session.run(tf.global_variables_initializer())
        return [[indices, all_data, all_labels, loss, train1, prob1], [accuracy1, session]]


    @action(model='standard_resnet')
    def train_standard_resnet(self, models, train_loss, accs, data, lables):
        """ Function for traning ResNet.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.
        Output:
            self """
        accuracy, sess = models[1]
        indices, all_data, all_lables, loss, train, _ = models[0]

        sess.run(train, feed_dict={indices:self.indices.reshape(-1, 1), all_lables:lables, all_data:data})

        train_loss.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
            all_lables:lables, all_data:data}))

        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1),\
            all_lables:lables, all_data:data}))
        return self


    @action(model='standard_resnet')
    def update_stats_standard_resnet(self, model, accs, loss_history, data, labels):
        """
        Append accuracy that is obtained by convy-model
        given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """

        sess = model[1][1]
        accuracy = model[1][0]
        indices, all_data, all_labels, loss, train, _ = model[0]

        loss_history.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
                                     all_data: data, all_labels: labels}))
        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1), \
                             all_data: data, all_labels: labels}))
        return self


    @action(model='resnet')
    def update_stats(self, model, accs, loss_history, data, labels):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session
        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        sess = model[1][1]
        accuracy = model[1][0]
        indices, all_data, all_labels, loss, train, _ = model[0]

        loss_history.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), all_data: data, all_labels: labels}))
        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1), all_data: data, all_labels: labels}))
        return self


    @action(model='resnet')
    def update_stats(self, model, accs, loss_history, data, labels):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        sess = model[1][1]
        accuracy = model[1][0]
        indices, all_data, all_labels, loss, train, _ = model[0]

        loss_history.append(sess.run(loss, feed_dict={indices:self.indices.reshape(-1, 1), \
                                     all_data: data, all_labels: labels}))
        accs.append(sess.run(accuracy, feed_dict={indices:self.indices.reshape(-1, 1), \
                             all_data: data, all_labels: labels}))
        return self

def conv_block(input_tensor, kernel, filters, name, strides=(2, 2)):
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
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='convo' + name, activation=tf.nn.relu,\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters2, kernel, name='convt' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name,\
                         kernel_initializer=xavier())

    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, name='short' + name,\
                         kernel_initializer=xavier())
    x = tf.add(x, shortcut)
    x = tf.nn.relu(x)
    return x

def identity_block(input_tensor, kernel, filters, name):
    """ Function to create block of ResNet network which incluce
    three convolution layers.
    Args:
        input_tensor: input tensorflow layer.
        kernel: tuple of kernel size in convolution layer.
        filters: list of nums filters in convolution layers.
        name: name of block.
    Output:
        x: Block output layer """
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='convo' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())


    x = tf.layers.conv2d(x, filters2, kernel, name='convt' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())

    x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name, padding='SAME',\
                         kernel_initializer=xavier())


    x = tf.add(x, input_tensor)
    x = tf.nn.relu(x)
    return x

def draw_stats(stats, title, y_label):
    plt.title(title)
    plt.plot(stats)
    plt.xlabel('iteration')
    plt.ylabel(y_label)
    plt.show()
