""" Custom batch class for storing mnist batch and models
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt

from dataset import Batch, action, model, inbatch_parallel, ImagesBatch


class MnistBatch(ImagesBatch):
    """ Mnist batch and models
    """
    def __init__(self, index, *args, **kwargs):
        """ Init func, inherited from base batch
        """
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.labels = None



    def post_function(self, list_results):
        '''Post function for parallel shift, gathers results of every worker'''
        result_batch = np.array(list_results)
        self.images = result_batch
        return self

    def init_function(self):
        '''Init function for parallel shift
        returns list of indices, each of them will be sent to the worker separately
        '''
        return range(self.images.shape[0])

    @action
    @inbatch_parallel(init='init_function', post='post_function', target='threads')
    def shift_flattened_pic(self, idx, max_margin=8):
        """ Apply random shift to a flattened pic
        
        Args:
            pic: ndarray of shape=(784) representing a pic to be flattened
        Return:
            flattened shifted pic
        """
        
        pic = self.images[idx]
        padded = np.pad(pic, pad_width=[[max_margin, max_margin], [max_margin, max_margin]], 
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        res = padded[slicing]
        return res

    
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

    @model()
    def convy():
        """ Conv-net mnist classifier

        Args:
            ___
        Return:
            [[placeholder for input, ph for true labels, loss, train_step],
             [true categorical labels, categorical_hat labels, accuracy]]
        """
        # build the net
        training = tf.placeholder(tf.bool, shape=[], name='mode')
        x = tf.placeholder(tf.float32, [None, 28, 28])
        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])
        
        
        net = tf.layers.conv2d(x_as_pics, filters=4, kernel_size=(7,7), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(6, 6), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)
        
        net = tf.layers.conv2d(net, filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(5, 5), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)


        net = tf.layers.conv2d(net, filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same')
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(2, 2), padding='same')
        net = tf.layers.batch_normalization(net, training=training)
        net = tf.nn.relu(net)

        
        net = tf.contrib.layers.flatten(net)

        # dropout 
        keep_prob = tf.placeholder(tf.float32)
        # net = tf.nn.dropout(net, keep_prob)


        net = tf.layers.dense(net, 128, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))

        net = tf.nn.relu(net)
        
        net = tf.layers.dense(net, 10, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))


        probs = tf.nn.softmax(logits=net)
        # placeholder for correct labels
        y_ = tf.placeholder(tf.float32, [None, 10])

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')
        
        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.00001
        learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                                   100, 0.85, staircase=True)

        # optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = (
                tf.train.GradientDescentOptimizer(learning_rate)
                .minimize(loss, global_step=global_step)
                )

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')
        

        return [[x, y_, loss, train_step, training, keep_prob], [labels, labels_hat, accuracy], [probs], [learning_rate]]

    @action(model='convy')
    def predict(self, model, sess, pics, y_true, y_predict, probabilities):
        ''' Predict labels '''
        x, y_, _, _, training, keep_prob = model[0]
        labels, labels_hat, _ = model[1]
        probs = model[2][0]
        probabilities.append(sess.run(probs, feed_dict={x:self.images, training: False, keep_prob: 1.0}))
        y_predict.append(sess.run(labels_hat, feed_dict={x:self.images, training: False, keep_prob: 1.0}))
        y_true.append(sess.run(labels, feed_dict={y_:self.labels}))
        pics.append(self.images)
        return self

    @action(model='convy')
    def train_convy(self, model, sess):
        """ Train-action for convy-model

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session in which learning variables are to be updated
        """
        
        x, y_, _, train_step, training, keep_prob = model[0]
        sess.run(train_step, feed_dict={x: self.images, y_: self.labels, training: True, keep_prob: 0.7})        
        return self

    @action(model='convy')
    def update_stats(self, model, sess, accs, lr_history):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        _, _, accuracy = model[1]
        x, y_, _, _, training, keep_prob = model[0]
        learning_rate = model[3]
        lr_history.append(sess.run(learning_rate))

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))
        return self

def draw_stats(stats, title):
    plt.title(title)
    plt.plot(stats)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()