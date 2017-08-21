"""Class with conwolution network to training on mnist dataset """
import sys
import os

import blosc
import numpy as np
import tensorflow as tf

sys.path.append('..')
from dataset import action, model, inbatch_parallel, ImagesBatch, any_action_failed

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
            with open(os.path.join(src, 'mnist_pics.blk'), 'rb') as file:
                self.images = blosc.unpack_array(file.read())[self.indices].reshape(-1, 28, 28)

            with open(os.path.join(src, 'mnist_labels.blk'), 'rb') as file:
                self.labels = blosc.unpack_array(file.read())[self.indices]
        elif fmt == 'ndarray':
            all_images, all_labels = src
            self.images = all_images[self.indices].reshape(-1, 28, 28)
            self.labels = all_labels[self.indices]

        return self

    @model()
    def convolution_nn():
        """ Conv-net mnist classifier
        Args:
            ___
        Return:
            [[placeholder for input, ph for true labels, loss, train_step],
             [true categorical labels, categorical_hat labels, accuracy]]
        """
        # build the net

        bnorm = True

        training = tf.placeholder(tf.bool, shape=[], name='mode')
        x = tf.placeholder(tf.float32, [None, 784], name='x')
        keep_prob = tf.placeholder(tf.float32)

        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])

        net = tf.layers.conv2d(inputs=x_as_pics, filters=16, kernel_size=(7, 7), strides=(2, 2), \
                               padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # if bnorm:
        #     net = tf.layers.batch_normalization(net, training=training, name='batch-norm5')
        net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=(2, 2))#, padding='SAME')
        # bnorm if needed
        if bnorm:
            net = tf.layers.batch_normalization(net, training=training, name='batch-norm1')
        net = tf.nn.relu(net)

        net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(5, 5), strides=(1, 1), \
                               padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        # if bnorm:
        #     net = tf.layers.batch_normalization(net, training=training, name='batch-norm4')
        net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2))#, padding='SAME')
        # bnorm if needed
        if bnorm:
            net = tf.layers.batch_normalization(net, training=training, name='batch-norm2')
        net = tf.nn.relu(net)
        net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), strides=(1, 1), \
                               padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
        net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(1, 1), padding='SAME')
        # bnorm if needed
        if bnorm:
            net = tf.layers.batch_normalization(net, training=training, name='batch-norm3')


        net = tf.layers.dropout(net, keep_prob, training=training)
        net = tf.contrib.layers.flatten(net)
        net = tf.contrib.layers.fully_connected(net, 128)
        net = tf.layers.dropout(net, keep_prob, training=training)
        net = tf.contrib.layers.fully_connected(net, 10)


        # placeholder for correct labels
        y_ = tf.placeholder(tf.float32, [None, 10])

        y = tf.nn.softmax(net)
        # loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss'))



        global_step = tf.Variable(0, trainable=False)
        starter_lr = 0.0001
        learning_rate = tf.train.exponential_decay(starter_lr, global_step, 150, 0.96, staircase=True)

        #optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.RMSPropOptimizer(learning_rate) \
                                                   .minimize(loss, global_step=global_step)

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

        return [[x, y_, loss, train_step, training, keep_prob, y], [labels, labels_hat, accuracy, learning_rate]]

    @action(model='convolution_nn')
    def train_conv(self, models, sess, train_acc):
        """ Train-action for convolution_nn-model
        Args:
            model: do not supply this arg, always the output of convolution_nn-model defined above
            sess: tf-session in which learning variables are to be updated
        # """
        accuracy = models[1][-2]
        x, y_, _, train_step, training, keep_prob, _ = models[0]
        acc, _ = sess.run([accuracy, train_step], feed_dict={x: self.images.reshape(-1, 784), \
                                                             y_: self.labels, training: True, keep_prob: 0.5})
        train_acc.append(acc)
        # sess.run(train_step, feed_dict={x: self.images, y_: self.labels, training: True, keep_prob: 0.8})
        return self

    @action(model='convolution_nn')
    def update_stats(self, models, sess, accs):
        """ Append accuracy that is obtained by convolution_nn-model given weights stored in sess Tf-session
        Args:
            model: do not supply this arg, always the output of convolution_nn-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        _, _, accuracy, _ = models[1]
        x, y_, _, _, training, keep_prob, _ = models[0]

        accs.append(sess.run(accuracy, feed_dict={x: self.images.reshape(-1, 784), \
                                                  y_: self.labels, training: False, keep_prob: 1.}))
        return self

    @action(model='convolution_nn')
    def prediction(self, models, sess, dict_pred):
        """ Predict function
            Args:
                sess: tf Session
                dict_pred: dict params
            Return:
                self"""
        x, _, _, _, training, keep_prob, y = models[0]
        dict_pred['imgs'].append(self.images)
        dict_pred['predict'].append(sess.run(y, feed_dict={x: self.images.reshape(-1, 784), \
                                                           training: True, keep_prob: 0.5}))
        dict_pred['answer'].append(np.argmax(np.array(dict_pred['predict'][-1])\
            .reshape(-1, 10), axis=1) == np.argmax(self.labels, axis=1))
        return self

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def shift_flattened_pic(self, ind, max_margin=8):
        """ Apply random shift to a flattened pic
        Args:
            pic: ndarray of shape=(784) representing a pic to be flattened
        Return:
            flattened shifted pic """

        squared = self.images[ind].reshape(28, 28)
        padded = np.pad(squared, pad_width=[[max_margin, max_margin], [max_margin, max_margin]],
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        return padded[slicing].reshape(-1)

    def post_func(self, list_of_res):
        """ Concat outputs from shift_flattened_pic """
        if any_action_failed(list_of_res):
            raise Exception("Something bad happend")
        else:
            self.images = np.stack(list_of_res)
            return self

    def init_func(self):
        """ Create queue to parallel.
        Return:
            Array with """
        return np.arange(len(self.indices))
