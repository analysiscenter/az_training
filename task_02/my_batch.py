""" Custom batch class for storing mnist batch and models
"""
import sys
import os

import blosc
import numpy as np
from layers import conv_mpool_activation, fc_layer
import tensorflow as tf

sys.path.append('..')
from dataset import Batch, action, model, inbatch_parallel, any_action_failed

class MnistBatch(Batch):
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
        """ Load mnist pics with specified indices

        Args:
            fmt: format of source. Can be either 'blosc' or 'ndarray'
            src: if fmt='blosc' then src is a path to dir with blosc-packed
                mnist images and labels are stored.
                if fmt='ndarray' - this is a tuple with arrays of images and labels

        Return:
            self
        """
        if fmt == 'blosc':
            # read blosc images, labels
            with open(os.path.join(src, 'mnist_pics.blk'), 'rb') as file:
                self.images = blosc.unpack_array(file.read())[self.indices]

            with open(os.path.join(src, 'mnist_labels.blk'), 'rb') as file:
                self.labels = blosc.unpack_array(file.read())[self.indices]
        elif fmt == 'ndarray':
            all_images, all_labels = src
            self.images = all_images[self.indices]
            self.labels = all_labels[self.indices]

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

    @action(model='convy')
    def prediction(self, models, sess, dict_pred):
        """ Predict """
        x, _, _, _, y = models[0]
        dict_pred['imgs'].append(self.images)
        dict_pred['predict'].append(sess.run(y, feed_dict={x: self.images}))
        dict_pred['answer'].append(np.argmax(np.array(dict_pred['predict'][-1])\
            .reshape(-1, 10), axis=1) == np.argmax(self.labels, axis=1))
        return self

    @model()
    def convy():
        """ Conv-net mnist classifier

        Return:
            list of 2 sublists:
            x: placeholder with data
            predict: model prediction
            loss: quality of model
            train: function - optimizer
            y: placeholder with answers to data

            labels: true labels
            labels_hat: predict of network
            accuracy: model accuracy
        """
        # build the net

        x = tf.placeholder(tf.float32, [None, 784])
        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])
        net = conv_mpool_activation('conv_first', x_as_pics, n_channels=4, mpool=True,
                                    kernel_conv=(7, 7), kernel_pool=(6, 6), stride_pool=(2, 2))
        net = conv_mpool_activation('conv_second', net, n_channels=16, kernel_conv=(5, 5),
                                    mpool=True, kernel_pool=(5, 5), stride_pool=(2, 2))
        net = conv_mpool_activation('conv_third', net, n_channels=32, kernel_conv=(3, 3),
                                    mpool=True, kernel_pool=(2, 2), stride_pool=(2, 2))

        net = tf.contrib.layers.flatten(net)
        net = fc_layer('fc_first', net, 128)
        net = tf.nn.relu(net)
        net = fc_layer('fc_second', net, 10)

        # placeholder for correct labels
        y_ = tf.placeholder(tf.float32, [None, 10])

        y = tf.nn.softmax(net)
        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')
        train_step = tf.train.AdamOptimizer().minimize(loss)

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

        return [[x, y_, loss, train_step, y], [labels, labels_hat, accuracy]]

    @action(model='convy')
    def train_convy(self, models, sess):
        """ Train-action for convy-model

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session in which learning variables are to be updated
        """
        x, y_, _, train_step, _ = models[0]
        sess.run(train_step, feed_dict={x: self.images, y_: self.labels})
        return self

    @action(model='convy')
    def update_stats(self, models, sess, accs):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        _, _, accuracy = models[1]
        x, y_, _, _, _ = models[0]

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels}))
        return self
