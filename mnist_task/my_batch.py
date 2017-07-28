""" Custom batch class for storing mnist batch and models
"""
import numpy as np
import os
import blosc
from dataset import Batch, action, model
from layers import conv_mpool_activation, fc_layer
import tensorflow as tf

class MnistBatch(Batch):
    """ Mnist batch and models
    """
    def __init__(self, index, *args, **kwargs):
        """ Init func, inherited from base batch
        """
        super().__init__(index, *args, **kwargs)

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
                self.images = blosc.unpack_array(file.read())[self.indices]

            with open(os.path.join(src, 'mnist_labels.blk'), 'rb') as file:
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

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')
        train_step = tf.train.AdamOptimizer().minimize(loss)

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

        return [[x, y_, loss, train_step], [labels, labels_hat, accuracy]]

    @action(model='convy')
    def train_convy(self, model, sess):
        """ Train-action for convy-model

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session in which learning variables are to be updated
        """
        x, y_, _, train_step = model[0]
        sess.run(train_step, feed_dict={x: self.images, y_: self.labels})
        return self

    @action(model='convy')
    def update_stats(self, model, sess, accs):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        _, _, accuracy = model[1]
        x, y_, _, _ = model[0]

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels}))
        return self
