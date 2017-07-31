""" Custom batch class for storing mnist batch and models
"""
import sys

import numpy as np
import os
import blosc
from dataset import Batch, action, model, inbatch_parallel
from layers import conv_mpool_activation, fc_layer
import tensorflow as tf

sys.path.append("/notebooks/Dari/az_trainig")

class MnistBatch(Batch):
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
        result_batch = np.stack(list_results)
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
        
        squared = pic.reshape(28, 28)
        padded = np.pad(squared, pad_width=[[max_margin, max_margin], [max_margin, max_margin]], 
                        mode='minimum')
        left_lower = np.random.randint(2 * max_margin, size=2)
        slicing = (slice(left_lower[0], left_lower[0] + 28),
                   slice(left_lower[1], left_lower[1] + 28))
        res = padded[slicing]
        res = res.reshape(-1)
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
        training = tf.placeholder(tf.bool, shape=[], name='mode')
        x = tf.placeholder(tf.float32, [None, 784])
        x_as_pics = tf.reshape(x, shape=[-1, 28, 28, 1])
        net = conv_mpool_bnorm_activation('conv_first', x_as_pics, n_channels=4, mpool=True, bnorm=True, training=training,
                                          kernel_conv=(7, 7), kernel_pool=(6, 6), stride_pool=(2, 2))
        net = conv_mpool_bnorm_activation('conv_second', net, n_channels=16, kernel_conv=(5, 5), bnorm=True, training=training,
                                          mpool=True, kernel_pool=(5, 5), stride_pool=(2, 2))
        net = conv_mpool_bnorm_activation('conv_third', net, n_channels=32, kernel_conv=(3, 3), bnorm=True, training=training,
                                          mpool=True, kernel_pool=(2, 2), stride_pool=(2, 2))
        net = tf.contrib.layers.flatten(net)
        net = fc_layer('fc_first', net, 128)
        net = tf.nn.relu(net)
        net = fc_layer('fc_second', net, 10)

        probs = tf.nn.softmax(logits=net)
        # placeholder for correct labels
        y_ = tf.placeholder(tf.float32, [None, 10])

        # loss
        loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')

        # optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = tf.train.AdamOptimizer().minimize(loss)

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

        return [[x, y_, loss, train_step, training], [labels, labels_hat, accuracy], [probs]]

    @action(model='convy')
    def predict(self, model, sess, pics, y_true, y_predict, probabilities):
        ''' Predict labels '''
        x, y_, _, _, _= model[0]
        labels, labels_hat, _ = model[1]
        probs = model[2][0]
        probabilities.append(sess.run(probs, feed_dict={x:self.images}))
        y_predict.append(sess.run(labels_hat, feed_dict={x:self.images}))
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
        x, y_, _, train_step, training = model[0]
        sess.run(train_step, feed_dict={x: self.images, y_: self.labels, training: True})
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
        x, y_, _, _, training = model[0]

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels, training: False}))
        return self