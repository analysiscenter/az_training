""" Custom batch class for storing mnist batch and ensemble models
"""
import sys

import numpy as np
import os
import blosc

import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict

from dataset import Batch, action, model, inbatch_parallel
from dataset.dataset.image import ImagesBatch


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
        x = tf.placeholder(tf.float32, [None, 28, 28], name='x')
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
        net = tf.nn.dropout(net, keep_prob)


        net = tf.layers.dense(net, 128, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))
        #         net = tf.layers.dense(net, 128, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        net = tf.nn.relu(net)

        net = tf.layers.dense(net, 10, kernel_initializer=tf.truncated_normal_initializer(0.0, 1))
        #         net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False))

        probs = tf.nn.softmax(logits=net, name='softmax_output')

        # placeholder for correct labels
        y_ = tf.placeholder(tf.float32, [None, 10], name='y_')

        # loss
        #         loss = tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss')
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y_, name='loss'))

        global_step = tf.Variable(0, trainable=False)
        starter_learning_rate = 0.0001

        learning_rate = tf.placeholder(tf.float32, shape=[], name='lr')
        # learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
        #                                           100, 0.85, staircase=True)

        # optimization step
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_step = (tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step))

        # stats
        labels_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')


        predicts = tf.placeholder(tf.float32, [None, 10], name='predicts')
        test_acc = tf.reduce_mean(tf.cast(tf.equal(predicts, labels), tf.float32), name='accuracy')

        # print_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=softmax, labels=y_, name='print_loss'))

        return [[x, y_, loss, train_step, training, keep_prob], [labels, labels_hat, accuracy], [probs],
                [learning_rate, global_step], [predicts, test_acc]]

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

    @action(model='convy')
    def train_convy(self, model, sess, alpha, period, n_iterations, accs, loss_history):
        """ Train-action for convy-model

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session in which learning variables are to be updated
        """        
        x, y_, loss, train_step, training, keep_prob = model[0]
        learning_rate, global_step = model[3]
        
        _, _, accuracy = model[1]

        alpha = tf.cast(alpha, tf.float32)
        
        global_step = sess.run(global_step)
    
        period = tf.cast(period, tf.float32)
        n_iterations = tf.cast(n_iterations, tf.float32)
        n_cycles = n_iterations // period
        
        cyclic_learning_rate = alpha / tf.cast(2.0, tf.float32) * (tf.cos(tf.cast(np.pi, tf.float32) * ((tf.cast(global_step, tf.float32) - 1) % (period)) / (period)) + 1)          

        cyclic_learning_rate = sess.run(cyclic_learning_rate)
        # print ('HHHHHHHHEEEEEEEEEEEY', cyclic_learning_rate)
        sess.run(train_step, feed_dict={x: self.images, y_: self.labels, training: True, keep_prob: 0.7, learning_rate:cyclic_learning_rate})        
        
        period = sess.run(period)
        
        if (global_step) % period == 0:
            if global_step == 0:
                pass
            else:
                print ('hey')
                saver = tf.train.Saver(name=str(global_step))
                address = 'trained' + '+' + str(global_step) + '/model'
                saver.save(sess, address, global_step=global_step)
        
        loss_history.append(sess.run(loss, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))

        return self

    
    @model(mode='dynamic')
    def ensemble(self):
        ''' Classifier which averages prediction from m models loaded from the disk 
            Args:
            __
            Returns:
        '''

        # HARDCODING FIX LATER:
        # period = config['period']
        # n_iterations = config['n_iterations']
        # period = config['period']
        # n_iterations = config['n_iterations']
        period = 300
        n_iterations = 1501
        print('i do hardcoding of n_iterations and period')

        n_cycles = n_iterations // period
        results = []
        ensemble_data = defaultdict(list)

        for i in range(1, n_cycles + 1):
            # dir = 'trained' + '+' + str(global_step) + '/model'

            # address = 'trained/model' + '-' + str(i*period) + '.meta'

            folder = 'trained+' + str(i*period) + '/'
            address = folder + 'model' + '-' + str(i*period) + '.meta'
            print ('currently loading', address)

            grapphy_2 = tf.Graph()
            with grapphy_2.as_default():
                new_sess = tf.Session()

                new_saver = tf.train.import_meta_graph(address)
                new_saver.restore(new_sess, tf.train.latest_checkpoint(folder))

                training = grapphy_2.get_tensor_by_name('mode:0')
                x = grapphy_2.get_tensor_by_name('x:0')
                softmax_output = grapphy_2.get_tensor_by_name('softmax_output:0')

                # res = new_sess.run(softmax_output, feed_dict={x:imgs, training:False})

                ensemble_data['sess'].append(new_sess)
                ensemble_data['graph'].append(grapphy_2)
                ensemble_data['training'].append(training)
                ensemble_data['x'].append(x)
                ensemble_data['softmax'].append(softmax_output)

                # results.append(res)
                
        return ensemble_data
            

    
    @action(model='ensemble')
    def update_stats_ensemble(self, model, config, accs, loss_history):
        ensemble_data = model
        results = []
        for i, training in enumerate(ensemble_data['training']):
            if i == 0:
                continue
            # if i == 1:
            #     continue

            sess = ensemble_data['sess'][i]
            # graph = ensemble_data['grapph'][i]
            # training = ensemble_data['training'][i]
            x = ensemble_data['x'][i]
            softmax_output = ensemble_data['softmax'][i]
            results.append(sess.run(softmax_output, feed_dict={x:self.images, training:False}))
        
        # print(len(results))
        # print('shape', results[0].shape)
        avg_logits = np.mean(np.array(results), axis=0)
        # print('pr', avg_logits)
        # print('pr.shape', avg_logits.shape)

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

            accs.append(small_sess.run(accuracy, feed_dict={y_:self.labels, logits:avg_logits}))
            loss_history.append(small_sess.run(loss, feed_dict={y_:self.labels, logits:avg_logits}))
        return self
    
    
    @action(model='convy')
    def update_stats(self, model, sess, accs, loss_history):
        """ Append accuracy that is obtained by convy-model given weights stored in sess Tf-session

        Args:
            model: do not supply this arg, always the output of convy-model defined above
            sess: tf-session with trained (to some extent) weights
            accs: list with accuracies
        """
        
        _, _, accuracy = model[1]
        x, y_, loss, _, training, keep_prob = model[0]
        loss_history.append(sess.run(loss, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))

        accs.append(sess.run(accuracy, feed_dict={x: self.images, y_: self.labels, training: False, keep_prob: 1.0}))
        return self


def draw_stats(stats, title):
    plt.title(title)
    plt.plot(stats)
    plt.xlabel('iteration')
    plt.ylabel('accuracy')
    plt.show()