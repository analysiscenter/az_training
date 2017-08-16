import sys 
import os

import blosc
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier
  
sys.path.append('..')
from dataset import action, model, Batch

def conv_block(input_tensor, kernel, filters, name, strides=(2, 2)):
    
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='convo' + name, activation=tf.nn.relu,\
                         kernel_initializer=xavier())
    
    x = tf.layers.conv2d(x, filters2, kernel, name='convt' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())
    
    x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name,\
                         kernel_initializer=xavier())
    
    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, name='short' + name, \
                         kernel_initializer=xavier())
    x = tf.concat([x, shortcut], axis=1)
    x = tf.nn.relu(x)
    return x

def identity_block(input_tensor, kernel, filters, name):
    
    filters1, filters2, filters3 = filters
    x = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='convo' + name, activation=tf.nn.relu,\
                         kernel_initializer=xavier())
    
    x = tf.layers.conv2d(x, filters2, kernel, name='convt' + name, activation=tf.nn.relu, padding='SAME',\
                         kernel_initializer=xavier())
    
    x = tf.layers.conv2d(x, filters3, (1, 1), name='convtr' + name,\
                         kernel_initializer=xavier())
    
  
    x = tf.concat([x, input_tensor], axis=1)
    x = tf.nn.relu(x)
    return x

def create_train(opt, src, global_step, loss, it):
    def learning_rate(last, src):
        
        last = int(last)
        bound = list(np.linspace(0, last, len(range(2, last+1)), dtype=np.int32))
        values = [0.5 * 0.1/last * (1 + np.cos(np.pi * i / last)) for i in range(2, last+1)]
        var = [i for i in tf.trainable_variables() if src in i.name or 'dense' in i.name]

        return bound, values, var

    b, v, var = learning_rate(it, src)
    learning_rate = tf.train.piecewise_constant(global_step, b, v)

    return opt(learning_rate, 0.9, use_nesterov=True).minimize(loss, global_step, var)

class ResBatch(Batch):
    """ """
	# def __init__(self, index, *args, **kwargs):
 #        """ 
 #        """
 #        super().__init__(index, *args, **kwargs)
 #        self.x = None
 #        self.y = None

    @property
    def components(self):
        """ Define componentis. """
        return 'x', 'y'  

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
    def Freeznet():

        x = tf.placeholder(tf.float32, shape=[None, 784])

        x_f_to_tens = tf.reshape(x, shape=[-1, 28, 28, 1])

        net = tf.layers.conv2d(x_f_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                               kernel_initializer=xavier(), name='1')
        net = tf.layers.max_pooling2d(net, (2, 2),(2, 2))

        net = conv_block(net, 3, [32, 32, 128], name='2', strides=(1, 1))
        net = identity_block(net, 3, [32, 32, 128], name='3')
        net = identity_block(net, 3, [32, 32, 128], name='4')

        net = conv_block(net, 3, [64, 64, 256], name='5', strides=(1, 1))
        net = identity_block(net, 3, [64, 64, 256], name='6')
        net = identity_block(net, 3, [64, 64, 256], name='7')

        net = tf.layers.average_pooling2d(net, (7, 7), strides=(1, 1))
        net = tf.contrib.layers.flatten(net)

        with tf.variable_scope('dense'):
            net = tf.layers.dense(net, 10)

        prob = tf.nn.softmax(net)
        y = tf.placeholder(tf.float32, [None, 10])
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y), name='loss')

        it = 400
        train = []
        global_steps = []
        for i in range(1, 8):
            global_steps.append(tf.Variable(0, trainable=False))
            train.append(create_train(tf.train.MomentumOptimizer, str(i), \
                                      global_steps[-1], loss, it*(i/10 + 0.5)))


        lables_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='lables_hat')
        lables = tf.cast(tf.argmax(y, axis=1), tf.float32, name='lables')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(lables_hat, lables), tf.float32, name='accuracy'))
        return [[x, y, loss, train, prob], [accuracy]]

    @action(model='Freeznet')
    def train_freez(self, models, sess, train_loss, train_acc):

        accuracy = models[1][-1]
        x, y, loss, train, _ = models[0]

        acc, loss, _ = sess.run([accuracy, loss, train], feed_dict={x:self.images.reshape(-1, 784), y:self.labels.reshape(-1, 10)})

        train_loss.append(loss)
        train_acc.append(acc)

        return self

    @model()
    def Resnet():
        x1 = tf.placeholder(tf.float32, shape=[None, 784])

        x1_to_tens = tf.reshape(x1, shape=[-1, 28, 28, 1])

        net1 = tf.layers.conv2d(x1_to_tens, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                               kernel_initializer=xavier(), name='11')
        net1 = tf.layers.max_pooling2d(net1, (2, 2),(2, 2))

        net1 = conv_block(net1, 3, [32, 32, 128], name='22', strides=(1, 1))

        net1 = identity_block(net1, 3, [32, 32, 128], name='33')
        net1 = identity_block(net1, 3, [32, 32, 128], name='43')

        net1 = conv_block(net1, 3, [64, 64, 256], name='53', strides=(1, 1))
        net1 = identity_block(net1, 3, [64, 64, 256], name='63')
        net1 = identity_block(net1, 3, [64, 64, 256], name='73')

        net1 = tf.layers.average_pooling2d(net1, (7, 7), strides=(1, 1))
        net1 = tf.contrib.layers.flatten(net1)

        with tf.variable_scope('dense3'):
            net1 = tf.layers.dense(net1, 10)

        prob1 = tf.nn.softmax(net1)
        y1 = tf.placeholder(tf.float32, [None, 10])
        loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net1, labels=y1), name='loss3')

        train1 = tf.train.MomentumOptimizer(0.01, 0.9, use_nesterov=True).minimize(loss1)
        lables_hat1 = tf.cast(tf.argmax(net1, axis=1), tf.float32, name='lables_3at')
        lables1 = tf.cast(tf.argmax(y1, axis=1), tf.float32, name='labl3es')

        accuracy1 = tf.reduce_mean(tf.cast(tf.equal(lables_hat1, lables1), tf.float32, name='a3ccuracy'))

        return [[x1, y1, loss1, train1, prob1], [accuracy1]]
	
    @action(model='Resnet')
    def train_res(self, models, sess,train_loss,  train_acc):

        accuracy = models[1][-1]
        x, y, loss, train, _ = models[0]

        acc, loss, _ = sess.run([accuracy, loss, train], feed_dict={x:self.images.reshape(-1, 784), y:self.labels.reshape(-1, 10)})

        train_loss.append(loss)
        train_acc.append(acc)

        return self
