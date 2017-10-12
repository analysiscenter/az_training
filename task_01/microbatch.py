"""Spliting into microbatches on np level"""

import sys
import os
import pickle
from time import time
import tensorflow as tf
import numpy as np
import psutil

sys.path.append('../')
from networks import conv_net_layers

def load(size):
    """
    Load MNIST data

    Parameters
    ----------
    size : int
    size of the loaded data

    Returns
    -------
    images : np.array
    MNIST images

    output : np.array
    MNIST one-hot labels
    """

    with open('./mnist/mnist_labels.pkl', 'rb') as file:
        labels = pickle.load(file)[:size]
    with open('./mnist/mnist_pics.pkl', 'rb') as file:
        images = pickle.load(file)[:size]
    return images, labels


def train_on_batch(session, x_ph, y_ph, batch_x, batch_y,
                   micro_batch_size, set_zero, accum_op, train_op):
    """
    Perform training on batch

    Parameters
    ----------
    session : tf.session

    x_ph, y_ph : tf.Placeholder for input

    batch_x, batch_y: np.array current batch

    micro_batch_size: microbatch size

    set_zero, accum_op, train_op : model operations
    """
    n_splits = np.ceil(len(batch_x) / micro_batch_size)
    x_splitted = np.array_split(batch_x, n_splits)
    y_splitted = np.array_split(batch_y, n_splits)

    pid = os.getpid()
    start = time()
    mem_before = psutil.Process(pid).memory_percent()
    session.run(set_zero)
    for x, y in zip(x_splitted, y_splitted):
        session.run(accum_op, feed_dict={x_ph: x, y_ph: y})
    session.run(train_op)
    stop = time()
    mem_after = psutil.Process(pid).memory_percent()
    time_it = stop - start
    memory = mem_after - mem_before
    return time_it, memory

def define_model():
    """
    Define classification model

    Returns
    -------
    session : tf.Session

    x_ph, y_ph : tf.Placeholder for inputs

    set_zero, accum_op, train_op, loss, accuracy : : model operations
    """
    graph = tf.Graph()
    with graph.as_default():
        x_ph = tf.placeholder(tf.float32, shape=[None, 784], name='image')
        y_ph = tf.placeholder(tf.float32, shape=[None, 10], name='label')

        batch_size = tf.cast(tf.shape(x_ph)[0], tf.float32)

        logits = conv_net_layers(x_ph, False)
        y_hat = tf.nn.softmax(logits)
        loss_sum = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_ph))
        loss = loss_sum / batch_size

        labels_hat = tf.cast(tf.argmax(y_hat, axis=1), tf.float32, name='labels_hat')
        labels = tf.cast(tf.argmax(y_ph, axis=1), tf.float32, name='labels')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

        opt = tf.train.AdamOptimizer()
        train_vars = tf.trainable_variables()

        grad_accum = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in train_vars]
        batch_size_accum = tf.Variable(tf.zeros(shape=(), dtype=tf.float32), trainable=False)
        grad_set_zero = [var.assign(tf.zeros_like(var)) for var in grad_accum]
        batch_size_set_zero = batch_size_accum.assign(tf.zeros(shape=(), dtype=tf.float32))
        set_zero = [grad_set_zero, batch_size_set_zero]

        grad = opt.compute_gradients(loss_sum, train_vars)
        accum_grad = [grad_accum[i].assign_add(g[0]) for i, g in enumerate(grad)]
        accum_batch_size = batch_size_accum.assign_add(batch_size)
        accum_op = [accum_grad, accum_batch_size]

        train_op = opt.apply_gradients([(grad_accum[i] / batch_size_accum, g[1]) for i, g in enumerate(grad)])

        session = tf.Session()
        session.run(tf.global_variables_initializer())
    return session, x_ph, y_ph, set_zero, accum_op, train_op, loss, accuracy
