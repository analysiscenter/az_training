"""Spliting into microbatches on np level"""
import pickle
from time import time
import tensorflow as tf
import numpy as np
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

    with open('../mnist/mnist_labels.pkl', 'rb') as file:
        labels = pickle.load(file)[:size]
    with open('../mnist/mnist_pics.pkl', 'rb') as file:
        images = pickle.load(file)[:size]
    return images, labels

def split_arrays(data, n_splits):
    "Split list of arrays into n_splits."
    return [np.array_split(res, n_splits) for res in data]


def train_on_batch(session, tensors, batch,
                   micro_batch_size, set_zero, accum_op, train_op):
    """
    Perform training on batch

    Parameters
    ----------
    session : tf.session

    tensors : list of tensors for input

    batch_x, batch_y: np.array current batch

    micro_batch_size: microbatch size

    set_zero, accum_op, train_op : model operations
    """
    x_ph, y_ph = tensors
    n_splits = np.ceil(len(batch[0]) / micro_batch_size)
    splitted = split_arrays(batch, n_splits)

    start = time()
    session.run(set_zero)
    for x, y in zip(*splitted):
        session.run(accum_op, feed_dict={x_ph: x, y_ph: y})
    session.run(train_op)
    return time() - start

def define_model():
    """
    Define classification model

    Returns
    -------
    session : tf.Session

    x_ph, y_ph : tf.Placeholder for inputs

    set_zero, accum_op, train_op, loss : : model operations
    """
    graph = tf.Graph()
    with graph.as_default(): # pylint: disable=not-context-manager
        x_ph = tf.placeholder(tf.float32, shape=[None, 784], name='image')
        y_ph = tf.placeholder(tf.float32, shape=[None, 10], name='label')

        batch_size = tf.cast(tf.shape(x_ph)[0], tf.float32)

        logits = conv_net_layers(x_ph, False)
        loss_sum = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_ph))
        loss = loss_sum / batch_size

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
    return session, x_ph, y_ph, set_zero, accum_op, train_op, loss
