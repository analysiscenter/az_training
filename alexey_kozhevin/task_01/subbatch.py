"""Split into microbatches on tf level"""
import sys
import pickle
from time import time
import tensorflow as tf

sys.path.append('..')

from dataset import Batch, model, action
from networks import conv_net_layers

def subbatch_static_model(sess, scope, n_subbatches, layers):
    """Compute gradients by subbatches
    Parameters
    ----------
        sess: tf session
        scope: tf scope
        n_subbatches: int, the number of subbatches
        layers: function which describes tf network
    Return
    ------
        x: tf placeholder for network input
        y: tf placeholder for network
        step: tf operation, training step
        total_loss: loss on the batch
        accuracy: accuracy on the batch
    """
    with sess.graph.as_default():
        with tf.variable_scope(scope): # pylint: disable=not-context-manager

            x = tf.placeholder(tf.float32, shape=[None, 784], name='image')
            y = tf.placeholder(tf.float32, shape=[None, 10], name='label')
            opt = tf.train.GradientDescentOptimizer(0.01)

            total_y_hat = layers(x, reuse=False)

            gradients_as_tensor = subbatch_gradients(x, y, n_subbatches, layers, opt)
            total_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=total_y_hat, labels=y))
            step = opt.apply_gradients(gradients_as_tensor)

            y_pred = tf.nn.softmax(total_y_hat)
            labels_hat = tf.cast(tf.argmax(y_pred, axis=1), tf.float32, name='labels_hat')
            labels = tf.cast(tf.argmax(y, axis=1), tf.float32, name='labels')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(labels_hat, labels), tf.float32), name='accuracy')

            sess.run(tf.global_variables_initializer())

    return x, y, step, total_loss, accuracy, sess


def split_tensors(tensors, number):
    """Split list of tensors"""
    return [tf.split(tensor, number) for tensor in tensors]


def subbatch_gradients(x, y, n_subbatches, models, opt):
    """Compute gradients on microbatches of batch."""
    subbatches = split_tensors([x, y], n_subbatches)
    gradients = []
    for subbatch_x, subbatch_y in zip(*subbatches):
        y_hat = models(subbatch_x, reuse=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=subbatch_y))
        grad_var = opt.compute_gradients(loss)
        grad, var = list(zip(*grad_var))
        gradients.append(grad)
    gradients = [tf.add_n(subbatch_grads) / n_subbatches for subbatch_grads in zip(*gradients)]
    gradients_as_tensor = list(zip(gradients, var))
    return gradients_as_tensor


class Subbatch(Batch):
    """Batch class for microbatches."""

    def __init__(self, index, *args, **kwargs):
        """Init function."""
        super().__init__(index, *args, **kwargs)
        self.x = None
        self.y = None

    @model(mode='static')
    def neural_net(pipeline): #pylint: disable=no-self-argument
        """Neural network model."""
        n_subbatches = pipeline.get_variable("NUM_SUBBATCHES")
        scope = "static_cn"
        sess = pipeline.get_variable("session")
        return subbatch_static_model(sess, scope, n_subbatches, conv_net_layers)

    @action(model='neural_net')
    def train(self, models, iter_time, acc):
        """Train iteration on batch."""
        x, y, step, _, accuracy, sess = models

        start = time()
        sess.run(step, feed_dict={x: self.x, y: self.y})
        stop = time()

        acc.append(sess.run(accuracy, feed_dict={x: self.x, y: self.y}))
        iter_time.append(stop-start)

        return self

    @action
    def load_images(self):
        """Load images and labels from files."""
        with open('../mnist/mnist_labels.pkl', 'rb') as file:
            self.y = pickle.load(file)[self.indices]
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.x = pickle.load(file)[self.indices]
        return self
