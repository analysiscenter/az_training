import sys
import pickle
from time import time
import tensorflow as tf

sys.path.append('../')
from dataset import Dataset, Batch, model, DatasetIndex, action


def dense_net_layers(inp, reuse):
    """TensorFlow dense network
    input:
        inp: neural network input
        reuse: If true reuse layers
    output:
        logit output
    """
    net = tf.layers.dense(inp, 20, name='first_layer', reuse=reuse)
    y_hat = tf.layers.dense(net, 10, name='second_layer', reuse=reuse)
    return y_hat

def conv_net_layers(inp, reuse):
    """TensorFlow convolutional network
    input:
        inp: neural network input
        reuse: If true reuse layers
    output:
        logit output
    """
    x_as_pics = tf.reshape(inp, shape=[-1, 28, 28, 1])
    net = tf.layers.conv2d(inputs=x_as_pics, filters=16, kernel_size=(7, 7), strides=(2, 2), 
                           padding='SAME', name = 'layer1', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=(2, 2))
    net = tf.layers.batch_normalization(net, name='batch-norm1', reuse=reuse)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(5, 5), strides=(1, 1), 
                           padding='SAME', name = 'layer2', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2))
    net = tf.layers.batch_normalization(net, name='batch-norm2', reuse=reuse)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), strides=(1, 1), \
                           padding='SAME', name = 'layer3', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(1, 1), padding='SAME')
    net = tf.layers.batch_normalization(net, name='batch-norm3', reuse=reuse)
    #net = tf.layers.dropout(net)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 128, name = 'layer4', reuse=reuse)
    #net = tf.layers.dropout(net)
    y_hat = tf.layers.dense(net, 10, name = 'layer5', reuse=reuse)
    return y_hat

def subbatch_static_model(sess, scope, n_subbatches, layers):
    """Compute gradients by subbatches
    input:
        sess: tf session
        scope: tf scope
        n_subbatches: int, the number of subbatches
        layers: function which describes tf network
    output:
        x: tf placeholder for network input
        y: tf placeholder for network
        step: tf operation, training step
        total_loss: loss on the batch
        accuracy: accuracy on the batch
    """
    with sess.graph.as_default():
        with tf.variable_scope(scope):

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

    return x, y, step, total_loss, accuracy


def split_tensors(tensors, number):
    return [tf.split(tensor, number) for tensor in tensors]


def subbatch_gradients(x, y, n_subbatches, model, opt):
    subbatches = split_tensors([x, y], n_subbatches)
    gradients = []
    for subbatch_x, subbatch_y in zip(*subbatches):
        y_hat = model(subbatch_x, reuse=True)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_hat, labels=subbatch_y))
        grad_var = opt.compute_gradients(loss)
        grad, var = list(zip(*grad_var))
        gradients.append(grad)
    gradients = [tf.add_n(subbatch_grads) / n_subbatches for subbatch_grads in zip(*gradients)]
    gradients_as_tensor = list(zip(gradients, var))
    return gradients_as_tensor


class Subbatch(Batch):

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.x = None
        self.y = None

    @model(mode='static')
    def neural_net(pipeline):
        n_subbatches = pipeline.get_variable("NUM_SUBBATCHES")
        scope = "static_cn"
        sess = pipeline.get_variable("session")
        return subbatch_static_model(sess, scope, n_subbatches, conv_net_layers)

    @action(model='neural_net')
    def train(self, models, sess, iter_time, acc):
        x, y, step, total_loss, accuracy = models

        start = time()
        sess.run(step, feed_dict={x: self.x, y: self.y})
        stop = time()

        acc.append(sess.run(accuracy, feed_dict={x: self.x, y: self.y}))
        iter_time.append(stop-start)

        return self

    @action
    def load(self):
        with open('./mnist/mnist_labels.pkl', 'rb') as file:
            self.y = pickle.load(file)[self.indices]
        with open('./mnist/mnist_pics.pkl', 'rb') as file:
            self.x = pickle.load(file)[self.indices]
        return self
