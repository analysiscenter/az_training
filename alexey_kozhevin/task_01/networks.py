"""Convolutional network architecture."""
import tensorflow as tf

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
                           padding='SAME', name='layer1', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(4, 4), strides=(2, 2))
    net = tf.layers.batch_normalization(net, name='batch-norm1', reuse=reuse)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=32, kernel_size=(5, 5), strides=(1, 1),
                           padding='SAME', name='layer2', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(3, 3), strides=(2, 2))
    net = tf.layers.batch_normalization(net, name='batch-norm2', reuse=reuse)
    net = tf.nn.relu(net)
    net = tf.layers.conv2d(inputs=net, filters=64, kernel_size=(3, 3), strides=(1, 1), \
                           padding='SAME', name='layer3', reuse=reuse,
                           kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    net = tf.layers.max_pooling2d(net, pool_size=(2, 2), strides=(1, 1), padding='SAME')
    net = tf.layers.batch_normalization(net, name='batch-norm3', reuse=reuse)
    net = tf.layers.dropout(net)
    net = tf.contrib.layers.flatten(net)
    net = tf.layers.dense(net, 128, name='layer4', reuse=reuse)
    net = tf.layers.dropout(net)
    y_hat = tf.layers.dense(net, 10, name='layer5', reuse=reuse)
    return y_hat
