"""VGG"""
import sys
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier

sys.path.append('..')

from dataset.dataset.models.tf import TFModel

B_NORM = True
MOMENTUM = 0.1

def VGG_conv_layer(inp, index, filters, kernel, b_norm, training, momentum):
    """VGG convolution layer and batch normalization"""
    net = tf.layers.conv2d(inp, filters, (kernel, kernel),
                           strides=(1, 1),
                           padding='SAME',
                           kernel_initializer=Xavier(),
                           name='conv'+str(index))
    if b_norm:
        net = tf.layers.batch_normalization(net,
                                            training=training,
                                            name='batch-norm'+str(index),
                                            momentum=momentum)
    return net


def VGG_conv_block(inp, name, depth, filters, last_layer, b_norm, training, momentum):
    """VGG convolution block"""
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        net = inp
        for i in range(depth-int(last_layer)):
            VGG_conv_layer(net, i+1, filters, 3, b_norm, training, momentum)
            net = tf.nn.relu(net)
        if last_layer:
            VGG_conv_layer(net, depth, filters, 1, b_norm, training, momentum)
            net = tf.nn.relu(net)
        net = tf.layers.max_pooling2d(net, (2, 2), strides=(2, 2), padding='SAME')
        return net

def VGG_fc_block(inp, output_dim, b_norm, training, momentum):
    """VGG fully connected block"""
    with tf.variable_scope('fc-block'): # pylint: disable=not-context-manager
        net = tf.layers.dense(inp, 4096, name='fc1')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1',
                                                momentum=momentum)   
            net = tf.nn.relu(net)     
        net = tf.layers.dense(net, 4096, name='fc2')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2',
                                                momentum=momentum)   
            net = tf.nn.relu(net)     
        net = tf.layers.dense(net, output_dim, name='fc3')
    return net


def VGG(inp, output_dim, b_norm=True, training=True, momentum=0.1):
    """VGG16 tf.layers.
    """
    vgg_architecture = [[2, 64, False], 
                        [2, 128, False],
                        [3, 256, True],
                        [3, 512, True],
                        [3, 512, True]]
    net = inp
    with tf.variable_scope('VGG'): # pylint: disable=not-context-manager
        for i, block in enumerate(vgg_architecture):
            net = VGG_conv_block(inp, 'conv-block-'+str(i), *block, b_norm, training, momentum)
        net = tf.contrib.layers.flatten(net)
        net = VGG_fc_block(net, output_dim, b_norm, training, momentum)
    return net


class VGGModel(TFModel):
    "VGG as TFModel"
    def _build(self, *args, **kwargs):
        "build for VGG"
        images_shape = [None] + list(self.get_from_config('images_shape'))

        b_norm = self.get_from_config('b_norm')
        momentum = self.get_from_config('momentum')

        x_ph = tf.placeholder(tf.float32, shape=images_shape, name='images')
        labels_ph = tf.placeholder(tf.uint8, shape=[None], name='labels')
        training = tf.placeholder(tf.bool, shape=[], name='training')

        targets = tf.one_hot(labels_ph, depth=10, name='targets')

        model_output = VGG(x_ph, 10, b_norm, training)
        predictions = tf.identity(model_output, name='predictions')

        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        predicted_labels = tf.argmax(y_pred_softmax, axis=1, name='predicted_labels')



