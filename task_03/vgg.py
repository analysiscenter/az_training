"""VGG"""
import sys
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier

sys.path.append('..')

from dataset.dataset.models.tf import TFModel


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


def VGG(inp, output_dim, vgg_arch, b_norm=True, training=True, momentum=0.1):
    """VGG16 tf.layers.
    """
    net = inp
    with tf.variable_scope('VGG'): # pylint: disable=not-context-manager
        for i, block in enumerate(vgg_arch):
            net = VGG_conv_block(inp, 'conv-block-'+str(i), *block, b_norm, training, momentum)
        net = tf.contrib.layers.flatten(net)
        net = VGG_fc_block(net, output_dim, b_norm, training, momentum)
    return net


VGG16 = [(2, 64, False),
         (2, 128, False),
         (3, 256, True),
         (3, 512, True),
         (3, 512, True)]


VGG19 = [(2, 64, False),
         (2, 128, False),
         (4, 256, False),
         (4, 512, False),
         (4, 512, False)]


class VGGModel(TFModel):
    """VGG as TFModel

    Parameters
    ----------
    images_shape : tuple of ints

    vgg_arch : str or list of tuples
        Describes VGG architecture. If str, it should be 'VGG16' or 'VGG19'. If list of tuple,
        each tuple describes VGG block:
            tuple[0] : int
                the number of convolution layers in block,
            tuple[1] : int
                the number of filters,
            tuple[2] : bool:
                True if the last kernel is 1x1, False if 3x3.

    b_norm : bool
        Use batch normalization. By default is True.

    momentum : float
        Batch normalization momentum. By default is 0.9.

    n_classes : int.
    """

    def _build(self, *args, **kwargs):
        """build function for VGG."""
        images_shape = [None] + list(self.get_from_config('images_shape'))

        vgg_arch = self.get_from_config('vgg_arch')

        if isinstance(vgg_arch, str):
            if vgg_arch == 'VGG16':
                vgg_arch = VGG16
            elif vgg_arch == 'VGG19':
                vgg_arch = VGG19
            else:
                raise NameError("{} is unknown NN.".format(vgg_arch))
        elif isinstance(vgg_arch, list):
            pass
        elif vgg_arch is None:
            vgg_arch = VGG16
        else:
            raise TypeError("vgg_arch must be list or str.")

        n_classes = self.get_from_config('n_classes')

        b_norm = self.get_from_config('b_norm')
        if b_norm is None:
            b_norm = True
        momentum = self.get_from_config('momentum')
        if momentum is None:
            momentum = 0.9

        x_ph = tf.placeholder(tf.float32, shape=images_shape, name='images')
        labels_ph = tf.placeholder(tf.uint8, shape=[None], name='labels')
        training_ph = tf.placeholder(tf.bool, shape=[], name='training')

        targets = tf.one_hot(labels_ph, depth=n_classes, name='targets')

        model_output = VGG(x_ph, n_classes, vgg_arch, b_norm, training_ph)
        predictions = tf.identity(model_output, name='predictions')

        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.argmax(y_pred_softmax, axis=1, name='predicted_labels')



