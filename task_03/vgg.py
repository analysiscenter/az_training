"""VGG"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv2d_block


def vgg_fc_block(inp, output_dim, b_norm, training, momentum):
    """VGG fully connected block"""
    with tf.variable_scope('fc-block'):  # pylint: disable=not-context-manager
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


def vgg(inp, output_dim, vgg_arch, b_norm, training, momentum):
    """VGG16 tf.layers.
    """
    net = inp
    with tf.variable_scope('VGG'):  # pylint: disable=not-context-manager
        for i, block in enumerate(vgg_arch):
            depth, filters, last_layer = block
            if last_layer:
                layout = 'cna' * (depth - 1)
                net = conv2d_block(net, filters, 3, layout, 'conv-block-' + str(i), is_training=training)
                layout = 'cnap'
                net = conv2d_block(net, filters, 1, layout, 'conv-block-last-' + str(i), is_training=training)
            else:
                layout = 'cna' * depth + 'p'
                net = conv2d_block(net, filters, 3, layout, 'conv-block-' + str(i), is_training=training)
        net = tf.contrib.layers.flatten(net)
        net = vgg_fc_block(net, output_dim, b_norm, training, momentum)
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

        vgg_arch = self.get_from_config('vgg_arch', 'VGG16')

        if isinstance(vgg_arch, str):
            if vgg_arch == 'VGG16':
                vgg_arch = VGG16
            elif vgg_arch == 'VGG19':
                vgg_arch = VGG19
            else:
                raise NameError("{} is unknown NN.".format(vgg_arch))
        elif isinstance(vgg_arch, list):
            pass
        else:
            raise TypeError("vgg_arch must be list or str.")

        n_classes = self.get_from_config('n_classes')

        b_norm = self.get_from_config('b_norm', True)
        momentum = self.get_from_config('momentum', 0.9)

        x_ph = tf.placeholder(tf.float32, shape=images_shape, name='images')
        labels_ph = tf.placeholder(tf.uint8, shape=[None], name='labels')
        training_ph = tf.placeholder(tf.bool, shape=[], name='training')

        tf.one_hot(labels_ph, depth=n_classes, name='targets')

        model_output = vgg(x_ph, n_classes, vgg_arch, b_norm, training_ph, momentum)
        predictions = tf.identity(model_output, name='predictions')

        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.argmax(y_pred_softmax, axis=1, name='predicted_labels')
