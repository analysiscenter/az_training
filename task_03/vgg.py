"""VGG"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf.layers import conv_block
from task_02.basemodels import NetworkModel


class VGGModel(NetworkModel):
    """VGG as TFModel

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    images_shape : tuple of ints

    vgg_arch : str or list of tuples
        see vgg()

    b_norm : bool
        Use batch normalization. By default is True.

    n_classes : int.
    """

    def _build(self, *args, **kwargs):
        """build function for VGG."""
        dim = self.get_from_config('dim', 2)
        n_classes = self.get_from_config('n_classes', 2)
        b_norm = self.get_from_config('b_norm')
        vgg_arch = self.get_from_config('vgg_arch', 'VGG16')

        inp = self.create_input()
        outp = vgg(dim, inp, n_classes, b_norm, 'predictions', self.is_training, vgg_arch)
        self.create_target('classification')
        self.create_output(outp)


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


VGG7 = [(2, 64, False),
        (2, 128, False),
        (3, 256, True)]


def vgg(dim, inp, n_classes, b_norm, output_name, training, vgg_arch):
    """VGG tf.layers.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    n_classes : int
        number of classes to segmentate

    b_norm : bool
        if True enable batch normalization

    output_name : string
        name of the output tensor

    training : tf.Tensor
        batch normalization training parameter

    vgg_arch : str or list of tuples
        Describes VGG architecture. If str, it should be 'VGG16' or 'VGG19'. If list of tuple,
        each tuple describes VGG block:
            tuple[0] : int
                the number of convolution layers in block,
            tuple[1] : int
                the number of filters,
            tuple[2] : bool:
                True if the last kernel is 1x1, False if 3x3.

    Return
    ------
    outp : tf.Tensor

    """
    with tf.variable_scope('VGG'):  # pylint: disable=not-context-manager
        net = vgg_convolution(dim, inp, b_norm, training, vgg_arch)
        net = tf.contrib.layers.flatten(net)
        net = vgg_fc_block(net, n_classes, b_norm, training)
    return tf.identity(net, output_name)


def vgg_fc_block(inp, n_classes, b_norm, training):
    """VGG fully connected block

    Parameters
    ----------
    inp : tf.Tensor

    n_classes : int
        number of output filters

    b_norm : bool
        if True enable batch normalization

    training : tf.Tensor
        batch normalization training parameter

    Return
    ------
    outp : tf.Tensor
    """

    with tf.variable_scope('VGG-fc'):  # pylint: disable=not-context-manager
        net = tf.layers.dense(inp, 100, name='fc1')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm1')
            net = tf.nn.relu(net)
        net = tf.layers.dense(net, 100, name='fc2')
        if b_norm:
            net = tf.layers.batch_normalization(net,
                                                training=training,
                                                name='batch-norm2')
            net = tf.nn.relu(net)
        outp = tf.layers.dense(net, n_classes, name='fc3')
    return outp


def vgg_convolution(dim, inp, b_norm, training, vgg_arch):
    """VGG convolution part.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    b_norm : bool
        if True enable batch normalization

    training : tf.Tensor
        batch normalization training parameter

    vgg_arch : str or list of tuples
        see vgg()

    Return
    ------
    outp : tf.Tensor
    """

    if isinstance(vgg_arch, str):
        if vgg_arch == 'VGG16':
            vgg_arch = VGG16
        elif vgg_arch == 'VGG19':
            vgg_arch = VGG19
        elif vgg_arch == 'VGG7':
            vgg_arch = VGG7
        else:
            raise NameError("{} is unknown NN.".format(vgg_arch))
    elif isinstance(vgg_arch, list):
        pass
    else:
        raise TypeError("vgg_arch must be list or str.")

    net = inp
    with tf.variable_scope('VGG-conv'):  # pylint: disable=not-context-manager
        for i, block in enumerate(vgg_arch):
            depth, filters, last_layer = block
            if last_layer:
                layout = ('c' + 'n' * b_norm + 'a') * (depth - 1)
                net = conv_block(dim, net, filters, 3, layout, 'conv-block-' + str(i), is_training=training)
                layout = 'c' + 'n' * b_norm + 'ap'
                net = conv_block(dim, net, filters, 1, layout, 'conv-block-1x1-' + str(i), is_training=training)
            else:
                layout = ('c' + 'n' * b_norm + 'a') * depth + 'p'
                net = conv_block(dim, net, filters, 3, layout, 'conv-block-' + str(i), is_training=training)
            net = tf.identity(net, name='conv-block-{}-output'.format(i))
    return net
