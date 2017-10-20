""" LinkNet as TFModel """
import tensorflow as tf
from dataset.dataset.models.tf.layers import conv_block
from segmentation import SegmentationModel

class LinkNetModel(SegmentationModel):
    """LinkNet as TFModel"""
    def build_model(self, dim, input_ph, n_classes, b_norm):
        """ build function for LinkNet """
        return linknet_layers(dim, input_ph, n_classes, b_norm, self.is_training)


def encoder_block(dim, inp, out_filters, name, b_norm, training):
    """LinkNet encoder block.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    out_filters : int
        number of output filters

    name : str
        tf.scope name

    b_norm : bool
        if True enable batch normalization

    training : tf.Tensor
        batch normalization training parameter

    Return
    ------
    outp : tf.Tensor
    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        layout = 'c' + 'n' * b_norm + 'a'
        net = conv_block(dim, inp, out_filters, 3, layout, 'encoder_conv_1', 2, is_training=training)
        net = conv_block(dim, net, out_filters, 3, layout, 'encoder_conv_2', is_training=training)
        shortcut = conv_block(dim, inp, out_filters, 1, layout, 'encoder_short_1', 2, is_training=training)
        encoder_add = tf.add(net, shortcut, 'encoder_add_1')

        net = conv_block(dim, encoder_add, out_filters, 3, 2*layout, 'encoder_conv_3', is_training=training)
        outp = tf.add(net, encoder_add, 'encoder_add_2')
    return outp


def decoder_block(dim, inp, out_filters, name, b_norm, training):
    """LinkNet decoder block.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    out_filters : int
        number of output filters

    name : str
        tf.scope name

    b_norm : bool
        if True enable batch normalization

    training : tf.Tensor
        batch normalization training parameter

    Return
    ------
    outp : tf.Tensor

    """
    with tf.variable_scope(name): # pylint: disable=not-context-manager
        layout = 'c' + 'n' * b_norm + 'a'
        layout_transpose = 't' + 'n' * b_norm + 'a'

        n_filters = inp.get_shape()[-1].value // 4

        net = conv_block(dim, inp, n_filters, 1, layout, 'decoder_conv_1', is_training=training)
        net = conv_block(dim, net, n_filters, 3, layout_transpose, 'decoder_conv_2', 2, is_training=training)
        outp = conv_block(dim, net, out_filters, 1, layout, 'decoder_conv_3', is_training=training)
        return outp


def linknet_layers(dim, inp, n_classes, b_norm, training):
    """LinkNet tf.layers.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    n_classes : int
        number of classes to segmentate.

    b_norm : bool
        if True enable batch normalization

    training : tf.Tensor
        batch normalization training parameter

    Return
    ------
    outp : tf.Tensor

    """
    with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
        layout = 'cp' + 'n' * b_norm + 'a'

        net = conv_block(dim, inp, 64, 7, layout, 'decoder_conv_3', 2, is_training=training, pool_size=3)

        encoder_output = []

        for i, n_filters in enumerate([64, 128, 256, 512]):
            net = encoder_block(dim, net, n_filters, 'encoder-'+str(i), b_norm, training)
            encoder_output.append(net)

        for i, n_filters in enumerate([256, 128, 64]):
            net = decoder_block(dim, net, n_filters, 'decoder-'+str(i), b_norm, training)
            net = tf.add(net, encoder_output[-2-i])

        net = decoder_block(dim, net, 64, 'decoder-3', b_norm, training)

        layout = 'c' + 'n' * b_norm + 'a'
        layout_transpose = 't' + 'n' * b_norm + 'a'

        net = conv_block(dim, net, 32, 3, layout_transpose, 'output_conv_1', 2, is_training=training)
        net = conv_block(dim, net, 32, 3, layout, 'output_conv_2', is_training=training)
        net = conv_block(dim, net, n_classes, 2, 't', 'output_conv_3', 2, is_training=training)

        return net
