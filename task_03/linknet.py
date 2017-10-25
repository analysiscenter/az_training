""" LinkNet as TFModel """
import tensorflow as tf
from dataset.dataset.models.tf.layers import conv_block
from basemodels import NetworkModel

class LinkNetModel(NetworkModel):
    """LinkNet as TFModel"""
    def _build(self, *args, **kwargs):
        """ build function for LinkNet """
        """
        input_config = self.get_from_config('input')
        input_shape = input_config['input_shape']

        output_config = self.get_from_config('output')

        dim = len(input_shape) - 1
        n_classes = output_config.get('n_outputs', 2)
        b_norm = self.get_from_config('b_norm', True)

        inp = self.create_input()
        linknet(dim, inp, n_classes, b_norm, 'predictions', self.is_training)
        self.create_target('segmentation')
        """

        placeholders = self.create_placeholders('placeholders')
        dim = len(inp.get_shape()) - 2
        n_classes = self.get_from_config('n_classes')
        b_norm = self.get_from_config('b_norm', True)

        conv = {'data_format': self.get_from_config('data_format', 'channels_last')}
        batch_norm = {'training': self.is_training, 'momentum': 0.99}

        logit = linknet(dim, placeholders[0], n_classes, b_norm, 'predictions',
                        conv=conv, batch_norm=batch_norm)

        self.create_outputs_from_logit(logit)

def encoder_block(dim, inp, out_filters, name, b_norm, **kwargs):
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
        net = conv_block(dim, inp, out_filters, 3, layout, 'encoder_conv_1', 2, **kwargs)
        net = conv_block(dim, net, out_filters, 3, layout, 'encoder_conv_2', **kwargs)
        shortcut = conv_block(dim, inp, out_filters, 1, layout, 'encoder_short_1', 2, **kwargs)
        encoder_add = tf.add(net, shortcut, 'encoder_add_1')

        net = conv_block(dim, encoder_add, out_filters, 3, 2*layout, 'encoder_conv_3', **kwargs)
        outp = tf.add(net, encoder_add, 'encoder_add_2')
    return outp


def decoder_block(dim, inp, out_filters, name, b_norm, **kwargs):
    """LinkNet decoder block.

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    inp : tf.Tensor

    n_classes : int
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

        net = conv_block(dim, inp, n_filters, 1, layout, 'decoder_conv_1', **kwargs)
        net = conv_block(dim, net, n_filters, 3, layout_transpose, 'decoder_conv_2', 2, **kwargs)
        outp = conv_block(dim, net, out_filters, 1, layout, 'decoder_conv_3', **kwargs)
        return outp


def linknet(dim, inp, n_classes, b_norm, output_name, **kwargs):
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

    output_name : string
        name of the output tensor

    training : tf.Tensor
        batch normalization training parameter

    Return
    ------
    outp : tf.Tensor

    """
    with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
        layout = 'cp' + 'n' * b_norm + 'a'

        net = conv_block(dim, inp, 64, 7, layout, 'decoder_conv_3', 2, pool_size=3, **kwargs)

        encoder_output = []

        for i, n_filters in enumerate([64, 128, 256, 512]):
            net = encoder_block(dim, net, n_filters, 'encoder-'+str(i), b_norm, **kwargs)
            encoder_output.append(net)

        for i, n_filters in enumerate([256, 128, 64]):
            net = decoder_block(dim, net, n_filters, 'decoder-'+str(i), b_norm, **kwargs)
            net = tf.add(net, encoder_output[-2-i])

        net = decoder_block(dim, net, 64, 'decoder-3', b_norm, **kwargs)

        layout = 'c' + 'n' * b_norm + 'a'
        layout_transpose = 't' + 'n' * b_norm + 'a'

        net = conv_block(dim, net, 32, 3, layout_transpose, 'output_conv_1', 2, **kwargs)
        net = conv_block(dim, net, 32, 3, layout, 'output_conv_2', **kwargs)
        net = conv_block(dim, net, n_classes, 2, 't', 'output_conv_3', 2, **kwargs)

    return tf.identity(net, output_name)
