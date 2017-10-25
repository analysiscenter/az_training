""" LinkNet as TFModel """
import tensorflow as tf
from dataset.dataset.models.tf.layers import conv_block
from basemodels import NetworkModel

class UNetModel(NetworkModel):
    """LinkNet as TFModel"""
    def _build(self, *args, **kwargs):
        placeholders = self.create_placeholders('placeholders')
        dim = 
        n_classes = self.get_from_config('n_classes')
        b_norm = self.get_from_config('b_norm', True)

        conv = {'data_format': self.get_from_config('data_format', 'channels_last')}
        batch_norm = {'training': self.is_training, 'momentum': 0.1}

        logit = unet(len(placeholders[0].get_shape()) - 2, 
                     placeholders[0],
                     self.get_from_config('n_classes'),
                     self.get_from_config('b_norm', True), 
                     'predictions',
                     conv=conv, 
                     batch_norm=batch_norm)

        self.create_outputs_from_logit(logit)


def unet(dim, inp, n_classes, b_norm, output_name, **kwargs):
    """UNet tf.layers.

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
    with tf.variable_scope('UNet'): # pylint: disable=not-context-manager
        layout = ('c' + 'n' * b_norm + 'a') * 2 + 'p'

        net = inp
        encoder_outputs = []
        unet_filters = [64, 128, 256, 512]
        data_format = kwargs['conv']['data_format']
        if data_format == 'channels_last':
            axis = dim + 1
        else:
            axis = 1

        for i, filters in enumerate(unet_filters):
            net = conv_block(dim, net, filters, 3, layout, 'encoder-'+str(i),
                             pool_size=2, **kwargs)
            encoder_outputs.append(net)

        layout = ('c' + 'n' * b_norm + 'a') * 2

        net = conv_block(dim, net, 1024, 3, layout, 'middle-block', **kwargs)
        net = conv_block(dim, net, 512, 2, 't', 'middle-block-out', 2, **kwargs)


        for i, filters in enumerate(unet_filters[:0:-1]):
            net = tf.concat([encoder_outputs[-i-2], net], axis=axis)
            net = conv_block(dim, net, filters, 3, layout, 'decoder-block-'+str(i), **kwargs)
            net = conv_block(dim, net, filters // 2, 2, 't', 'decoder-block-out-'+str(i), 2, **kwargs)

        net = conv_block(dim, net, 64, 3, layout, 'decoder-block-3', **kwargs)
        net = conv_block(dim, net, n_classes, 1, layout, 'decoder-block-3-out', **kwargs)

    return tf.identity(net, output_name)
