""" LinkNet as TFModel """
import tensorflow as tf
from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf import TFModel

class LinkNetModel(TFModel):
    """LinkNet as TFModel"""
    def _build(self, inp1, inp2, *args, **kwargs):

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        inp = inp2['images']
        with tf.variable_scope('LinkNet'): # pylint: disable=not-context-manager
            layout = 'cpna' if b_norm else 'cpa'

            net = conv_block(dim, inp, 64, 7, layout, 'input_conv', 2, pool_size=3)

            encoder_output = []

            for i, n_filters in enumerate([64, 128, 256, 512]):
                net = self.encoder_block(dim, net, n_filters, 'encoder-'+str(i), b_norm, **kwargs)
                encoder_output.append(net)

            for i, n_filters in enumerate([256, 128, 64]):
                net = self.decoder_block(dim, net, n_filters, 'decoder-'+str(i), b_norm, **kwargs)
                net = tf.add(net, encoder_output[-2-i])

            net = self.decoder_block(dim, net, 64, 'decoder-3', b_norm, **kwargs)

            layout = 'cna' if b_norm else 'ca'
            layout_transpose = 'tna' if b_norm else 'ta'

            net = conv_block(dim, net, 32, 3, layout_transpose, 'output_conv_1', 2, **kwargs)
            net = conv_block(dim, net, 32, 3, layout, 'output_conv_2', **kwargs)
            net = conv_block(dim, net, n_classes, 2, 't', 'output_conv_3', 2, **kwargs)

        tf.identity(net, 'predictions')

    @staticmethod
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
            layout = 'cna' if b_norm else 'ca'
            net = conv_block(dim, inp, out_filters, 3, layout, 'encoder_conv_1', 2, **kwargs)
            net = conv_block(dim, net, out_filters, 3, layout, 'encoder_conv_2', **kwargs)
            shortcut = conv_block(dim, inp, out_filters, 1, layout, 'encoder_short_1', 2, **kwargs)
            encoder_add = tf.add(net, shortcut, 'encoder_add_1')

            net = conv_block(dim, encoder_add, out_filters, 3, 2*layout, 'encoder_conv_3', **kwargs)
            outp = tf.add(net, encoder_add, 'encoder_add_2')
        return outp

    @staticmethod
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
            layout = 'cna' if b_norm else 'ca'
            layout_transpose = 'tna' if b_norm else 'ta'

            n_filters = inp.get_shape()[-1].value // 4

            net = conv_block(dim, inp, n_filters, 1, layout, 'decoder_conv_1', **kwargs)
            net = conv_block(dim, net, n_filters, 3, layout_transpose, 'decoder_conv_2', 2, **kwargs)
            outp = conv_block(dim, net, out_filters, 1, layout, 'decoder_conv_3', **kwargs)
            return outp
