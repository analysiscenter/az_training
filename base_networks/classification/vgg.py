"""VGG"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf import TFModel

class VGGModel(TFModel):
    """VGG as TFModel

    Parameters
    ----------
    images_shape : tuple of ints

    vgg_arch : str or list of tuples
        see vgg()

    b_norm : bool
        Use batch normalization. By default is True.

    momentum : float
        Batch normalization momentum. By default is 0.9.

    n_classes : int.
    """

    def _build(self, inp1, inp2, *args, **kwargs):
        n_classes = self.num_channels('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        vgg_arch = self.get_from_config('vgg_arch', 'VGG16')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.9}
        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        net = self.fully_conv_block(dim, inp2['images'], b_norm, vgg_arch, **kwargs)

        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 100, name='fc1')
        layout = 'na' if b_norm else 'a'
        net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.layers.dense(net, 100, name='fc2')
        net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.layers.dense(net, n_classes, name='fc3')
        tf.identity(net, name='predictions')

    @staticmethod
    def block(dim, inp, depth, filters, last_layer, b_norm, name='vgg_block', **kwargs):
        """VGG block.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels

        inp : tf.Tensor

        depth : int
            the number of layers in VGG block

        filters : int

        last_layer : bool
            if True the last layer is convolution with 1x1 kernel else with 3x3

        b_norm : bool
            if True enable batch normalization

        training : tf.Tensor
            batch normalization training parameter

        vgg_arch : list of tuples
            see vgg()

        Return
        ------
        outp : tf.Tensor
        """
        net = inp
        with tf.variable_scope(name):  # pylint: disable=not-context-manager
            if last_layer:
                layout = 'cna' if b_norm else 'ca'
                layout = layout * (depth - 1)
                net = conv_block(dim, net, filters, 3, layout, **kwargs)
                layout = 'cnap' if b_norm else 'cap'
                net = conv_block(dim, net, filters, 1, layout, **kwargs)
            else:
                layout = 'cna' if b_norm else 'ca'
                layout = layout * depth + 'p'
                net = conv_block(dim, net, filters, 3, layout, **kwargs)
            net = tf.identity(net, name='output')
        return net

    @staticmethod
    def fully_conv_block(dim, inp, b_norm, vgg_arch, **kwargs):
        """VGG convolution part.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels

        inp : tf.Tensor

        b_norm : bool
            if True enable batch normalization

        training : tf.Tensor
            batch normalization training parameter

        vgg_arch : list of tuples
            see vgg()

        Return
        ------
        outp : tf.Tensor
        """

        arch = {'VGG16': [(2, 64, False),
                          (2, 128, False),
                          (3, 256, True),
                          (3, 512, True),
                          (3, 512, True)],
                'VGG19': [(2, 64, False),
                          (2, 128, False),
                          (4, 256, False),
                          (4, 512, False),
                          (4, 512, False)],
                'VGG7': [(2, 64, False),
                         (2, 128, False),
                         (3, 256, True)],
                'VGG6': [(2, 32, False),
                         (2, 64, False),
                         (2, 64, False)]}

        if isinstance(vgg_arch, list):
            pass
        elif isinstance(vgg_arch, str):
            vgg_arch = arch[vgg_arch]
        else:
            raise TypeError("vgg_arch must be str or list but {} was given.".format(type(vgg_arch)))
        net = inp
        with tf.variable_scope('fconv'):  # pylint: disable=not-context-manager
            for i, block_cfg in enumerate(vgg_arch):
                net = VGGModel.block(dim, net, *block_cfg, b_norm, 'block-' + str(i), **kwargs)
        return net

class VGG16Model(VGGModel):
    """VGG16 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG16'
        super()._build(inputs, *args, **kwargs)

class VGG19Model(VGGModel):
    """VGG19 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG19'
        super()._build(inputs, *args, **kwargs)

class VGG7Model(VGGModel):
    """VGG7 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG7'
        super()._build(inputs, *args, **kwargs)
