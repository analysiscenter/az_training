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

    def _build(self, inputs, *args, **kwargs):

        n_classes = self.num_channels('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        vgg_arch = self.get_from_config('vgg_arch', 'VGG16')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.99}
        kwargs = {'conv': conv, 'batch_norm': batch_norm}   

        net = self.fully_conv_block(dim, inputs['images'], b_norm, vgg_arch, **kwargs)
        
        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 100, name='fc1')
        layout = 'na' if b_norm else 'a'
        net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.layers.dense(net, 100, name='fc2')
        net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.layers.dense(net, n_classes, name = 'fc3')
        logit = tf.identity(net, name='predictions')
        self._create_outputs_from_logit(logit)

    def _create_outputs_from_logit(self, logit):
        """Create output for models which produce logit output
        Return
        ------
        predicted_prob, predicted_labels : tf.tensors
        """

        predicted_prob = tf.nn.softmax(logit, name='predicted_prob')
        max_axis = len(logit.get_shape())-1
        predicted_labels = tf.argmax(predicted_prob, axis=max_axis, name='predicted_labels')
        return predicted_prob, predicted_labels

    def block(self, dim, inp, depth, filters, last_layer, b_norm, name='vgg_block', **kwargs):
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

    def fully_conv_block(self, dim, inp, b_norm, vgg_arch, **kwargs):
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

        if isinstance(vgg_arch, list):
            pass
        else:
            raise TypeError("vgg_arch must be list or str.")

        net = inp
        with tf.variable_scope('fconv'):  # pylint: disable=not-context-manager
            for i, block in enumerate(vgg_arch):
                net = self.block(dim, net, *block, b_norm, 'block-' + str(i), **kwargs)
        return net

class VGG16Model(VGGModel):
    """VGG16 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = [(2, 64, False),
                                   (2, 128, False),
                                   (3, 256, True),
                                   (3, 512, True),
                                   (3, 512, True)]
        super()._build(inputs, *args, **kwargs)

class VGG19Model(VGGModel):
    """VGG19 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = [(2, 64, False),
                                   (2, 128, False),
                                   (4, 256, False),
                                   (4, 512, False),
                                   (4, 512, False)]
        super()._build(inputs, *args, **kwargs)

class VGG7Model(VGGModel):
    """VGG7 as TFModel"""
    def _build(self, inputs, *args, **kwargs):
        self.config['vgg_arch'] = [(2, 64, False),
                                   (2, 128, False),
                                   (3, 256, True)]
        super()._build(inputs, *args, **kwargs)
