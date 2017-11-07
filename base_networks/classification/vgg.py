"""VGG"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.models.tf.layers import conv_block
from dataset.models.tf import TFModel

_ARCH = {'VGG16': [(2, 0, 64),
                  (2, 0, 128),
                  (2, 1, 256),
                  (2, 1, 512),
                  (2, 1, 512)],
        'VGG19': [(2, 0, 64),
                  (2, 0, 128),
                  (4, 0, 256),
                  (4, 0, 512),
                  (4, 0, 512)],
        'VGG7': [(2, 0, 64),
                 (2, 0, 128),
                 (2, 1, 256)],
        'VGG6': [(2, 0, 32),
                 (2, 0, 64),
                 (2, 0, 64)]}

class VGG(TFModel):
    """VGG as TFModel

    **Configuration**
    -----------------
    inputs : dict
        dict with keys 'images' and 'labels' (see :meth:`._make_inputs`)
    b_norm : bool
        if True enable batch normalization layers

    arch : str or list of tuple
        if str, it 
        tuple[0] : int
            depth of the block
        tuple[1] : int
            the number of filters in each layer of the block
        tuple[2] : bool
            if True the last layer is convolution with 1x1 kernel else with 3x3.
    """

    def _build(self):
        """Builds a VGG model."""
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        arch = self.get_from_config('arch', 'VGG16')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.9,
                      'training': self.is_training}
        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        net = self.body(dim, inputs['images'], b_norm, arch, **kwargs)

        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 100, name='fc1')
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, 100, name='fc2')
        net = tf.nn.relu(net)        
        net = tf.layers.dense(net, n_classes, name='fc3')

        logits = tf.identity(net, name='predictions')
        pred_proba = tf.nn.softmax(logits, name='predicted_prob')
        pred_labels = tf.argmax(pred_proba, axis=-1, name='predicted_labels')
        true_labels = tf.argmax(inputs['labels'], axis=-1, name='true_labels')
        equality = tf.equal(pred_labels, true_labels)
        equality = tf.cast(equality, dtype=tf.float32)
        tf.reduce_mean(equality, name='accuracy')

    @staticmethod
    def block(dim, inp, depth_3, depth_1, filters, b_norm, name='block', **kwargs):
        """VGG block.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels

        inp : tf.Tensor

        depth_3 : int
            the number of convolution layers with 3x3 kernel

        depth_1 : int
            the number of convolution layers with 1x1 kernel

        filters : int

        b_norm : bool
            if True enable batch normalization

        Return
        ------
        outp : tf.Tensor
        """
        net = inp
        with tf.variable_scope(name):  # pylint: disable=not-context-manager
            layout = 'cna' if b_norm else 'ca'
            layout = layout * (depth_3 + depth_1) + 'p'
            kernels = [3] * depth_3 + [1] * depth_1
            net = conv_block(dim, net, filters, kernels, layout, **kwargs)
            net = tf.identity(net, name='output')
        return net

    @staticmethod
    def body(dim, inp, b_norm, arch, *args, **kwargs):
        """VGG convolution part.

        Parameters
        ----------
        dim : int
            spatial dimension of input without the number of channels

        inp : tf.Tensor

        b_norm : bool
            if True enable batch normalization

        arch : str or list of tuples

        Return
        ------
        outp : tf.Tensor
        """

        if isinstance(arch, list):
            pass
        elif isinstance(arch, str):
            arch = _ARCH[arch]
        else:
            raise TypeError("arch must be str or list but {} was given.".format(type(arch)))
        net = inp
        with tf.variable_scope('body'):  # pylint: disable=not-context-manager
            for i, block_cfg in enumerate(arch):
                net = VGG.block(dim, net, *block_cfg, b_norm, 'block-' + str(i), **kwargs)
        return net

class VGG16(VGG):
    '''
    Builds a VGG16 model.
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG16'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inp, b_norm, *args, **kwargs):
        """VGG16 convolution part.
        """
        return VGG.body(dim, inp, b_norm, 'VGG16', *args, **kwargs)


class VGG19Model(VGG):
    '''
    Builds a VGG19 model.
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG19'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inp, b_norm, *args, **kwargs):
        """VGG16 convolution part.
        """
        return VGG.body(dim, inp, b_norm, 'VGG19', *args, **kwargs)

class VGG7Model(VGG):
    '''
    Builds a VGG7 model.
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG7'
        super()._build(*args, **kwargs)

    @staticmethod
    def body(dim, inp, b_norm, *args, **kwargs):
        """VGG16 convolution part.
        """
        return VGG.body(dim, inp, b_norm, 'VGG7', *args, **kwargs)
