"""VGG"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.models.tf.layers import conv_block
from dataset.models.tf import TFModel

class VGGModel(TFModel):
    """VGG as TFModel
    """

    def _build(self):
        '''
        Builds a VGG model.
        Parameters are taken from the config
        ----------
        input_config: a dict containing

            b_norm : bool
                if True enable batch normalization layers

            vgg_arch : str or list of tuple
                if str, it 
                tuple[0] : int
                    depth of the block
                tuple[1] : int
                    the number of filters in each layer of the block
                tuple[2] : bool
                    if True the last layer is convolution with 1x1 kernel else with 3x3.
        Returns
        -------
        '''
        names = ['images', 'labels']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('labels')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        vgg_arch = self.get_from_config('vgg_arch', 'VGG16')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.9,
                      'training': self.is_training}
        kwargs = {'conv': conv, 'batch_norm': batch_norm}

        net = self.fully_conv_block(dim, inputs['images'], b_norm, vgg_arch, **kwargs)

        net = tf.contrib.layers.flatten(net)
        net = tf.layers.dense(net, 100, name='fc1')
        layout = 'na' if b_norm else 'a'
        #net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.nn.relu(net)
        net = tf.layers.dense(net, 100, name='fc2')
        net = tf.nn.relu(net)        
        #net = conv_block(dim, net, None, None, layout, **kwargs)
        net = tf.layers.dense(net, n_classes, name='fc3')

        logits = tf.identity(net, name='predictions')
        pred_proba = tf.nn.softmax(logits, name='predicted_prob')
        pred_labels = tf.argmax(pred_proba, axis=-1, name='predicted_labels')
        true_labels = tf.argmax(inputs['labels'], axis=-1, name='true_labels')
        equality = tf.equal(pred_labels, true_labels)
        equality = tf.cast(equality, dtype=tf.float32)
        tf.reduce_mean(equality, name='accuracy')

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

        Return
        ------
        outp : tf.Tensor
        """
        net = inp
        with tf.variable_scope(name):  # pylint: disable=not-context-manager
            if last_layer:
                layout = 'cna' if b_norm else 'ca'
                layout = layout * (depth - 1)
                net = conv_block(dim, net, filters, 3, layout, name='0', **kwargs)
                layout = 'cnap' if b_norm else 'cap'
                net = conv_block(dim, net, filters, 1, layout, name='1', **kwargs)
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

        vgg_arch : str or list of tuples
            see _build doc-string

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
    '''
    Builds a VGG16 model.
    Parameters are taken from the config
    ----------
    input_config: a dict containing
        b_norm : bool
            if True enable batch normalization layers
    Returns
    -------
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG16'
        super()._build(*args, **kwargs)

class VGG19Model(VGGModel):
    '''
    Builds a VGG19 model.
    Parameters are taken from the config
    ----------
    input_config: a dict containing
        b_norm : bool
            if True enable batch normalization layers
    Returns
    -------
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG19'
        super()._build(*args, **kwargs)

class VGG7Model(VGGModel):
    '''
    Builds a VGG7 model.
    Parameters are taken from the config
    ----------
    input_config: a dict containing
        b_norm : bool
            if True enable batch normalization layers
    Returns
    -------
    '''
    def _build(self, *args, **kwargs):
        self.config['vgg_arch'] = 'VGG7'
        super()._build(*args, **kwargs)
