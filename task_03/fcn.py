"""Fully convoltional network"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf.layers import conv_block
from vgg import vgg_convolution
from task_02.basemodels import NetworkModel


class FCNModel(NetworkModel):
    """FCN as TFModel

    Parameters
    ----------
    dim : int
        spacial dimension of input without the number of channels

    images_shape : tuple of ints

    fcn_arch : str or list of tuples
        see fcn()

    b_norm : bool
        Use batch normalization. By default is True.

    n_classes : int.
    """

    def _build(self, *args, **kwargs):
        """build function for VGG."""

        dim = self.get_from_config('dim', 2)
        n_classes = self.get_from_config('n_classes', 2)
        b_norm = self.get_from_config('b_norm')
        fcn_arch = self.get_from_config('fcn_arch', 'FCN32')

        inp = self.create_input()
        outp = fcn(dim, inp, n_classes, b_norm, 'predictions', self.is_training, fcn_arch)
        self.create_target('segmentation')
        self.create_output(outp)


def fcn(dim, inp, n_classes, b_norm, output_name, training, fcn_arch):
    """FCN network.

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

    fcn_arch : str
        Describes FCN architecture. It should be 'FCN8', 'FCN16' or 'FCN32'

    Return
    ------
    outp : tf.Tensor

    """
    with tf.variable_scope(fcn_arch):  # pylint: disable=not-context-manager
        net = vgg_convolution(dim, inp, b_norm, training, 'VGG16')
        net = conv_block(dim, net, 4096, 7, 'ca', 'conv-out-1', is_training=training)
        net = conv_block(dim, net, 4096, 1, 'ca', 'conv-out-2', padding='VALID', is_training=training)
        net = conv_block(dim, net, n_classes, 1, 'ca', 'conv-out-3', padding='VALID', is_training=training)
        conv7 = net
        pool4 = tf.get_default_graph().get_tensor_by_name(fcn_arch+"/VGG-conv/conv-block-3-output:0")
        pool3 = tf.get_default_graph().get_tensor_by_name(fcn_arch+"/VGG-conv/conv-block-2-output:0")
        if fcn_arch == 'FCN32':
            net = conv_block(dim, conv7, n_classes, 64, 't', 32)
        else:
            conv7 = conv_block(dim, conv7, n_classes, 1, 't', 2)
            pool4 = conv_block(dim, pool4, n_classes, 1, 'c', 1)
            fcn16_sum = tf.add(conv7, pool4)
            if fcn_arch == 'FCN16':
                net = conv_block(dim, fcn16_sum, n_classes, 32, 't', 16)
            elif fcn_arch == 'FCN8':
                pool3 = conv_block(dim, pool3, n_classes, 1, 'c')
                fcn16_sum = conv_block(dim, fcn16_sum, n_classes, 1, 't', 2)
                fcn8_sum = tf.add(pool3, fcn16_sum)
                net = conv_block(dim, fcn8_sum, n_classes, 16, 't', 8)
            else:
                raise ValueError('Wrong value of fcn_arch')
    return tf.identity(net, output_name)
