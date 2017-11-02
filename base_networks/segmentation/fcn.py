"""Fully convoltional network"""
import sys
import tensorflow as tf

sys.path.append('../..')
sys.path.append('../classification')

from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf import TFModel
from vgg import VGGModel

class FCNModel(TFModel):
    """FCN as TFModel
    """

    def _build(self):
        """build function for VGG."""
        names = ['images', 'masks']
        _, inputs = self._make_inputs(names)

        n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)
        fcn_arch = self.get_from_config('fcn_arch', 'FCN32')

        conv = {'data_format': data_format,
                'dilation_rate': self.get_from_config('dilation_rate', 1)}
        batch_norm = {'momentum': 0.1,
                      'training': self.is_training}

        layers_dicts = {'conv': conv, 'batch_norm': batch_norm}

        net = VGGModel.fully_conv_block(dim, inputs['images'], b_norm, 'VGG16', **layers_dicts)
        net = conv_block(dim, net, 100, 7, 'ca', 'conv-out-1', **layers_dicts)
        net = conv_block(dim, net, 100, 1, 'ca', 'conv-out-2', padding='VALID', **layers_dicts)
        net = conv_block(dim, net, n_classes, 1, 'ca', 'conv-out-3', padding='VALID', **layers_dicts)
        conv7 = net
        pool4 = tf.get_default_graph().get_tensor_by_name("fconv/block-3/output:0")
        pool3 = tf.get_default_graph().get_tensor_by_name("fconv/block-2/output:0")
        if fcn_arch == 'FCN32':
            net = conv_block(dim, conv7, n_classes, 64, 't', 'output', 32, **layers_dicts)
        else:
            conv7 = conv_block(dim, conv7, n_classes, 1, 't', 'conv7', 2, **layers_dicts)
            pool4 = conv_block(dim, pool4, n_classes, 1, 'c', 'pool4', 1, **layers_dicts)
            fcn16_sum = tf.add(conv7, pool4)
            if fcn_arch == 'FCN16':
                net = conv_block(dim, fcn16_sum, n_classes, 32, 't', 'output', 16, **layers_dicts)
            elif fcn_arch == 'FCN8':
                pool3 = conv_block(dim, pool3, n_classes, 1, 'pool3', 'c')
                fcn16_sum = conv_block(dim, fcn16_sum, n_classes, 1, 't', 'fcn16_sum', 2, **layers_dicts)
                fcn8_sum = tf.add(pool3, fcn16_sum)
                net = conv_block(dim, fcn8_sum, n_classes, 16, 't', 'output', 8, **layers_dicts)
            else:
                raise ValueError('Wrong value of fcn_arch')
        tf.nn.softmax(tf.identity(net, 'predictions'), name='predicted_prob')
