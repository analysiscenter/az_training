import sys
import tensorflow as tf

sys.path.append('../task_03')

from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf import TFModel
from vgg import VGGModel

class FRCNNModel(TFModel):
    """LinkNet as TFModel"""
    def _build(self, inp1, inp2, *args, **kwargs):

        #n_classes = self.num_channels('masks')
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}
        
        inp = inp2['images']
        with tf.variable_scope('FRCNN'): # pylint: disable=not-context-manager
            net = VGGModel.fully_conv_block(dim, inp, b_norm, 'VGG6', **kwargs)
            net = conv_block(dim, net, 512, 3, 'ca', **kwargs)
            reg = conv_block(dim, net, 4*9, 1, 'ca', **kwargs)
            cls = conv_block(dim, net, 2*9, 1, 'ca', **kwargs)
        net = tf.concat([reg,cls], axis=-1, name='predictions')