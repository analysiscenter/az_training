import sys
import tensorflow as tf

sys.path.append('../task_03')

from dataset.models.tf.layers import conv_block
from dataset.models.tf import TFModel
from vgg import VGGModel

MNIST_PER_IMAGE = 5

class RPNModel(TFModel):
    """RPNNet as TFModel"""
    def _build(self, *args, **kwargs):

        #n_classes = self.num_channels('masks')
        _, inp2 = self._make_inputs(['images'])
        data_format = self.data_format('images')
        dim = self.spatial_dim('images')
        b_norm = self.get_from_config('batch_norm', True)

        conv = {'data_format': data_format}
        batch_norm = {'momentum': 0.1}

        kwargs = {'conv': conv, 'batch_norm': batch_norm}
        
        inp = inp2['images']
        with tf.variable_scope('FRCNN'): # pylint: disable=not-context-manager
            net = VGGModel.fully_conv_block(dim, inp, b_norm, 'VGG7', **kwargs)
            net = conv_block(dim, net, 512, 3, 'ca', **kwargs)
            reg = conv_block(dim, net, 4*9, 1, 'ca', **kwargs)
            cls = conv_block(dim, net, 1*9, 1, 'c', **kwargs)

        tf.identity(net, name='output_map')
        output_map_shape = net.get_shape().as_list()[1:3]
        self.config['output_map_shape'] = output_map_shape

        n_anchors = output_map_shape[0] * output_map_shape[1] * 9

        reg = tf.reshape(reg, [-1, n_anchors, 4], name='RoI')
        cls = tf.reshape(cls, [-1, n_anchors], name='IoU')
        true_cls = tf.placeholder(tf.float32, shape = [None, n_anchors], name='proposal_targets')
        true_reg = tf.placeholder(tf.float32, shape = [None, n_anchors, 4], name='bbox_targets')
        
        loss = self.rpn_loss(reg, cls, true_reg, true_cls, n_anchors)
        loss = tf.identity(loss, name='loss')
        tf.losses.add_loss(loss)

    
    def rpn_loss(self, reg, cls, true_reg, true_cls, n_anchors):
        cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=true_cls, logits=cls) / MNIST_PER_IMAGE   
        cls_loss = tf.reduce_sum(cls_loss, axis=-1)
        cls_loss = tf.reduce_mean(cls_loss, name='cls_loss')

        sums = tf.reduce_sum((true_reg - reg) ** 2, axis=-1)
        reg_mask = tf.cast(true_cls, dtype=tf.float32)
        reg_mask = tf.reshape(reg_mask, shape=[-1, n_anchors])

        reg_loss = sums * true_cls
        reg_loss = tf.reduce_sum(reg_loss, axis=-1)
        reg_loss = tf.reduce_mean(reg_loss, name='reg_loss')
        
        loss = reg_loss * 10 + cls_loss
        return loss