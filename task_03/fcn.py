"""Fully convoltional network"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv2d_block
from vgg import vgg_convolution

def fcn(input_ph, n_classes, fcn_arch, b_norm, training_ph):
    """FCN network.
    TODO: check post-VGG part 
    """
    vgg_arch = 'VGG16'
    with tf.variable_scope('FCN-conv'):  # pylint: disable=not-context-manager
        net = vgg_convolution(input_ph, vgg_arch, b_norm, training_ph)

        net = conv2d_block(net, 4096, 1, 'ca', 'conv-out-1', 
                           padding='VALID', 
                           is_training=training_ph)
        net = conv2d_block(net, 4096, 1, 'ca', 'conv-out-2', 
                           padding='VALID', 
                           is_training=training_ph)
        net = conv2d_block(net, n_classes, 1, 'ca', 'conv-out-3', 
                           padding='VALID', 
                           is_training=training_ph)

        net = tf.layers.conv2d_transpose(net, n_classes, kernel_size=64, strides=32, padding='SAME')
    return net


class FCNModel(TFModel):
    """FCN as TFModel
    """

    def _build(self, *args, **kwargs):
        """build function for VGG."""

        images_shape = list(self.get_from_config('images_shape'))
        n_classes = self.get_from_config('n_classes')
        b_norm = self.get_from_config('b_norm', True)
        momentum = self.get_from_config('momentum', 0.9)
        fcn_arch = self.get_from_config('fcn_arch', 'FCN32')

        input_ph = tf.placeholder(tf.float32, shape=[None] + images_shape, name='input_image')

        if len(images_shape) == 2:
            input_ph = tf.reshape(input_ph, [-1] + images_shape + [1])
            mask_ph = tf.placeholder(tf.float32, shape=[None] + images_shape + [n_classes], name='targets')
        elif len(images_shape) == 3:
            mask_ph = tf.placeholder(tf.float32, shape=[None] + images_shape[:-1] + [n_classes], name='targets')           
        else:
            raise ValueError('len(images_shape) must be 2 or 3')

        training_ph = tf.placeholder(tf.bool, shape=[], name='bn_mode')
        model_output = fcn(input_ph, n_classes, fcn_arch, b_norm, training_ph)

        predictions = tf.identity(model_output, name='predictions')
        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')
