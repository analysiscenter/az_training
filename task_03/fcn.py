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
    with tf.variable_scope(fcn_arch):  # pylint: disable=not-context-manager
        net = vgg_convolution(input_ph, vgg_arch, b_norm, training_ph)

        net = conv2d_block(net, 4096, 7, 'ca', 'conv-out-1',
                           padding='SAME',
                           is_training=training_ph)
        net = conv2d_block(net, 4096, 1, 'ca', 'conv-out-2',
                           padding='VALID',
                           is_training=training_ph)
        net = conv2d_block(net, n_classes, 1, 'ca', 'conv-out-3',
                           padding='VALID',
                           is_training=training_ph)
        conv7 = net
        pool4 = tf.get_default_graph().get_tensor_by_name(fcn_arch+"/VGG-conv/conv-block-3-output:0")
        pool3 = tf.get_default_graph().get_tensor_by_name(fcn_arch+"/VGG-conv/conv-block-2-output:0")
        if fcn_arch == 'FCN32':
            net = tf.layers.conv2d_transpose(conv7, n_classes, kernel_size=64, strides=32, padding='SAME')
        else:
            conv7 = tf.layers.conv2d_transpose(conv7, n_classes, kernel_size=1, strides=2, padding='SAME')
            pool4 = tf.layers.conv2d(pool4, n_classes, kernel_size=1, strides=1, padding='SAME')
            fcn16_sum = tf.add(conv7, pool4)
            if fcn_arch == 'FCN16':
                net = tf.layers.conv2d_transpose(fcn16_sum, n_classes, kernel_size=32, strides=16, padding='SAME')
            elif fcn_arch == 'FCN8':
                pool3 = tf.layers.conv2d(pool3, n_classes, kernel_size=1, strides=1, padding='SAME')
                fcn16_sum = tf.layers.conv2d_transpose(fcn16_sum, n_classes, kernel_size=1, strides=2, padding='SAME')
                fcn8_sum = tf.add(pool3, fcn16_sum)
                net = tf.layers.conv2d_transpose(fcn8_sum, n_classes, kernel_size=16, strides=8, padding='SAME')
            else:
                raise ValueError('Wrong value of fcn_arch')
    return net


class FCNModel(TFModel):
    """FCN as TFModel
    """

    def _build(self, *args, **kwargs):
        """build function for VGG."""

        images_shape = list(self.get_from_config('images_shape'))
        n_classes = self.get_from_config('n_classes')
        b_norm = self.get_from_config('b_norm', True)
        fcn_arch = self.get_from_config('fcn_arch', 'FCN32')

        input_ph = tf.placeholder(tf.float32, shape=[None] + images_shape, name='input_image')

        if len(images_shape) == 2:
            input_ph = tf.reshape(input_ph, [-1] + images_shape + [1])
            tf.placeholder(tf.float32, shape=[None] + images_shape + [n_classes], name='targets')
        elif len(images_shape) == 3:
            tf.placeholder(tf.float32, shape=[None] + images_shape[:-1] + [n_classes], name='targets')
        else:
            raise ValueError('len(images_shape) must be 2 or 3')

        training_ph = tf.placeholder(tf.bool, shape=[], name='bn_mode')
        model_output = fcn(input_ph, n_classes, fcn_arch, b_norm, training_ph)

        predictions = tf.identity(model_output, name='predictions')
        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')
