"""UNet"""
import sys
import tensorflow as tf

sys.path.append('..')

from dataset.dataset.models.tf import TFModel
from dataset.dataset.models.tf.layers import conv2d_block

def unet(input_ph, n_classes, b_norm, training_ph):
    """UNet network.
    """
    with tf.variable_scope(fcn_arch):  # pylint: disable=not-context-manager

        enc1 = conv2d_block(input_ph, 64, 3, 'cacap', 'encoder-1', is_training=training_ph, pool_size=2)
        enc2 = conv2d_block(enc1, 128, 3, 'cacap', 'encoder-2', is_training=training_ph, pool_size=2)
        enc3 = conv2d_block(enc2, 256, 3, 'cacap', 'encoder-3', is_training=training_ph, pool_size=2)
        enc4 = conv2d_block(enc3, 512, 3, 'cacap', 'encoder-4', is_training=training_ph, pool_size=2)

        middle = conv2d_block(enc4, 1024, 3, 'caca', 'middle', is_training=training_ph, pool_size=2)

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
