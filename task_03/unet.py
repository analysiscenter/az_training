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
