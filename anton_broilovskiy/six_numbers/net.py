"""network"""
import tensorflow as tf

from dataset.dataset.models.tf import Inception_v4
from dataset.dataset.models.tf.layers import conv_block

class SixHeadedInception(Inception_v4):
    """ inception with many outputs """

    def build_config(self, names=None):
        """ build configuration function """
        config = super().build_config(names)
        config['head'].update(dict(layout='Vf'))
        config['head'].pop('units')
        return config

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ head function """
        _ = name

        output = conv_block(inputs, units=50, name='out', **kwargs)
        tf.nn.softmax(output, name='output')
        # number_1 = conv_block(inputs, units=10, name='numbers_1', **kwargs)
        # number_2 = conv_block(inputs, units=10, name='numbers_2', **kwargs)
        # number_3 = conv_block(inputs, units=10, name='numbers_3', **kwargs)
        # number_4 = conv_block(inputs, units=10, name='numbers_4', **kwargs)
        # number_5 = conv_block(inputs, units=10, name='numbers_5', **kwargs)

        first = tf.get_default_graph().get_tensor_by_name('SixHeadedInception/inputs/first:0')
        second = tf.get_default_graph().get_tensor_by_name('SixHeadedInception/inputs/second:0')
        third = tf.get_default_graph().get_tensor_by_name('SixHeadedInception/inputs/third:0')
        fourth = tf.get_default_graph().get_tensor_by_name('SixHeadedInception/inputs/fourth:0')
        fifth = tf.get_default_graph().get_tensor_by_name('SixHeadedInception/inputs/fifth:0')


        loss_1 = tf.losses.softmax_cross_entropy(first, output[:, 0:10])
        loss_2 = tf.losses.softmax_cross_entropy(second, output[:, 10:20])
        loss_3 = tf.losses.softmax_cross_entropy(third, output[:, 20:30])
        loss_4 = tf.losses.softmax_cross_entropy(fourth, output[:, 30:40])
        loss_5 = tf.losses.softmax_cross_entropy(fifth, output[:, 40:50])
        loss = loss_1 + loss_2 + loss_3 + loss_4 + loss_5

        tf.losses.add_loss(loss)

        return output
