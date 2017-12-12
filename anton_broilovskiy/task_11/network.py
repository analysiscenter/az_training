""" File with inception"""
import tensorflow as tf

from dataset.dataset.models.tf import Inception_v4
from dataset.dataset.models.tf.layers import conv_block

class FourHeadedInception(Inception_v4):
    """ inception with many outputs """

    def build_config(self, names=None):
        """ coufiguration function """
        config = super().build_config(names)
        config['head'] = (dict(layout='Vf'))
        return config

    @classmethod
    def head(cls, inputs, name='head', **kwargs):
        """ head function """
        _ = name
        first_color = conv_block(inputs, units=2, name='first_col', **kwargs)
        first_color = tf.nn.softmax(first_color, name='first_color')
        first_number = conv_block(inputs, units=10, name='first_num', **kwargs)
        first_number = tf.nn.softmax(first_number, name='first_number')

        second_color = conv_block(inputs, units=2, name='second_col', **kwargs)
        second_color = tf.nn.softmax(second_color, name='second_color')
        second_number = conv_block(inputs, units=10, name='second_num', **kwargs)
        second_number = tf.nn.softmax(second_number, name='second_number')

        col1 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/color:0')
        num1 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/first_number:0')
        num2 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/second_number:0')

        loss = tf.losses.softmax_cross_entropy(col1, first_color)
        loss += tf.losses.softmax_cross_entropy(1 - col1, second_color)
        loss += tf.losses.softmax_cross_entropy(num1, first_number)
        loss += tf.losses.softmax_cross_entropy(num2, second_number)

        tf.losses.add_loss(loss)

        return first_color
