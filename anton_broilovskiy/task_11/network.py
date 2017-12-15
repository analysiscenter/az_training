""" File with inception"""
import tensorflow as tf

from dataset.dataset.models.tf import Inception_v4
from dataset.dataset.models.tf.layers import conv_block

class FourHeadedInception(Inception_v4):
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
        color_importance = kwargs['color_importance']

        first_color = conv_block(inputs, units=2, name='first_col', **kwargs)
        first_number = conv_block(inputs, units=10, name='first_num', **kwargs)

        second_color = conv_block(inputs, units=2, name='second_col', **kwargs)
        second_number = conv_block(inputs, units=10, name='second_num', **kwargs)

        tf.nn.softmax(first_color, name='first_color')
        tf.nn.softmax(first_number, name='first_number')
        tf.nn.softmax(second_color, name='second_color')
        tf.nn.softmax(second_number, name='second_number')
        
        col1 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/first_color:0')
        col2 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/second_color:0')
        num1 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/first_number:0')
        num2 = tf.get_default_graph().get_tensor_by_name('FourHeadedInception/inputs/second_number:0')


        loss_c1 = color_importance * tf.losses.softmax_cross_entropy(col1, first_color)
        loss_c2 = color_importance * tf.losses.softmax_cross_entropy(col2, second_color)
        loss_n1 = tf.losses.softmax_cross_entropy(num1, first_number)
        loss_n2 = tf.losses.softmax_cross_entropy(num2, second_number)
        loss = loss_n2 + loss_n1 + loss_c2 + loss_c1

        tf.identity(loss_c1, name='loss_c1')
        tf.identity(loss_c2, name='loss_c2')
        tf.identity(loss_n1, name='loss_n1')
        tf.identity(loss_n2, name='loss_n2')

        tf.losses.add_loss(loss)

        return first_color
