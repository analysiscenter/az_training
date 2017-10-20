""" Common segmentation TFModel"""

import tensorflow as tf
from dataset.dataset.models.tf import TFModel

class SegmentationModel(TFModel):
    """Common segmentation TFModel

    Parameters
    ----------
    images_shape : tuple of ints

    b_norm : bool
        Use batch normalization. By default is True.

    n_classes : int
        number of classes to segmentate.

    dim : int
    	spacial dimension of input without the number of channels.
    """
    def _build(self, *args, **kwargs):
        """ build function for LinkNet """
        images_shape = list(self.get_from_config('images_shape'))
        n_classes = self.get_from_config('n_classes', 2)
        b_norm = self.get_from_config('b_norm', True)
        dim = self.get_from_config('dim', 2)

        input_ph = tf.placeholder(tf.float32, shape=[None] + images_shape, name='input_image')

        if len(images_shape) == dim:
            input_ph = tf.reshape(input_ph, [-1] + images_shape + [1])
            tf.placeholder(tf.float32, shape=[None] + images_shape + [n_classes], name='targets')
        elif len(images_shape) == dim + 1:
            tf.placeholder(tf.float32, shape=[None] + images_shape[:-1] + [n_classes], name='targets')
        else:
            raise ValueError('len(images_shape) must be equal to dim or dim + 1')

        model_output = self.build_model(dim, input_ph, n_classes, b_norm)
        predictions = tf.identity(model_output, name='predictions')
        tf.nn.softmax(predictions, name='prob_predictions')
