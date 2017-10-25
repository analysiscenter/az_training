""" TFModel subclass with methods to create placeholders for input, output and targets."""

import tensorflow as tf
from dataset.dataset.models.tf import TFModel

class NetworkModel(TFModel):
    """Common segmentation TFModel

    Parameters
    ----------
    images_shape : tuple of ints

    n_classes : int
        number of classes to segmentate.

    dim : int
        spacial dimension of input without the number of channels.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_shape = None

    def create_placeholders(self):
        """Create tf.placeholders from dict config['placeholders'].

        Return
        ------
        out : list of tf.Tensors

        Placeholder will be created for each key in dictionary self.get_from_config(name) where key
        is placeholder name and corresponding values are config dictionaries with the following keys : values:

        type : str or tf.DType (by default 'float32')
            type of the tensor. If str, it must be name of the tf type (int32, float64, etc).
        shape : tuple, list or None (default),
            the shape of placeholder which include the number of channels and doesn't include batch size.
            Feeded tensor will be reshaped into [-1, shape].
        data_format : str (by default 'channels_last')
            one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs.
        transform : str or None
            if transform='oh' one-hot encoding will be produced with n_classes. The new axis is created
            at the last dimension if data_format is channels_last or at the first after batch size otherwise.
        n_classes : int
        name : str
            name of the transformed and reshaped tensor.
        """

        placeholders_config = self.get_from_config('placeholders')

        res = dict()

        for name, config in placeholders_config.items():
            dtype = self._create_type(config)
            postname = config.get('name', name+'-post')

            placeholder = tf.placeholder(dtype, name=name)

            placeholder = self._transform(placeholder, config)
            placeholder = self._reshape(placeholder, config)

            res[name] = tf.identity(placeholder, name=postname)
        return res

    def _transform(self, placeholder, config):
        transform = config.get('transform', None)
        if transform is not None:
            transform = transform.split('-')
            transform_dict = {'oh': self._one_hot_transform}
            for transform_name in transform:
                placeholder = transform_dict[transform_name](placeholder, config)
        return placeholder

    def _reshape(self, placeholder, config):
        shape = config.get('shape', None)
        if shape is not None:
            placeholder = tf.reshape(placeholder, [-1] + list(shape))
        return placeholder

    def create_outputs_from_logit(self, logit):
        """Create output for models which produce logit output
        Return
        ------
        predicted_prob, predicted_labels : tf.tensors
        """

        predicted_prob = tf.nn.softmax(logit, name='predicted_prob')
        max_axis = len(logit.get_shape())-1
        predicted_labels = tf.argmax(predicted_prob, axis=max_axis, name='predicted_labels')
        return predicted_prob, predicted_labels

    def _create_type(self, config):
        type_in_config = config.get('type', 'float32')
        if isinstance(type_in_config, str):
            type_in_config = getattr(tf, type_in_config)
        elif isinstance(type_in_config, tf.DType):
            pass
        else:
            raise ValueError('type must be str or tf.DType but {} was given'.format(type(type_in_config)))
        return type_in_config

    def _one_hot_transform(self, placeholder, config):
        shape = config.get('shape', None)
        data_format = config.get('data_format', 'channels_last')
        if data_format == 'channels_last':
            n_classes = shape[-1]
            placeholder = tf.one_hot(placeholder, depth=n_classes, axis=-1)
        elif data_format == 'channels_first':
            n_classes = shape[0]
            placeholder = tf.one_hot(placeholder, depth=n_classes, axis=1)
        else:
            raise ValueError("data_format must be channels_last or channels_first",
                             "but {} was given".format(data_format))
        return placeholder


