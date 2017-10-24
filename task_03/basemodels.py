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

    def create_placeholders(self, name):

        """Create tf.placeholder.

        Parameters
        ----------
        name : str
            name of the placeholder before transformation
        postname : str or None (default)
            name of the placeholder after transformation

        Return
        ------
        out : tf.Tensor
        
        If postname is None, create_placeholder uses dict self.get_from_config(name),
        if str, uses dict self.get_from_config(postname):

        shape : tuple or list,
            the shape of placeholder the input of model which include the number of channels, input batch will be
            reshaped into that shape.
        input_type : str or tf.DType
            type of the input tensor. If str, it must be name of the tf type (int32, float64, ...).
        name : str
            name of the input in tf graph
        data_format : str
            one of channels_last (default) or channels_first. The ordering of the dimensions in the inputs. 
        n_outputs : int
        """
        
        placeholders_config = self.get_from_config(name)

        res = []

        for name, config in placeholders_config.items():
            shape = config.get('shape', None)
            dtype = self._create_type(config)
            transform = config.get('transform', None)
            postname = config.get('name', name+'-post')

            pl = tf.placeholder(dtype, name=name)

            if shape is not None:
                pl = tf.reshape(pl, [-1] + list(shape))
            if transform == 'label_to_oh':
                data_format = config.get('data_format', 'channels_last')
                n_classes = config.get('n_classes', None)
                if data_format == 'channels_last':
                    pl = tf.one_hot(pl, depth=n_classes, axis=-1)
                elif data_format == 'channels_first':
                    pl = tf.one_hot(pl, depth=n_classes, axis=1)
                else:
                    raise ValueError("data_format must be channels_last or channels_first",
                                     "but {} was given".format(data_format))
            res.append(tf.identity(pl, name=postname))
        return res

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
            type_in_config = tf.__getattribute__(type_in_config)
        elif isinstance(type_in_config, tf.DType):
            pass
        else:
            raise ValueError('type must be str or tf.DType but {} was given'.format(type(type_in_config)))
        return type_in_config
