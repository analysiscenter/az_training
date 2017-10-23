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
        super(NetworkModel, self).__init__(*args, **kwargs)
        self.input_shape = None

    def create_input(self):
        """Create tf.placeholder for model input.

        Return
        ------
        reshaped_inp : tf.Tensor


        dim is a spatial dimension of input and ignores channels.
        images_shape is a shape of the input array.
        create_input reshapes input placeholder into (dim+1)-tensor.

        For example,
        1) input array of greyscale images with shape (n, 28, 28). Then images_shape is (28, 28), dim = 2.
        The output is tensor of shape (n, 28, 28, 1).
        2) input array of RGB images with shape (n, 640, 480, 3). Then images_shape is (640, 480, 3), dim = 2;
        The output is tensor of shape (n, 640, 480, 3).
        3) input array of flatten greyscale images with shape (n, 784). Then images_shape is (784, ), dim = 2;
        The output is tensor of shape (n, 28, 28, 1).
        """

        images_shape = list(self.get_from_config('images_shape'))
        dim = self.get_from_config('dim', 2)

        inp = tf.placeholder(tf.float32, shape=[None] + images_shape, name='input_image')

        if len(images_shape) == dim + 1:
            true_images_shape = images_shape
        elif len(images_shape) == dim:
            true_images_shape = images_shape + [1]
        elif len(images_shape) == 1:
            root = round(images_shape[0] ** (1 / dim))
            if root ** dim == images_shape[0]:
                true_images_shape = [root] * dim + [1]
            else:
                raise ValueError("images_shape has incorrect value")
        else:
            raise ValueError('len(images_shape) must be equal to 1, dim or dim + 1')
        self.input_shape = true_images_shape
        reshaped_inp = tf.reshape(inp, [-1] + true_images_shape)
        return reshaped_inp

    def create_target(self, task='classification'):
        """Create tf.placeholders for model target.

        Parameters
        ----------
        task : str
            type of the target: 'classification', 'classification-one-hot', 'segmenatation', 'regression'

        Return
        ------
        targets : tf.placeholder
            shape depends on task
            'classification', 'classification-one-hot' : [None, n_classes]
            'regression' : [None]
            'segmentation' : [None] + reshaped_images_shape + [n_classes]
        """
        n_classes = self.get_from_config('n_classes', 2)
        images_shape = self.input_shape

        if task == 'regression' or task == 'classification':
            shape = [None]
        elif task == 'segmentation':
            shape = [None] + images_shape[:-1] + [n_classes]
        elif task == 'classification-one-hot':
            shape = [None] + [n_classes]
        else:
            raise ValueError('Unknown task')

        if task == 'classification':
            labels = tf.placeholder(tf.int32, shape=shape, name='labels')
            targets = tf.one_hot(labels, n_classes, name='targets')
        else:
            targets = tf.placeholder(tf.float32, shape=shape, name='targets')
        return targets

    def create_output(self, outp):
        """Create output for models which produce logit output
        Return
        ------
        predicted_prob, predicted_labels : tf.tensors
        """
        predicted_prob = tf.nn.softmax(outp, name='predicted_prob')
        max_axis = len(outp.get_shape())-1
        predicted_labels = tf.argmax(predicted_prob, axis=max_axis, name='predicted_labels')
        return predicted_prob, predicted_labels
