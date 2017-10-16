"LinkNet as TFModel"
import tensorflow as tf
from dataset.dataset.models.tf import TFModel

from layers import linknet_layers

class LinkNetModel(TFModel):
    "LinkNet as TFModel"
    def _build(self):
        "build for LinkNet"
        SIZE = self.get_from_config('image_size')
        x_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='mask')

        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, SIZE, SIZE, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, SIZE, SIZE, 1])

        targets = tf.concat([1 - mask_as_pics, mask_as_pics], axis=3, name='targets') # pylint: disable=unused-variable

        model_output = linknet_layers(x_as_pics, training, 2)

        predictions = tf.identity(model_output, name='predictions')

        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        predicted_labels = tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction') # pylint: disable=unused-variable
