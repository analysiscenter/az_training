"LinkNet as TFModel"
import tensorflow as tf
from dataset.dataset.models.tf import TFModel

from layers import linknet_layers

class LinkNetModel(TFModel):
    "LinkNet as TFModel"
    def _build(self, *args, **kwargs):
        "build for LinkNet"
        image_size = self.get_from_config('image_size')
        x_ph = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, image_size, image_size], name='mask')
        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, image_size, image_size, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, image_size, image_size, 1])

        tf.concat([1 - mask_as_pics, mask_as_pics], axis=3, name='targets')

        model_output = linknet_layers(x_as_pics, training, 2)
        predictions = tf.identity(model_output, name='predictions')
        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')
