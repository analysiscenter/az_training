import tensorflow as tf
from dataset.dataset.models.tf import TFModel

from layers import linknet_layers

class LinkNetModel(TFModel):
    def _build(self, *args, **kwargs):
        SIZE = self.get_from_config('image_size')
        x_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='mask')

        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, SIZE, SIZE, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, SIZE, SIZE, 1])

        targets = tf.concat([1 - mask_as_pics, mask_as_pics], axis=3, name='targets')

        model_output = linknet_layers(x_as_pics, training, 2)

        predictions = tf.identity(model_output, name='predictions')

        y_pred_softmax = tf.nn.softmax(predictions, name='predicted_prob')
        predicted_labels = tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')

        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=mask_as_pics_one_hot, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer().minimize(loss)
        """
