import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('..')

from dataset import Batch, model, action, inbatch_parallel, any_action_failed
from layers import linknet_layers


class LinkNetBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.masks = None

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def noise_and_mask(self, ind):
        level = 0.5
        threshold = 0.1
        pure_mnist = self.images[ind].reshape(28, 28)

        new_x, new_y = np.random.randint(0, 100, 2)
        mask = np.zeros((128,128))
        mask[new_x:new_x+28, new_y:new_y+28] = 1 #+= pure_mnist > threshold

        noised_mnist = np.random.random((128,128))*level
        noised_mnist[new_x:new_x+28, new_y:new_y+28] += pure_mnist
        noised_mnist /= 1 + level
        return noised_mnist.reshape(-1), mask.reshape(-1)

    def init_func(self):
        return [{'ind': i} for i in range(self.images.shape[0])]

    def post_func(self, list_of_res):
        """ Concat outputs from shift_flattened_pic """
        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, masks = list(zip(*list_of_res))
            self.images = np.stack(images)
            self.masks = np.stack(masks)
            return self

    @action
    def load(self):
        with open('../mnist/mnist_labels.pkl', 'rb') as file:
            self.masks = pickle.load(file)[self.indices]
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.images = pickle.load(file)[self.indices]
        return self

    @action
    def get_images(self, images, masks):
        images.append(self.images)
        masks.append(self.masks)
        return self

    @model()
    def linknet():
        x_ph = tf.placeholder(tf.float32, shape=[None, 128 * 128], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, 128 * 128], name='mask')

        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, 128, 128, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, 128, 128, 1])

        mask_as_pics = tf.concat([mask_as_pics, 1 - mask_as_pics], axis=3)

        logits = linknet_layers(x_as_pics, training, 2)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=mask_as_pics, logits=logits))

        y_pred_softmax = tf.nn.softmax(logits)
        y_pred = tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')

        step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        return x_ph, mask_ph, training, step, loss, y_pred, y_pred_softmax

    @action(model='linknet')
    def train(self, models, sess, log):
        x_ph, mask_ph, training, step, loss, _, _ = models
        sess.run(step, feed_dict={x_ph: self.images, mask_ph: self.masks, training: True})
        batch_loss = sess.run(loss, feed_dict={x_ph: self.images, mask_ph: self.masks, training: True})
        log.append(batch_loss)
        return self

    @action(model='linknet')
    def predict(self, models, sess, pred):
        x_ph, mask_ph, training, step, loss, y_pred, _ = models
        res = sess.run(y_pred, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        pred.append([self.images, res])
        return self

    def predict_proba(self, models, sess, pred):
        x_ph, mask_ph, training, step, loss, _, y_pred = models
        res = sess.run(y_pred, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        pred.append([self.images, res])
        return self
