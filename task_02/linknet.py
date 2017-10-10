import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('..')

from dataset import Batch, model, action, inbatch_parallel, any_action_failed
from layers import linknet_layers

SIZE = 64

class LinkNetBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.masks = None

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def noise_and_mask(self, ind):
        level = 0.7
        # threshold = 0.1
        pure_mnist = self.images[ind].reshape(28, 28)

        new_x, new_y = np.random.randint(0, SIZE-28, 2)
        mask = np.zeros((SIZE, SIZE))
        mask[new_x:new_x+28, new_y:new_y+28] = 1  # += pure_mnist > threshold

        noised_mnist = np.random.random((SIZE, SIZE))*level
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
        x_ph = tf.placeholder(tf.float32, shape=[None, SIZE * SIZE], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, SIZE * SIZE], name='mask')

        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, SIZE, SIZE, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, SIZE, SIZE, 1])

        mask_as_pics = tf.concat([mask_as_pics, 1 - mask_as_pics], axis=3)

        logits = linknet_layers(x_as_pics, training, 2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=mask_as_pics, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        y_pred_softmax = tf.nn.softmax(logits)
        y_pred = tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer(0.01).minimize(loss)

        #step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        return x_ph, mask_ph, training, step, loss, y_pred, y_pred_softmax, mask_as_pics

    @action(model='linknet')
    def train(self, models, sess, log):
        x_ph, mask_ph, training, step, loss, _, _, _ = models
        sess.run(step, feed_dict={x_ph: self.images, mask_ph: self.masks, training: True})
        batch_loss = sess.run(loss, feed_dict={x_ph: self.images, mask_ph: self.masks, training: True})
        log.append(batch_loss)
        return self

    @action(model='linknet')
    def predict(self, models, sess, pred):
        x_ph, mask_ph, training, step, loss, y_pred, _, _ = models
        res = sess.run(y_pred, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        pred.append([self.images, res])
        return self

    @action(model='linknet')
    def predict_proba(self, models, sess, pred):
        x_ph, mask_ph, training, step, loss, _, y_pred, _ = models
        proba = sess.run(y_pred, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        loss_test = sess.run(loss, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        pred.append([loss_test, self.images, self.masks, proba])
        return self

    @action(model='linknet')
    def get_mask(self, models, sess, masks):
        x_ph, mask_ph, training, step, loss, _, y_pred, mask_as_pics = models
        mask = sess.run(mask_as_pics, feed_dict={x_ph: self.images, mask_ph: self.masks, training: False})
        masks.append(mask)
        return self