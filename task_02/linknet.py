import sys
import numpy as np
import pickle
import tensorflow as tf

sys.path.append('..')

from dataset import Batch, model, action, inbatch_parallel, any_action_failed
from layers import linknet_layers

SIZE = 128


class LinkNetBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.masks = None

    @property
    def components(self):
        """ Define components. """
        return 'images', 'masks'

    @action
    @inbatch_parallel(init='init_func', post='post_func', target='threads')
    def noise_and_mask(self, ind):
        level = 0.7
        #threshold = 0.1
        pure_mnist = self.images[ind].reshape(28, 28)

        new_x, new_y = np.random.randint(0, SIZE-28, 2)
        mask = np.zeros((SIZE, SIZE))
        mask[new_x:new_x+28, new_y:new_y+28] += 1 #pure_mnist > threshold

        noised_mnist = np.random.random((SIZE, SIZE))*level
        noised_mnist[new_x:new_x+28, new_y:new_y+28] += pure_mnist
        noised_mnist /= 1 + level
        return noised_mnist, mask

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


    @model()
    def linknet():
        x_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='image')
        mask_ph = tf.placeholder(tf.float32, shape=[None, SIZE, SIZE], name='mask')

        training = tf.placeholder(tf.bool, shape=[], name='mode')

        x_as_pics = tf.reshape(x_ph, [-1, SIZE, SIZE, 1])
        mask_as_pics = tf.reshape(mask_ph, [-1, SIZE, SIZE, 1])

        mask_as_pics_one_hot = tf.concat([1 - mask_as_pics, mask_as_pics], axis=3)

        logits = linknet_layers(x_as_pics, training, 2)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=mask_as_pics_one_hot, logits=logits)
        loss = tf.reduce_mean(cross_entropy)

        y_pred_softmax = tf.nn.softmax(logits)
        y_pred = tf.cast(tf.argmax(y_pred_softmax, axis=3), tf.float32, name='mask_prediction')
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            step = tf.train.AdamOptimizer().minimize(loss)

        return x_ph, mask_ph, training, step, loss, y_pred, y_pred_softmax

    def get_tensor_value(self, models, sess, index, training):
        feed_dict={models[0]: self.images, models[1]: self.masks, models[2]: training}
        return sess.run(models[index], feed_dict=feed_dict)

    @action
    def get_images(self, images, masks):
        images.append(self.images)
        masks.append(self.masks)
        return self

    @action(model='linknet')
    def train(self, models, sess):
        self.get_tensor_value(models, sess, 3, True)
        return self
    
    @action(model='linknet')
    def get_stat(self, models, sess, log, training):
        log.append(self.get_tensor_value(models, sess, 4, training))
        return self

    @action(model='linknet')
    def predict(self, models, sess, pred):
        pred.append(self.get_tensor_value(models, sess, 5, False))
        return self

    @action(model='linknet')
    def predict_proba(self, models, sess, pred):
        pred.append(self.get_tensor_value(models, sess, 6, False))
        return self
