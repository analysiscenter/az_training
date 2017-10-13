"""
LinkNet implementation as Batch class
"""

import pickle
import numpy as np
import tensorflow as tf
import sys
import itertools

sys.path.append('..')

from dataset import Batch, model, action, inbatch_parallel, any_action_failed
from layers import linknet_layers

SIZE = 64

def uniform_pieces(*args, **kwargs):
    size = kwargs['size']
    n_pieces = kwargs['n_pieces']
    xs = np.random.randint(0, 28 - size, n_pieces)
    ys = np.random.randint(0, 28 - size, n_pieces)
    return xs, ys

def uniform_arrangement(*args, **kwargs):
    return np.random.randint(0, SIZE, 2)

def crop_images(images, coordinates):
    images_for_noise = []
    for i in range(len(images)):
        image = images[i]
        coord = coordinates[i]
        images_for_noise.append(image[coord[0]:coord[1] + 28, coord[0]:coord[1] + 28])
    return images_for_noise

def create_pieces(images, n_pieces, size, pieces_distr):
    """Cut each image into pieces."""
    pieces = []
    for image in images:
        x_pieces, y_pieces = locals()[pieces_distr](size=size, n_pieces=n_pieces) 
        coordinates = list(zip(x_pieces, y_pieces))
        pieces += [image[x:x+size, y:y+size] for x, y in coordinates]
    return pieces


def arrange_pieces(image, pieces, level=1):
    for piece in pieces:
        x_piece, y_piece = locals()[location_distr]()
        height = piece.shape[0]
        width = piece.shape[1]
        image_to_change = image[x_piece:x_piece+height, y_piece:y_piece+width]
        height_to_change, width_to_change = image_to_change.shape
        image_to_change = np.max([level*piece[:height_to_change, :width_to_change], image_to_change], axis=0)
        image[x_piece:x_piece+height, y_piece:y_piece+width] = image_to_change
    return image


class LinkNetBatch(Batch):
    """Batch class for LinkNet."""

    def __init__(self, index, *args, **kwargs):
        """Init function."""
        super().__init__(index, *args, **kwargs)
        self.images = None
        self.masks = None
        self.coordinates = None

    @property
    def components(self):
        """Define components."""
        return 'images', 'masks', 'coordinates'
    
    @action
    def load_images(self):
        """Load MNIST images from file."""
        with open('../mnist/mnist_pics.pkl', 'rb') as file:
            self.images = pickle.load(file)[self.indices]
        return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_image', target='threads')
    def random_location(self, ind):
        """Put MNIST image in random location"""
        pure_mnist = self.images[ind].reshape(28, 28)
        new_x, new_y = np.random.randint(0, SIZE-28, 2)
        large_mnist = np.zeros((SIZE, SIZE))
        large_mnist[new_x:new_x+28, new_y:new_y+28] = pure_mnist
        return large_mnist, new_x, new_y

    def init_func(self, *args, **kwargs):
        """Create tasks."""
        return [{'ind': i} for i in range(self.images.shape[0])]

    def post_func_image(self, list_of_res, *args, **kwargs):
        """Concat outputs from random_location.

        Parameters
        ----------
        list_of_res : list of tuples of np.arrays and two ints
        """

        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            images, new_x, new_y = list(zip(*list_of_res))
            self.images = np.stack(images)
            self.coordinates = list(zip(new_x, new_y))
            return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_mask', target='threads')
    def create_mask(self, ind):
        """Get mask of MNIST image"""
        new_x, new_y = self.coordinates[ind]
        mask = np.zeros((SIZE, SIZE))
        mask[new_x:new_x+28, new_y:new_y+28] += 1
        return mask

    def post_func_mask(self, list_of_res, *args, **kwargs):
        """Concat outputs from random_location.

        Parameters
        ----------
        list_of_res : list of tuples of np.arrays and two ints
        """

        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            self.masks = np.stack(list_of_res)
            return self

    @action
    @inbatch_parallel(init='init_func', post='post_func_noise', target='threads')
    def add_noise(self, ind, noise_params):
        """Add noise at MNIST image"""
        n_image = noise_param['n_image']
        n_pieces = noise_param['n_pieces']
        size = noise_param['size']
        
        
        
        ind_for_noise = np.random.choice(len(self.images), noise_param['n_image'])
        images = [self.images[i] for i in ind_for_noise]
        coordinates = [self.coordinates[i] for i in ind_for_noise]
        images_for_noise = crop_images(images, coordinates)
        pieces = create_pieces(images_for_noise, **noise_param)
        noise = arrange_pieces(self.images[ind], **noise_param)

        return noise

    def post_func_noise(self, list_of_res, *args, **kwargs):
        """Concat outputs from add_noise.
        """

        if any_action_failed(list_of_res):
            print(list_of_res)
            raise Exception("Something bad happened")
        else:
            self.images = np.stack(list_of_res)
            return self


    @model()
    def linknet():
        """Define LinkNet model."""
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
        """Get values of model return.

        Parameters
        ----------
        models : list of tensors
            return of linknet()
        sess : tf.Session

        index : int
            index of the computed tensor in models

        training: bool
            training parameter for tf.layers.batch_normalization
        """

        feed_dict = {models[0]: self.images, models[1]: self.masks, models[2]: training}
        return sess.run(models[index], feed_dict=feed_dict)

    @action
    def get_images(self, images, masks):
        """Get images from batch.

        Parameters
        ----------
        images : list of np.array

        masks : list of np.array
        """

        images.append(self.images)
        masks.append(self.masks)
        return self

    @action(model='linknet')
    def train(self, models, sess):
        """Train iteration.

        Parameters
        ----------
        models : list of tensors
            return of linknet()

        sess : tf.Session
        """
        
        self.get_tensor_value(models, sess, 3, True)
        return self

    @action(model='linknet')
    def get_stat(self, models, sess, log, training):
        """Loss on batch.

        Parameters
        ----------
        models : list of tensors
            return of linknet()

        sess : tf.Session

        log : list
            list to append loss on batch

        training: bool
            training parameter for tf.layers.batch_normalization
        """

        log.append(self.get_tensor_value(models, sess, 4, training))
        return self

    @action(model='linknet')
    def predict(self, models, sess, pred):
        """Get segmentation for batch.

        Parameters
        ----------
        models : list of tensors
            return of linknet()

        sess : tf.Session

        training: bool
            training parameter for tf.layers.batch_normalization
        """
        pred.append(self.get_tensor_value(models, sess, 5, False))
        return self

    @action(model='linknet')
    def predict_proba(self, models, sess, pred):
        """Get predicted pixel-wise class probabilities for batch.

        Parameters
        ----------
        models : list of tensors
            return of linknet()

        sess : tf.Session

        training: bool
            training parameter for tf.layers.batch_normalization
        """
        pred.append(self.get_tensor_value(models, sess, 6, False))
        return self
