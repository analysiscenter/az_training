""" File with class batch with resnet network """
import sys

import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier

sys.path.append('..')
from batch import conv_block, identity_block
from dataset import action, model, Batch

def create_train(opt, src, global_step, it, global_it, learn, scaled):
    """ Function for create optimizer to each layer.
        Args:
            src: name of layer which be optimize.
            global_step: tenforflow Variable. Need to count train steps.
            it: number of last iteraion for current layer.
            global_it: number of last interation for all layers.
            learn: Basic learning rate for current layer.
            scaled: method of disable layers.
        Output:
            New optimizer. """
    def learning_rate(last, src, global_it, learn, scaled):
        """ Function for create step of changing learning rate.

        Args:
            last: number of last iteration.
            src: mane of layer which be optimize.
            global_it: number of last interation for all layers.
            learn: Basic learning rate for current layer.
            scaled: method of disable layers.

        Output:
            bound: list of bounders - number of iteration, after which learning rate will change.
            values: list of new learnings rates.
            var: name of optimize layers"""

        last = int(last)
        if scaled is True:
            values = [0.5 * learn/last * (1 + np.cos(np.pi * i / last)) for i in range(2, last+1)] + [1e-2]
        else:
            values = [0.5 * learn * (1 + np.cos(np.pi * i / last)) for i in range(2, last+1)] + [1e-2]

        bound = list(np.linspace(0, last, len(range(2, last+1)), dtype=np.int32)) + [global_it]
        var = [i for i in tf.trainable_variables() if src in i.name]

        return list(np.int32(bound)), list(np.float32(values)), var
    if src == '6':
        var = [i for i in tf.trainable_variables() if src in i.name]
        return opt(0.0001, 0.9, use_nesterov=True), var
    b, val, var = learning_rate(it, src, global_it, learn, scaled)
    learning_rate = tf.train.piecewise_constant(global_step, b, val)
    return opt(learning_rate, 0.9, use_nesterov=True), var


class ResBatch2(Batch):
    """ Batch to train models with and without FreezeOut """

    def __init__(self, index, *args, **kwargs):
        """ Init function """
        super().__init__(index, *args, **kwargs)

    @property
    def components(self):
        """ Define componentis. """
        return 'images', 'lables'


    @model(mode='dynamic')
    def new_model(self, config=None): # too-many-locals
        """ Simple implementation of ResNet with FreezeOut method.
        Args:
            config: dict with params:
                -iterations: Total number iteration for train model.
                -degree: 1 or 3.
                -learning_rate: initial learning rate.
                -scaled: True or False.

        Outputs:
            Method return list with len = 2 and some params:
            [0][0]: indices - Plcaeholder which takes batch indices.
            [0][1]: all_data - Placeholder which takes all images.
            [0][2]; all_lables - Placeholder for lables.
            [0][3]: loss - Value of loss function.
            [0][4]: train - List of train optimizers.
            [0][5]: prob - softmax output, need to prediction.

            [1][0]: accuracy - Current accuracy
            [1][1]: session - tf session """

        with tf.Graph().as_default():

            indices = tf.placeholder(tf.int32, shape=[None, 1], name='indices')
            all_data = tf.placeholder(tf.float32, shape=[50000, 28, 28], name='all_data')
            input_batch = tf.gather_nd(all_data, indices, name='input_batch')
            input_batch = tf.reshape(input_batch, shape=[-1, 28, 28, 1], name='x_to_tens')

            net = tf.layers.conv2d(input_batch, 32, (7, 7), strides=(2, 2), padding='SAME', activation=tf.nn.relu, \
                                   kernel_initializer=xavier(), name='0')
            net = tf.layers.max_pooling2d(net, (2, 2), (2, 2), name='max_pool')

            net = conv_block(net, 3, [32, 32, 128], name='1', strides=(1, 1))
            net = identity_block(net, 3, [32, 32, 128], name='2')

            net = conv_block(net, 3, [64, 64, 256], name='3', strides=(1, 1))
            net = identity_block(net, 3, [64, 64, 256], name='4')

            net = tf.layers.average_pooling2d(net, (7, 7), strides=(1, 1))
            net = tf.contrib.layers.flatten(net)

            net = tf.layers.dense(net, 10, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='5')

            prob = tf.nn.softmax(net, name='soft')
            all_lables = tf.placeholder(tf.float32, [None, 10], name='all_lables')
            y = tf.gather_nd(all_lables, indices, name='y')
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=y), name='loss')

            global_steps = []
            optimizer, grad_accum = [], []
            apl = []

            for i in range(6):
                global_steps.append(tf.Variable(0, trainable=False, name='var_{}'.format(i)))
                opt = create_train(tf.train.MomentumOptimizer, str(i), \
                                   global_steps[-1], config['iteration'] * (i / 10 + 0.5) ** config['degree'], \
                                   config['iteration'], config['learning_rate'], config['scaled'])

                optimizer.append(opt[0])
                grad_accum.append(opt[0].compute_gradients(loss, opt[1]))

            for i in range(6):
                apl.append(optimizer[i].apply_gradients(grad_accum[i], global_steps[i]))


            lables_hat = tf.cast(tf.argmax(net, axis=1), tf.float32, name='lables_hat')
            lables = tf.cast(tf.argmax(y, axis=1), tf.float32, name='lables')
            accuracy = tf.reduce_mean(tf.cast(tf.equal(lables_hat, lables), tf.float32, name='accuracy'))

            session = tf.Session()
            session.run(tf.global_variables_initializer())
        return [[indices, all_data, all_lables, loss, apl, prob], [accuracy, session]]

    @action(model='new_model')
    def train_freez(self, models, train_loss, data, lables):
        """ Function for traning ResNet with freezeout method.
        Args:
            sess: tensorflow session.
            train_loss: list with info of train loss.
            train_acc: list with info of train accuracy.

        Output:
            self """
        indices, all_data, all_lables, loss, train, _ = models[0]
        session = models[1][1]

        loss, _ = session.run([loss, train], feed_dict={indices:self.indices.reshape(-1, 1), all_lables:lables, \
            all_data:data})

        train_loss.append(loss)

        return self
