""" File with some useful functions"""
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from pandas import ewma
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as Xavier


plt.style.use('seaborn-poster')
plt.style.use('ggplot')

def draw(first, first_label, second=None, second_label=None, type_data='loss', window=50, bound=None, axis=None):

    """ Draw on graph first and second data.

    The graph shows a comparison of the average values calculated with a 'window'. You can draw one graph
    or create your oun subplots and one of it in 'axis'.

    Parameters
    ----------
    first: list or numpy array
    Have a values to show

    first_label: str
    Name of first data

    second: list or numpy array, optional
    Have a values to show


    second_label: str, optional
    Name of second data

    type_data: str, optional
    Type of data. Example 'loss', 'accuracy'

    window: int, optional
    window width for calculate average value

    bound: list or None
    Bounds to limit graph: [min x, maxis x, min y, maxis y]

    axis: None or element of subplot
    If you want to draw more subplots give the element of subplot """

    firt_ewma = ewma(np.array(first), span=window, adjust=False)
    second_ewma = ewma(np.array(second), span=window, adjust=False) if second else None

    plot = axis or matplotlib.pyplot
    plot.plot(firt_ewma, label='{} {}'.format(first_label, type_data))
    if second_label:
        plot.plot(second_ewma, label='{} {}'.format(second_label, type_data))

    if axis is None:
        plot.xlabel('Iteration', fontsize=16)
        plot.ylabel(type_data, fontsize=16)
    else:
        plot.set_xlabel('Iteration', fontsize=16)
        plot.set_ylabel(type_data, fontsize=16)

    plot.legend(fontsize=14)
    if bound:
        plot.axis(bound)

def bottle_conv_block(input_tensor, kernel, filters, name=None, \
 strides=(2, 2)):
    """ Function to create block bottleneck of ResNet network which incluce \
    three convolution layers and one skip-connetion layer.

    Parameters
    ----------
    input_tensor: tf.layer
    input tensorflow layer

    kernel: tuple
    Kernel size in convolution layer

    filters:list
    Nums filters in convolution layers with len = 3

    name:str
    Name of blocks

    strides: typle
    Strides in convolution layers

    Returns:
    ----------

    X: tf.layer
    Block output layer """
    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3 = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), strides, name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
        kernel_initializer=Xavier(uniform=True))

    shortcut = tf.layers.conv2d(input_tensor, filters3, (1, 1), strides, \
               name='shortcut_conv_1X1_of_' + name, kernel_initializer=Xavier(uniform=True))

    X = tf.add(X, shortcut)
    X = tf.nn.relu(X)
    return X

def bottle_identity_block(input_tensor, kernel, filters, name=None):
    """ Function to create bottleneck-identity block of ResNet network.
    Parameters
    ----------
    input_tensor: tf.layer
    input tensorflow layer

    kernel: tuple
    Kernel size in convolution layer

    filters:list
    Nums filters in convolution layers with len = 3

    name:str
    Name of blocks

    Returns:
    ----------

    X: tf.layer
    Block output layer """
    if name is None:
        name = str(sum(np.random.random(100)))
    filters1, filters2, filters3 = filters
    X = tf.layers.conv2d(input_tensor, filters1, (1, 1), name='first_conv_1X1_of_' + name, \
        activation=tf.nn.relu, kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters2, kernel, name='conv_3X3_of_' + name, activation=tf.nn.relu, \
        padding='SAME', kernel_initializer=Xavier(uniform=True))

    X = tf.layers.conv2d(X, filters3, (1, 1), name='second_conv_1X1_of_' + name,\
                         kernel_initializer=Xavier(uniform=True))

    X = tf.add(X, input_tensor)
    X = tf.nn.relu(X)
    return X

def accum_accuracy(ind, models, data):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
    Output:
        acc: accuracy after one iteration.
        loss: loss after one iteration. """
    indices, all_images, all_labels, loss, _ = models[0]
    acc, sess = models[1]
    loss, acc = sess.run([loss, acc], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1]})
    return acc, loss

def training(ind, models, data, all_time):
    """ Function to calculate accuracy.
    Args:
        ind: indices elements of this batch.
        model: what model to u use.
        data: all data.
        all_time: list with len=1, calculate time for training model.
    Output:
        loss: loss after one iteration. """
    indices, all_images, all_labels, loss, train = models[0]
    sess = models[1][1]
    timer = time.time()
    _, loss = sess.run([train, loss], feed_dict={indices:ind.reshape(-1, 1), all_images:data[0], \
                                                 all_labels:data[1]})
    all_time[0] += time.time() - timer
    return loss



def axis_draw(freeze_loss, res_loss, src, axis):
    """ Draw graphs to compare models. Theaxis graph shows a comparison of the average
        values calculated with a window in 10 values.
    Args:
        freeze_loss: List with loss value in resnet and freezeout model
        res_loss: List with loss value in clear resnet
        src: List with parameters of model with FreezeOut
        axis: Plt sublot """
    fr_loss = []
    n_loss = []

    for i in range(10, len(res_loss) - 10):
        fr_loss.append(np.mean(freeze_loss[i-10:i+10]))
        n_loss.append(np.mean(res_loss[i-10:i+10]))

    axis.set_title('Freeze model with: LR={} Degree={} It={} Scaled={}'.format(*src))
    axis.plot(fr_loss, label='freeze loss')
    axis.plot(n_loss, label='no freeze loss')
    axis.set_xlabel('Iteration', fontsize=16)
    axis.set_ylabel('Loss', fontsize=16)
    axis.legend(fontsize=14, loc=3)
