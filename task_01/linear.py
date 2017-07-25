''' Logistic regression using dataset and tensor flow '''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston, make_blobs
from sklearn.preprocessing import scale
from sklearn.datasets import make_classification

from dataset import Dataset, Batch, action, model

NUM_DIM = 13


class MyBatch(Batch):
    ''' A Batch with logistic regression model '''

    @property
    def components(self):
        ''' Define components '''
        return "features", "labels"

    @model()
    def linear_regression():
        ''' Define tf graph for linear regression '''
        learning_rate = 0.01
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.ones([NUM_DIM, 1]))
        bias = tf.Variable(tf.ones([1]))

        y_cup = tf.add(tf.matmul(x_features, weights), bias)
        cost = tf.reduce_mean(tf.square(y_cup - y_target))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        return training_step, cost, x_features, y_target, y_cup

    @action(model='linear_regression')
    def train_linear(self, model, my_sess, my_cost_history):
        ''' Train linear regression on the batch '''
        training_step, cost, x_features, y_target = model[:-1]
        my_sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(my_sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return self

    @action(model='linear_regression')
    def test_linear(self, model, sess):
        ''' Test linear regression on the batch '''
        x_features = model[2]
        y_cup = model[4]
        y_pred = sess.run(y_cup, feed_dict={x_features:self.features})
        mse = tf.reduce_mean(tf.square(y_pred - self.labels))
        print("MSE: %.4f" % sess.run(mse))
        
        axis = plt.subplots()[1]
        axis.scatter(self.labels, y_pred)
        axis.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--')
        axis.set_xlabel('Measured')
        axis.set_ylabel('Predicted')
        plt.show()
        return self

    @model()
    def logistic_regression():
        '''Define tf graph for logistic regression model'''
        learning_rate = 0.005
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.zeros([NUM_DIM, 1]))
        bias = tf.Variable(tf.zeros([1]))
        
        wx_b = tf.add(tf.matmul(x_features, weights), bias)

        prob = tf.sigmoid(wx_b)
        y_cup = tf.sign(wx_b)
        
        margin = tf.multiply(y_target, wx_b)
        cost = tf.reduce_mean(tf.log(tf.add(tf.ones([1]), tf.exp(-margin))))
        training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)

        return training_step, cost, x_features, y_target, y_cup

    @action(model='logistic_regression')
    def train_logistic(self, model, my_sess, my_cost_history):
        ''' Train logistic regression on the batch '''
        training_step, cost, x_features, y_target = model[:-1]
        my_sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(my_sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return self
    
    @action(model='logistic_regression')
    def test_logistic(self, model, my_sess):
        ''' Test logistic regression on the batch'''
        x_features = model[2]
        y_cup = model[4]
        y_pred = my_sess.run(y_cup, feed_dict={x_features:self.features})
        
        y_target_int = tf.cast(self.labels, tf.int32)
        y_pred_int = tf.cast(y_pred, tf.int32)
        
        acc = tf.contrib.metrics.accuracy(y_pred_int, y_target_int)
        print("ACCURACY: %.4f" % my_sess.run(acc))
        return self

    @model()
    def poisson_regression():
        '''Define tf graph for logistic regression model'''
        learning_rate = 0.005
        x_features = tf.placeholder(tf.float32, [None, None])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.zeros([NUM_DIM, 1]))
        bias = tf.Variable(tf.zeros([1]))

        log_input = tf.add(tf.matmul(x_features, weights), bias)
        cost = tf.reduce_mean(tf.nn.log_poisson_loss(y_target, log_input, compute_full_loss=False))
        
        y_cup = tf.exp(log_input)
            
        training_step = tf.train.AdamOptimizer(learning_rate).minimize(cost)
        return training_step, cost, x_features, y_target, y_cup


    @action(model='poisson_regression')
    def train_poisson(self, model, my_sess, my_cost_history):
        ''' Train poisson regression on the batch '''
        training_step, cost, x_features, y_target = model[:-1]
        my_sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(my_sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return self


    @action(model='poisson_regression')
    def test_poisson(self, model, my_sess):
        ''' Test poisson regression on the batch '''
        cost = model[1]
        x_features = model[2]
        y_cup = model[-1]

        print ('Y_CUP SHAPE ', y_cup.shape)
        
        y_pred = my_sess.run(y_cup, feed_dict={x_features:self.features})
        # y_pred_sm = tf.round(y_pred) + 1 if ? tf.round(y_pred) 0 : tf.round(y_pred) + 1 : tf.round(y_pred)
        print (y_pred.shape, '@@@@@@@@@@')

        y_pred_int = my_sess.run(tf.cast(tf.round(y_pred) + 1, tf.int32))


        mse = tf.reduce_mean(tf.square(y_pred_int - self.labels))
        # log_loss = tf.losses.log_loss(self.labels, y_pred_int, epsilon=1e-05)
        # print ('PRED ', y_pred_int)
        # print ('TRUE ', self.labels)
        print ()
        print ("MSE: %.4f" % my_sess.run(mse))
        return self


def load_boston_data():
    ''' load some data '''
    boston = load_boston()
    labels = np.reshape(boston.target, [boston.target.shape[0], 1])
    return boston.data, labels


def load_poisson_data():
    '''generate poisson data'''
    features = np.random.random_sample([500, NUM_DIM])
    weights = np.random.random_sample([NUM_DIM])
    xw = np.exp(np.dot(features, weights))
    labels = np.reshape(np.array([np.random.poisson(lmbd) for lmbd in xw]), [500, 1])
    labels.astype(np.float32)
    return features, labels


def load_random_data(blobs=False):
    ''' load some data '''
    if blobs:
        features, labels = make_blobs(n_samples=500, n_features=13, random_state=1, centers=2)
    else:
        features, labels = make_classification(n_samples=500, n_features=13, random_state=1, n_classes=2, flip_y=0.005)
    labels = np.reshape(labels, [labels.shape[0], 1])
    return features, labels


def load_dataset(input_data):
    ''' create Dataset with given data '''
    dataset = Dataset(index=np.arange(input_data[0].shape[0]), batch_class=MyBatch, preloaded=input_data)
    dataset.cv_split()
    return dataset


def preprocess_linear_data(data):
    ''' Normalize data '''
    scale(data[0], axis=0, with_mean=True, with_std=True, copy=False)
    return data


def preprocess_logistic_data(data):
    ''' Change labbel of the second class to '-1' instead of 0'''
    features, labels = data
    labels = 2*data[1] - np.ones((len(data[1]), 1), dtype=np.float32)
    return features, labels

