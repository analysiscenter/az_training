''' Linear regression using dataset and tensor flow '''
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

from dataset import Dataset, Batch, action, model

NUM_DIM = 13

class MyBatch(Batch):
    ''' A Batch with linear regression model '''

    @property
    def components(self):
        ''' Define components '''
        return "features", "labels"

    @model()
    def linear_regression():
        ''' Define tf grapg for linear regression '''
        learning_rate = 0.01
        x_features = tf.placeholder(tf.float32, [None, NUM_DIM])
        y_target = tf.placeholder(tf.float32, [None, 1])
        weights = tf.Variable(tf.ones([NUM_DIM, 1]))
        bias = tf.Variable(tf.ones([1]))

        y_cup = tf.add(tf.matmul(x_features, weights), bias)
        cost = tf.reduce_mean(tf.square(y_cup - y_target))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        return training_step, cost, x_features, y_target, y_cup


    @action(model='linear_regression')
    def train(self, model_spec, session, my_cost_history):
        ''' Train batch '''
        training_step, cost, x_features, y_target = model[:-1]
        session.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(session.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return self

    @action(model='linear_regression')
    def test(self, model, session):
        ''' Test batch '''
        x_features = model[2]
        y_cup = model[4]
        y_pred = session.run(y_cup, feed_dict={x_features:self.features})
        mse = tf.reduce_mean(tf.square(y_pred - self.labels))
        print("MSE: %.4f" % session.run(mse))

        axis = plt.subplots()[1]
        axis.scatter(self.labels, y_pred)
        axis.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--')
        axis.set_xlabel('Measured')
        axis.set_ylabel('Predicted')
        plt.show()
        return self

def load_boston_data():
    ''' load some data '''
    boston = load_boston()
    labels = np.reshape(boston.target, [boston.target.shape[0], 1]) #pylint: disable=no-member
    return boston.data, labels #pylint: disable=no-member

def load_dataset(input_data):
    ''' create Dataset with given data '''
    dataset = Dataset(index=np.arange(input_data[0].shape[0]), batch_class=MyBatch, preloaded=input_data)
    dataset.cv_split()
    return dataset
