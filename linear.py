import numpy as np
import tensorflow as tf
import os
import sys
import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.preprocessing import scale

from dataset import Dataset, Batch, DatasetIndex, action, model

NUM_DIM = 13

class MyBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)

        
    @property
    def components(self):
        return "features", "labels"
    
        
    @model()
    def linear_regression():
        learning_rate = 0.01
        X = tf.placeholder(tf.float32, [None, NUM_DIM])
        y = tf.placeholder(tf.float32, [None, 1])
        w = tf.Variable(tf.ones([NUM_DIM, 1]))
        b = tf.Variable(tf.ones([1]))
        
        y_cup = tf.add(tf.matmul(X, w), b)
        cost = tf.reduce_mean(tf.square(y_cup - y))
        training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

        return training_step, cost, X, y, y_cup


    
    @action(model='linear_regression')
    def train(self, model, sess, cost_history):
        training_step, cost, X, y, y_cup = model
        sess.run(training_step, feed_dict={X:self.features, y:self.labels})
        cost_history.append( sess.run(cost, feed_dict={X:self.features, y:self.labels}))
        return self
    
    @action(model='linear_regression')
    def test(self, model, sess):
        training_step, cost, X, y, y_cup = model
        y_pred = sess.run(y_cup, feed_dict={X:self.features})
        mse = tf.reduce_mean(tf.square(y_pred - self.labels))
        print("MSE: %.4f" % sess.run(mse)) 

        fig, ax = plt.subplots()
        ax.scatter(self.labels, y_pred)
        ax.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--')
        ax.set_xlabel('Measured')
        ax.set_ylabel('Predicted')
        plt.show()
        return self
    

def load_boston_data():
    boston = load_boston()
    labels = np.reshape(boston.target, [boston.target.shape[0], 1])
    return boston.data, labels

def load_dataset(data):
    dataset = Dataset(index=np.arange(data[0].shape[0]), batch_class=MyBatch, preloaded=data)
    dataset.cv_split()
    return dataset

    
if __name__ == "__main__":
    batch_size = 100
    training_epochs = 500
    
    data = load_boston_data()
    scale(data[0], axis=0, with_mean=True, with_std=True, copy=False)

    dataset = load_dataset(data)
    
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    cost_history = []

    for batch in dataset.train.gen_batch(batch_size, shuffle=True, n_epochs=training_epochs):
        batch.train(sess, cost_history)
    
    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, training_epochs * (len(dataset.train.indices)  / batch_size), 0, np.max(cost_history)])
    plt.show()

    test_batch = dataset.test.next_batch(len(dataset.test.indices))
    test_batch.test(sess)
