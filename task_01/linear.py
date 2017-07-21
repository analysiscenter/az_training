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

    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)

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
    def train(self, model, my_sess, my_cost_history):
    ''' Train batch ''' 
        training_step, cost, x_features, y_target = model[:-1]
        my_sess.run(training_step, feed_dict={x_features:self.features, y_target:self.labels})
        my_cost_history.append(my_sess.run(cost, feed_dict={x_features:self.features, y_target:self.labels}))
        return self

    @action(model='linear_regression')
    def test(self, model, sess):
    ''' Test batch '''
        x_features, y_target = model[2:4]
        y_cup = model[4]
        y_pred = sess.run(y_cup, feed_dict={x_features:self.features})
        mse = tf.reduce_mean(tf.square(y_pred - self.labels))
        print("MSE: %.4f" % sess.run(mse))

        fig, axis = plt.subplots()
        axis.scatter(self.labels, y_pred)
        axis.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--')
        axis.set_xlabel('Measured')
        axis.set_ylabel('Predicted')
        plt.show()
        return self

def load_boston_data():
    ''' load some data '''
    boston = load_boston()
    labels = np.reshape(boston.target, [boston.target.shape[0], 1])
    return boston.data, labels

def load_dataset(input_data):
    ''' create Dataset with given data '''
    dataset = Dataset(index=np.arange(input_data[0].shape[0]), batch_class=MyBatch, preloaded=input_data)
    dataset.cv_split()
    return dataset


if __name__ == "__main__":
    BATCH_SIZE = 100
    TRAINING_EPOCHS = 500

    data = load_boston_data()
    scale(data[0], axis=0, with_mean=True, with_std=True, copy=False)

    my_dataset = load_dataset(data)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    cost_history = []

    for batch in my_dataset.train.gen_batch(BATCH_SIZE, shuffle=True, n_epochs=TRAINING_EPOCHS):
        batch.train(sess, cost_history)

    plt.plot(range(len(cost_history)), cost_history)
    plt.axis([0, TRAINING_EPOCHS * (len(my_dataset.train.indices)  / BATCH_SIZE), 0, np.max(cost_history)])
    plt.show()

    test_batch = my_dataset.test.next_batch(len(my_dataset.test.indices))
    test_batch.test(sess)

