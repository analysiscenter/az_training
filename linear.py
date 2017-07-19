import tensorflow as tf
import numpy as np
import dataset
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
# run this line to permit tf to use only gpu with number 1
#%env CUDA_VISIBLE_DEVICES=1
# replace on =[] if only cpu

import tensorflow as tf
# create session, allocate 50 % of gpu memory
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5


from sklearn.datasets import load_boston
from dataset import Dataset, Batch, DatasetIndex, action, model



class MyBatch(Batch):
    def __init__(self, index, *args, **kwargs):
        super().__init__(index, *args, **kwargs)

        
    @property
    def components(self):
        return "features", "labels"
    
#     @action
#     def load(self, f, l):
#         self.feature = f
#         self.label = l
#         return self
        
    @model()
    def linear_regression():
        num_dim = 14
        learning_rate = 0.01
        X = tf.placeholder(tf.float32, [None, num_dim])
        y = tf.placeholder(tf.float32, [None, 1])
        w = tf.Variable(tf.ones([num_dim, 1]))
        init = tf.global_variables_initializer()
        y_cup = tf.matmul(X, w)
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

        #fig, ax = plt.subplots()
        #ax.scatter(self.labels, y_pred)
        #ax.plot([self.labels.min(), self.labels.max()], [self.labels.min(), self.labels.max()], 'k--')
        #ax.set_xlabel('Measured')
        #ax.set_ylabel('Predicted')
        #plt.show()
        return self
    
    
def read_boston_data():
    boston = load_boston()
    features = np.array(boston.data)
    labels = np.array(boston.target)
    
    mu = np.mean(features, axis=0)
    sigma = np.std(features, axis=0)
    features = (features - mu)/sigma 
    features = np.c_[np.ones(features.shape[0]), features]
    labels = np.reshape(labels, [features.shape[0], 1])
    return features, labels


def load_dataset():
    features, labels = read_boston_data()
    data = features, labels
    num_items = features.shape[0]
    num_dim = features.shape[1]
    
    dataset_index = DatasetIndex(np.arange(num_items))
    dataset = Dataset(index=dataset_index, batch_class=MyBatch, preloaded=data)
    dataset.cv_split()
    return dataset, num_dim

    
if __name__ == "__main__":
    batch_size = 100
    training_epochs = 500
    dataset, num_dim = load_dataset()

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    cost_history = []

    for batch in dataset.train.gen_batch(batch_size, shuffle=True, n_epochs=training_epochs):
        batch.train(sess, cost_history)
    
    #plt.plot(range(len(cost_history)), cost_history)
    #plt.axis([0, training_epochs * (len(dataset.train.indices)  / batch_size), 0, np.max(cost_history)])
    #plt.show()

    dataset.test.reset_iter()
    test_batch = dataset.test.next_batch(len(dataset.test.indices))
    test_batch.test(sess)