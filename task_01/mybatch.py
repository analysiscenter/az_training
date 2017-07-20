import re
import numpy as np
import sys
sys.path.append('../')
import tensorflow as tf
from dataset import Dataset, DatasetIndex, Batch, action, model

def generate_linear_data(lenght = 10):
    """
    Generation of data for fit linear regression.

    lenght - lenght of data.

    return:
    X - array [0..lenght]
    y - array [0..lenght] with some random noize
    """

    y = np.linspace(0,10,lenght)
    X = y + np.random.random(lenght) - 0.5
    return X,y

def generate_logistic_data(lenght = 10):
    """
    Generation of data for fit logistic regression.

    lenght - lenght of data.

    return:
    X - random numbers from the range of -10 to 10 
    y - array of 1 or 0. if X[i] < 0 y[i] = 0 else y[i] = 1
    """
    X = np.array(np.random.randint(-10, 10, lenght),dtype=np.float32)
    y = np.array([1. if i > 0 else 0. for i in X])
    return X,y

def generate_poisson_data(lambd, lenght=10):
    """
    Generation of data for fit poisson regression.

    lenght - lenght of data.

    lambd - Poisson distribution parameter.

    return:
    y - array of poisson distribution numbers
    X - matrix with shape(lenght,3) with random numbers of uniform distribution
    """
    y = np.random.poisson(lambd, size=lenght)
    X = np.random.uniform(0,np.exp(-lambd), lenght)
    for _ in range(2):
        X = np.vstack(( X, np.random.uniform(0, np.exp(-lambd), lenght)))
    return X.T,y

class MyBatch(Batch):
    """
    Main class
    """
    def __init__(self, index, *args, **kwargs):
        """
        Initialization of variable from parent class - Batch. 
        """    

        super().__init__(index, *args, **kwargs)
    
    @property
    def components(self):
        """
        Define componentis.
        """
        return 'x', 'y','W','b','loss'

    @action
    def generate(self, lenght = 10, ttype = "linear"):
        """
        Create batch by self.indices by rewrite self.x and self.y.
        lenght - size all data.

        ttype - name of using algorithm:
            * 'linear' for linear regression. (default)
            * 'logistic' for logistic regression.
            * 'poisson'  for poisson regression.

        return: self
        """
    
        if self.x == None or self.y == None:
            self = self.load(lenght,ttype)
            
        self.x, self.y = self.x[self.indices], self.y[self.indices]
        return self
    
    @action
    def load(self, lenght = 10, ttype = 'linear', lambd = 1):
        """
        Generate data for ttype-algorihm.

        lenght - size all data.

        ttype - name of using algorithm:
            * 'linear' for linear regression. (default)
            * 'logistic' for logistic regression.
            * 'poisson'  for poisson regression.

        return: self
        """
    
        if ttype == 'poisson':
            exec('self.x, self.y = generate_{}_data(lambd,lenght)'.format(ttype,lambd))
        else:
            exec('self.x, self.y = generate_{}_data(lenght)'.format(ttype))
        return self
    
    @model()
    def linear_model():
        """
        Function with graph of linear regression. 

        return:
        array with shape = (3,2)
        X - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        W - slope coefficient of straight line.
        b - bias.
        """
        
        X = tf.placeholder(name='input',dtype=tf.float32)
        y = tf.placeholder(name='true_y',dtype=tf.float32)
        
        W = tf.Variable(np.random.randint(-1,1,size=1),name='weight',dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1,1),dtype=tf.float32)

        predict = tf.multiply(W,X,name='output') + b
        loss = tf.reduce_mean(tf.square(predict - y))

        optimize = tf.train.GradientDescentOptimizer(learning_rate = 0.007)
        train = optimize.minimize(loss)

        return [[X, y],[train, loss],[W,b]]
    
    @action(model='linear_model')
    def train_linear_model(self, model, session):
        """
        Train linear regression.

        model - fit funtion. In this case it's linear_model.
        
        session - tensorflow session.

        return self.
        """
       
        X,y = model[0]
        optimizer, cost = model[1]
        params = model[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={X:self.x, y: self.y})
        self.W = params[0][0]
        self.b = params[1]
        self.loss = loss
        return self
    
    @model()
    def logistic_model():
        """
        Function with graph of logistic regression. 

        return:
        array with shape = (3,2(3))
        X - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        predict - model prediction.
        W - slope coefficient of straight line.
        b - bias.
        """
        X = tf.placeholder(name='input',dtype=tf.float32)
        y = tf.placeholder(name='true_y',dtype=tf.float32)

        W = tf.Variable(np.random.randint(-1,1,size=1),name='weight',dtype=tf.float32)
        b = tf.Variable(np.random.randint(-1,1),dtype=tf.float32)

        predict = tf.sigmoid(tf.multiply(W,X,name='output') + b)
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y,logits=predict))

        optimize = tf.train.AdamOptimizer(learning_rate = 0.005)
        train = optimize.minimize(loss)
        return [[X, y],[train, loss, predict],[W,b]]

    @action(model='logistic_model')
    def train_logistic_model(self, model, session, result, test):
        """
        Train logistic regression.

        model - fit funtion. In this case it's linear_model.
        
        session - tensorflow session.

        result - result of prediction.

        test - data to predict.

        return self.
        """

        X,y = model[0]
        optimizer, cost, predict = model[1]
        params = model[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={X:self.x, y: self.y})
        self.W = params[0][0]
        self.b = params[1]
        self.loss = loss
        result[:] = session.run([predict], feed_dict={X:test})[0]
        return self

    @model()
    def poisson_model():
        """
        Function with graph of poisson regression. 

        return:
        array with shape = (3,2(3))
        X - data.
        y - answers to data.
        train - function - optimizer.
        loss - quality of model.
        predict - model prediction.
        W - array of weights.
        """
        X = tf.placeholder(name='input',shape = [None,3],dtype=tf.float32)
        y = tf.placeholder(name='true_y',dtype=tf.float32)

        W = tf.Variable(np.random.randint(-1,1,size=3).reshape(3,1),name='weight',dtype=tf.float32)

        predict = tf.exp(tf.matmul(X,W))
        loss = tf.reduce_sum(tf.nn.log_poisson_loss(y,predict))
        optimize = tf.train.AdamOptimizer(learning_rate = 0.005)
        train = optimize.minimize(loss)
        return [[X, y],[train, loss, predict],[W]]
    
    @action(model='poisson_model')
    def train_poisson_model(self, model, session, result, test):
        """
        Train poisson regression.

        model - fit funtion. In this case it's linear_model.
        
        session - tensorflow session.

        return self.
        """
        X,y = model[0]
        optimizer, cost, predict = model[1]
        params = model[2]
        _, loss, params = session.run([optimizer, cost, params], feed_dict={X:self.x, y: self.y})
        self.W = params[0]
        self.loss = loss
        result[:] = session.run([predict], feed_dict={X:test})[0]
        return self