import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


def Bayesian_approach():
    W = yield tfp.distributions.Normal(loc=tf.zeros([784, 10]), scale=tf.ones([784, 10]))
    B = yield tfp.distributions.Normal(loc=tf.zeros([10]), scale=tf.ones([10]))
    # B = yield tfp.distributions.Cauchy(loc=tf.zeros([10]),scale=tf.ones([10]))
    # W = yield tfp.distributions.Gamma(concentration=tf.ones([784,10]),rate=tf.ones([784,10]))
    # B = yield tfp.distributions.Gamma(concentration=tf.ones([10]),rate=tf.ones([10]))
    Y = yield tfp.distributions.Multinomial(total_count=1, probs=tf.nn.softmax(tf.matmul(train_X, W)+B))


model = tfp.distributions.JointDistributionSequentialAutoBatched(
    [tfp.distributions.Normal(loc=tf.zeros([784, 10]), scale=tf.ones([784, 10])),
     tfp.distributions.Normal(loc=tf.zeros([10]), scale=tf.ones([10])),
     lambda b, w: tfp.distributions.Multinomial(
         total_count=1, probs=tf.nn.softmax(tf.matmul(train_X, w)+b))
     ]
)

if __name__ == '__main__':
    (train_X, train_y), (test_X, test_y) = tf.keras.datasets.mnist.load_data()

    train_X = tf.cast(tf.reshape(
        train_X/255, shape=(-1, 784)), dtype=tf.float32)
    test_X = tf.cast(tf.reshape(test_X/255, shape=(-1, 784)), dtype=tf.float32)

    train_y = tf.one_hot(train_y, depth=10)
    test_y = tf.one_hot(test_y, depth=10)

    # model = tfp.distributions.JointDistributionCoroutineAutoBatched(
    #     Bayesian_approach)

    w, b, pred = model.sample()
    new_W = tf.Variable(tf.ones([784, 10]))
    new_B = tf.Variable(tf.ones([10]))

    loss = (lambda w,b: model.log_prob(w,b,train_y))
    tfp.math.minimize(lambda : -loss(new_W,new_B),500,tf.keras.optimizers.Adam())
    
    w_new,b_new,pred = model.sample(value=(new_W,new_B))
    print(np.sum(tf.argmax(pred,axis=1)==tf.argmax(train_y,axis=1))/len(train_y))
    
    test_new = tf.nn.softmax(tf.matmul(test_X,w_new)+b_new)
    print(np.mean(tf.argmax(test_new,axis=1)==tf.argmax(test_y,axis=1)))
    
    train_new = tf.nn.softmax(tf.matmul(train_X,w_new)+b_new)
    print(np.mean(tf.argmax(train_new,axis=1)==tf.argmax(train_y,axis=1)))