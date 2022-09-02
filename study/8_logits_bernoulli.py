import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 
import matplotlib.pyplot as plt 

if __name__ == '__main__':
    theta = 0.8
    psi = tf.math.log(theta/(1-theta))
    bern_theta = tfp.distributions.Bernoulli(probs = theta)
    bern_logit = tfp.distributions.Bernoulli(logits = psi)
    
    print(bern_theta.mean())
    print(bern_logit.mean())
    print(tf.math.sigmoid(psi))