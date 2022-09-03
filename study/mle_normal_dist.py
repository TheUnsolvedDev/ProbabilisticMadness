import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	mu_true = tf.constant(2.5)
	sigma_true = tf.constant(0.5)

	grade = tfp.distributions.Normal(loc = mu_true,scale = sigma_true)
	dataset = grade.sample(100)

	print(dataset)

	mu_mle = tf.reduce_mean(dataset)
	sigma_mle = tf.math.sqrt(tf.reduce_mean((dataset - mu_mle)**2))
	print(mu_mle,sigma_mle)