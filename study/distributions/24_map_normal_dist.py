import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	mu_true = 50.0
	sigma_true = 1.0

	diameter = tfp.distributions.Normal(loc = mu_true,scale = sigma_true)
	dataset = diameter.sample(100)

	print('dataset',dataset)

	mu_fix = mu_true
	sigma_mle = tf.sqrt(tf.reduce_mean((dataset - mu_fix)**2))
	print('sigma_mle',sigma_mle)

	alpha_0 = 5.0
	beta_0 = 4.0
	sigma_map = tf.sqrt((beta_0 + 0.5*tf.reduce_sum((dataset - mu_fix)**2))/(alpha_0 + 100/2 - 1))
	print('sigma_map',sigma_map)

	alpha_N = alpha_0 + 100/2
	beta_N = beta_0 + 0.5*tf.reduce_sum((dataset - mu_fix)**2)
	print(alpha_N,beta_N)

	tau_posterior = tfp.distributions.Gamma(alpha_N,beta_N)
	print('mode',tau_posterior.mode())
	print(tf.sqrt(1/tau_posterior.mode()))

	dataset_corrupt = dataset.numpy()
	dataset_corrupt[60:] = 55.0

	sigma_mle_corrupt = tf.sqrt(tf.reduce_mean((dataset_corrupt - mu_fix)**2))
	print('sigma_mle_corrupt',sigma_mle_corrupt)

	sigma_map_corrupt = tf.sqrt((beta_0 + 0.5*tf.reduce_sum((dataset_corrupt - mu_fix)**2))/(alpha_0 + 100/2 - 1))
	print('sigma_map_corrupt',sigma_map_corrupt)