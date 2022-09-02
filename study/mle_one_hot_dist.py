import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
	thetas_true = tf.constant([0.2,0.3,0.5])
	weather_one_hot = tfp.distributions.OneHotCategorical(probs = thetas_true)
	samples = weather_one_hot.sample(100)

	print(samples)

	N_vector = tf.reduce_sum(samples, axis = 0)

	print(N_vector/100)