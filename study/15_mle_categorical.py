
import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
	thetas_true = [0.2,0.3,0.5]
	weather = tfp.distributions.Categorical(probs=thetas_true)
	dataset = weather.sample(365)
	print(dataset)

	n_cloudy = tf.reduce_sum(tf.cast(dataset == 0,tf.int16))
	n_rainy = tf.reduce_sum(tf.cast(dataset == 1,tf.int16))
	n_sunny = tf.reduce_sum(tf.cast(dataset == 2,tf.int16))
	print(n_cloudy,n_rainy,n_sunny)

	thetas_mle = tf.constant([n_cloudy.numpy()/365,n_rainy.numpy()/365,n_sunny.numpy()/365])
	print(thetas_mle)