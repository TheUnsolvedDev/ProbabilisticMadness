import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
	thetas_weather = tfp.distributions.Dirichlet([1.5,2.0,5.0])
	print(thetas_weather)

	dataset = thetas_weather.sample(100)
	print(dataset)

	print(thetas_weather.prob([0.2,0.3,0.5]))

	print(tf.math.exp(tf.math.lbeta([1.5,2.0,5.0])))