import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	weather_yesterday = tfp.distributions.Deterministic(1)
	dataset = weather_yesterday.sample(10)

	print(dataset)
	print(weather_yesterday.prob(1))