import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	weather_obs = tfp.distributions.Multinomial(probs = [0.2,0.3,0.5],total_count = 31)
	dataset = weather_obs.sample(10)
	print(dataset)

	print(weather_obs.prob([2,8,23]))