
import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def sample_categorical(theta,n_samples):
	n_class = len(theta)
	thetas_cumsum = np.cumsum(theta)
	dataset_x = np.random.rand(n_samples)

	lower_than_limits = [dataset_x <= limit for limit in thetas_cumsum]
	class_matrix = [i*np.ones(n_samples) for i in range(n_class)]
	dataset_w = np.select(lower_than_limits,class_matrix)
	return dataset_w

if __name__ == '__main__':
	thetas = [0.2,0.3,0.5]
	psi = np.cumsum(thetas)

	print(psi)

	n_samples = 10
	dataset_x = np.random.rand(n_samples)
	print(dataset_x)

	condlist = [dataset_x <= limit for limit in psi]
	print(condlist)
	choicelist = [i*np.ones(n_samples) for i in range(len(thetas))]
	dataset_w = np.select(condlist,choicelist)

	print(dataset_w)

	print(sample_categorical(thetas,n_samples))