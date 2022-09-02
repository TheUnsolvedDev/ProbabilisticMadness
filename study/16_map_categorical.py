
import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
	thetas_true = [0.2,0.3,0.5]
	weather = tfp.distributions.Categorical(probs=thetas_true)
	dataset = weather.sample(365)

	print(dataset)

	weather_one_hot = tfp.distributions.OneHotCategorical(probs=thetas_true)
	dataset = weather_one_hot.sample(365)
	n_samples = 365 
	n_per_state = tf.reduce_sum(dataset,axis = 0)
	print(n_per_state)

	thetas_mle = n_per_state/n_samples
	print(thetas_mle)

	alphas = tf.convert_to_tensor([20.0,30.0,60.0])
	n_per_state = tf.cast(n_per_state,tf.float32)

	thetas_map = (n_per_state + alphas - 1)/(n_samples - 3+tf.reduce_sum(alphas))
	print(thetas_map)

	dataset_corrupt = dataset.numpy()
	dataset_corrupt[200:,:] = [1,0,0]

	print(dataset_corrupt)

	n_per_state_corrupt = tf.reduce_sum(dataset_corrupt,axis = 0)
	print(n_per_state_corrupt)

	thetas_mle_corrupt = n_per_state_corrupt/n_samples
	print(thetas_mle_corrupt)

	n_per_state_corrupt = tf.cast(n_per_state_corrupt,tf.float32)
	thetas_map_corrupt = (n_per_state_corrupt + alphas - 1)/(n_samples - 3+tf.reduce_sum(alphas))
	print(thetas_map_corrupt)