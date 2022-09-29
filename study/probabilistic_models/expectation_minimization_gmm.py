import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

from tqdm import tqdm


def EM(dataset, n_classes, n_iterations, random_seed):
	n_samples = dataset.shape[0]
	np.random.seed(random_seed)

	# initial guesses
	mus = np.random.rand(n_classes)
	sigmas = np.random.rand(n_classes)
	class_probs = np.random.dirichlet(np.ones(n_classes))

	for en_it in tqdm(range(n_iterations)):
		responsibilities = tfp.distributions.Normal(loc=mus, scale=sigmas).prob(
			dataset.reshape(-1, 1)).numpy() * class_probs
		responsibilities /= tf.linalg.norm(responsibilities,
			axis=1, ord=1, keepdims=True)
		class_responsibilities = np.sum(responsibilities, axis=0)

		for c in range(n_classes):
			class_probs[c] = class_responsibilities[c] / n_samples
			mus[c] = np.sum(responsibilities[:, c] * dataset) / \
			                class_responsibilities[c]
			sigmas[c] = np.sqrt(
				np.sum(responsibilities[:, c] * (dataset - mus[c])
				       ** 2) / class_responsibilities[c]
            )

	return class_probs, mus, sigmas

def main():
	class_probs_true = tf.constant([0.6,0.4])
	mus_true = tf.constant([2.5,4.8])
	sigmas_true = tf.constant([0.6,0.3])

	random_seed = 42
	n_samples= 1000
	n_iterations = 30
	n_classes = 2

	gmm = tfp.distributions.MixtureSameFamily(
		mixture_distribution = tfp.distributions.Categorical(probs = class_probs_true),
		components_distribution = tfp.distributions.Normal(
			loc = mus_true,scale = sigmas_true
			)
		)
	dataset = gmm.sample(n_samples).numpy()
	print(dataset)

	class_prob,mus,sigmas = EM(dataset,n_classes,n_iterations,random_seed)
	print(class_prob,mus,sigmas)

if __name__ == '__main__':
	main()
