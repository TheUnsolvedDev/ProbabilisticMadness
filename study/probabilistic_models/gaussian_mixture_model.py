import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    pi_cluster = tf.constant([0.4, 0.6])
    mus = tf.constant([2.8, 4.8])
    sigmas = tf.constant([0.6, 0.3])

    cluster_distribution = tfp.distributions.Categorical(
        probs=pi_cluster, name='cluster')
    print(cluster_distribution)
    print(cluster_distribution.sample(10))

    grade_distribution = tfp.distributions.Normal(loc=mus, scale=sigmas)
    print(grade_distribution)
    print(grade_distribution.sample(10))

    gaussian_mixture = tfp.distributions.MixtureSameFamily(
        mixture_distribution=cluster_distribution, components_distribution=grade_distribution)
    print(gaussian_mixture.sample(10))

    x = np.linspace(0, 6, 100)
    y = gaussian_mixture.prob(x).numpy()

    plt.plot(x, y)
    plt.show()
