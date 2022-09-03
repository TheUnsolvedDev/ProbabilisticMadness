import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    mu_true = 2.5
    sigma_true = 0.7

    grade = tfp.distributions.Normal(loc=mu_true, scale=sigma_true)
    dataset = grade.sample(100)

    mu_mle = tf.reduce_mean(dataset)
    sigma_fix = sigma_true

    mu_0 = 2.3
    sigma_0 = 0.2

    mu_N = (sigma_fix**2 * mu_0 + sigma_0**2 * tf.reduce_sum(dataset)) / \
        (sigma_fix**2 + 100*sigma_0**2)
    print(mu_N)

    mu_map = mu_N
    sigma_N = (sigma_0*sigma_fix)/tf.math.sqrt(sigma_fix**2 + 100*sigma_0**2)

    print(sigma_N)

    mu_posterior = tfp.distributions.Normal(loc = mu_N,scale = sigma_N)
    print(mu_posterior.mode())

    dataset_corrupt = dataset.numpy()
    dataset_corrupt[:70] -= 1

    print(dataset_corrupt)

    mu_mle_corrupt = tf.reduce_mean(dataset_corrupt)
    mu_map_corrupt= (sigma_fix**2 * mu_0 + sigma_0**2 * tf.reduce_sum(dataset_corrupt)) / \
        (sigma_fix**2 + 100*sigma_0**2)
    print(mu_mle_corrupt,mu_map_corrupt)

