import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == '__main__':
    normal = tfp.distributions.MultivariateNormalDiag(
        loc=[-1, 0.5], scale_diag=[1, 1.5])
    batched_normal = tfp.distributions.Normal(loc=[-1, 0.5], scale=[1, 1.5])

    independent_normal = tfp.distributions.Independent(
        batched_normal, reinterpreted_batch_ndims=1)
    print(independent_normal)
    print(independent_normal.log_prob([-0.2, 1.8]))

    locs = [-1, 1]
    scales = [0.5, 1]
    batch_of_normal = tfp.distributions.Normal(loc=locs, scale=scales)
    t = np.linspace(-4, 4, 1000)
    densities = batch_of_normal.prob(np.repeat(t[:, np.newaxis], 2, axis=1))

    plt.plot(t, densities[:, 0], label='loc = {},scale = {}'.format(
        locs[0], scales[0]))
    plt.plot(t, densities[:, 1], label='loc = {},scale = {}'.format(
        locs[1], scales[1]))
    plt.ylabel('Probability Density')
    plt.xlabel('Value')
    plt.legend()
    plt.show()

    bivariate_normal_form_independent = tfp.distributions.Independent(
        batch_of_normal,
        reinterpreted_batch_ndims=1)
    print(bivariate_normal_form_independent)

    samples = bivariate_normal_form_independent.sample(10000)
    x1 = samples[:, 0]
    x2 = samples[:, 1]
    sns.jointplot(x = x1,y = x2, kind='kde',space = 0,color = 'b')
    plt.show()
