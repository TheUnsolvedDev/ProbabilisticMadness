import silence_tensorflow.auto
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import numpy as np


class Solution:
    def mini(self):
        normal = tfp.distributions.Normal(loc=0.1, scale=1.0)
        print(normal)
        print(normal.sample(10))
        print(normal.prob(0.5))
        print(normal.log_prob(0.5))

        bernoulli = tfp.distributions.Bernoulli(probs=0.7)
        print(bernoulli)
        print(bernoulli.sample(10))
        bernoulli = tfp.distributions.Bernoulli(logits=0.847)
        print(bernoulli)
        print(bernoulli.sample(10))
        print(bernoulli.prob(1))
        print(bernoulli.log_prob(1))
        bernoulli = tfp.distributions.Bernoulli(probs=[0.4, 0.5])
        print(bernoulli)
        print(bernoulli.sample(10))
        print(bernoulli.prob([1, 1]))

    def big(self):
        # normal = tfp.distributions.Normal(loc=0.1, scale=1.)
        # plt.hist(normal.sample(10000), bins=50,density=True)
        # plt.show()

        # exponential = tfp.distributions.Exponential(rate=1)
        # plt.hist(exponential.sample(10000), bins=50,density=True)
        # plt.show()

        bernoulli = tfp.distributions.Bernoulli(probs=0.8)
        print(bernoulli.sample(20))

        for k in [0, 0.5, 1, -1]:
            print('Prob result {} for k = {}'.format(bernoulli.prob(k), k))

        def my_bernoulli(p_success, k):
            return np.power(p_success, k)*np.power(1-p_success, 1-k)

        for k in [0, 0.5, 1, -1]:
            print('Prob result {} for k = {}'.format(
                my_bernoulli(p_success=0.8, k=k), k))

        bernoulli_batch = tfp.distributions.Bernoulli(
            probs=[0.1, 0.2, 0.3, 0.4, 0.5])
        print(bernoulli_batch.sample(10))


if __name__ == '__main__':
    sol = Solution()
    # sol.mini()
    sol.big()
