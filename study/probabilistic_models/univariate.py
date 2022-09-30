import tensorflow as yf
import tensorflow_probability as tfp
from torch import bernoulli

if __name__ == '__main__':
    normal = tfp.distributions.Normal(loc=0., scale=1.)
    print(normal)
    print("Single sample", normal.sample())
    print("Multiple sample", normal.sample(3))

    print(normal.prob(0.5))
    print(normal.log_prob(0.5))

    bernoulli = tfp.distributions.Bernoulli(probs=0.7)
    bernoulli = tfp.distributions.Bernoulli(logits=0.847)
    print(bernoulli.sample(3))
    print(bernoulli.prob(1))
    print(bernoulli.log_prob(1))

    bernoulli = tfp.distributions.Bernoulli(probs=[0.4, 0.5])
    print(bernoulli)
    print(bernoulli.sample(3))

    print(bernoulli.prob([1, 1]))
    print(bernoulli.log_prob([1, 1]))
