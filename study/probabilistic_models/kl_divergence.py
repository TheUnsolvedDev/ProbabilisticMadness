import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def kl_divergence_bernoulli(prob1, prob2):
    return (1-prob1)*np.log((1-prob1)/(1-prob2)) + prob1*np.log(prob1/prob2)


if __name__ == '__main__':
    weather_A = tfp.distributions.Bernoulli(probs=0.8)
    weather_B = tfp.distributions.Bernoulli(probs=0.7)

    datasetA = weather_A.sample(365)
    datasetB = weather_B.sample(365)
    print(datasetB)

    print(kl_divergence_bernoulli(0.8,0.7))
    print(tfp.distributions.kl_divergence(weather_A, weather_B))
    print(tfp.distributions.kl_divergence(weather_B, weather_A))
