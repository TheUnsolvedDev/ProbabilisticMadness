import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    alpha = 3.0
    beta = 2.0

    tau = tfp.distributions.Gamma(alpha, beta)
    print(tau.sample(100))
    print('Mode:',tau.mode())

    print(tau.prob(1))