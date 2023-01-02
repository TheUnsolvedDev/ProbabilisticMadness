import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np

gradient_estimator = tfp.vi.GradientEstimators.SCORE_FUNCTION

if __name__ == '__main__':
    weather_A = tfp.distributions.Bernoulli(probs=0.8)
    weather_B = tfp.distributions.Bernoulli(probs=0.7)

    div = tfp.distributions.kl_divergence(weather_A, weather_B)

    datasetA = weather_A.sample(36500)
    datasetB = weather_B.sample(36500)
    print(datasetB)
    print(div)

    print(tf.reduce_mean(tf.math.log(
        weather_A.prob(datasetA)/weather_B.prob(datasetA))))
    print(tf.reduce_mean(weather_A.log_prob(
        datasetA) - weather_B.log_prob(datasetA)))
    
    print(tfp.vi.monte_carlo_variational_loss(
        weather_B.log_prob, weather_A, 36500,gradient_estimator = gradient_estimator))
