import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def model_1(weather_prob, weather_to_happiness):
    weather = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Bernoulli(probs=weather_prob, name='weather'))
    happiness = yield tfp.distributions.Bernoulli(probs=weather_to_happiness[weather], name='happiness')


def model_2(weather_prob, weather_to_happiness, n_days):
    weather = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Bernoulli(probs=weather_prob*tf.ones(n_days), name='weather'))
    happiness = yield tfp.distributions.Bernoulli(
        probs=tf.where(weather == 1, weather_to_happiness[1], weather_to_happiness[0]), name='happiness')


def model_3(alpha, beta, weather_to_happiness, n_days):
    theta = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Beta(alpha, beta, name='theta_weather'))
    weather = yield tfp.distributions.Bernoulli(probs=theta*tf.ones(n_days), name='weather')
    happiness = yield tfp.distributions.Bernoulli(
        probs=tf.where(weather == 1, weather_to_happiness[1], weather_to_happiness[0], name='happiness'))


if __name__ == '__main__':
    theta_weather = tf.constant(0.8)
    theta_happiness = tf.constant([0.7, 0.95])

    joint1 = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: model_1(theta_weather, theta_happiness))
    print(joint1)
    print(joint1.sample(10))

    dataset = joint1.sample(10)
    print('Likelyhood:', tf.reduce_prod(joint1.prob(dataset)))

    joint2 = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: model_2(theta_weather, theta_happiness, 10))
    print(joint2)
    print(joint2.sample(10))
    print('Likelyhood:', joint2.prob(dataset))

    alpha = tf.constant(10.0)
    beta = tf.constant(5.0)

    joint3 = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: model_3(alpha, beta, theta_happiness, 10))
    print(joint3)
    print(joint3.sample(10))
