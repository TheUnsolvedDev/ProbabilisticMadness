import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np


def season_weather_model(season_probs, season_to_weather_probs):
    season = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Categorical(probs=season_probs, name='season'))
    weather = yield tfp.distributions.Bernoulli(probs=season_to_weather_probs[season], name='weather')


if __name__ == '__main__':
    thetas_season = tf.constant([0.25, 0.25, 0.25, 0.25])
    thetas_weather = tf.constant([0.4, 0.9, 0.5, 0.3])

    mixture_joint = tfp.distributions.JointDistributionCoroutineAutoBatched(
        lambda: season_weather_model(thetas_season, thetas_weather))
    print(mixture_joint)
    print(mixture_joint.sample(10))

    print(mixture_joint.prob([0,0]))
    print(mixture_joint.prob([3,0]))

    print(tf.tensordot(thetas_season,thetas_weather,axes = 1))

    # marginal case
    mixture_joint2 = tfp.distributions.Mixture(
        cat = tfp.distributions.Categorical(probs = thetas_season),
        components = [
            tfp.distributions.Bernoulli(probs = thetas_weather[0]),
            tfp.distributions.Bernoulli(probs = thetas_weather[1]),
            tfp.distributions.Bernoulli(probs = thetas_weather[2]),
            tfp.distributions.Bernoulli(probs = thetas_weather[3]),
            ])
    print(mixture_joint2)
    print(mixture_joint2.sample(10))

    mixture_joint3 = tfp.distributions.MixtureSameFamily(
        mixture_distribution = tfp.distributions.Categorical(probs = thetas_season),
        components_distribution = tfp.distributions.Bernoulli(probs = thetas_weather))
    print(mixture_joint3)
    print(mixture_joint3.sample(10))
