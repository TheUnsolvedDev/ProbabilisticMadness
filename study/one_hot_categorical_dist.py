import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf

if __name__ == "__main__":
    weather = tfp.distributions.Categorical(probs=[0.2, 0.3, 0.5])
    weather_one_hot = tfp.distributions.OneHotCategorical(probs=[0.2, 0.3, 0.5])
    print(weather,weather_one_hot)

    print(weather.sample(10))
    print(weather_one_hot.sample(10))
    print(weather.prob([2, 2, 2, 1, 2, 0]))