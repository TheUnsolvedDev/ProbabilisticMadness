import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt

if __name__ == '__main__':
    thetas_true = tf.constant([0.2, 0.3, 0.5])
    weather = tfp.distributions.Categorical(probs=thetas_true)
    dataset = weather.sample(100)

    print(dataset)

    n_cloudy = tf.reduce_sum(tf.cast(dataset == 0, dtype=tf.int16))
    n_rainy = tf.reduce_sum(tf.cast(dataset == 1, dtype=tf.int16))
    n_sunny = tf.reduce_sum(tf.cast(dataset == 2, dtype=tf.int16))
    print(n_cloudy)

    thetas_mle = tf.constant(
        [n_cloudy.numpy()/100, n_rainy.numpy()/100, n_sunny.numpy()/100])
    print(thetas_mle)
