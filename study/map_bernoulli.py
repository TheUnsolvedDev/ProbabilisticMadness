import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf

if __name__ == "__main__":
    theta_weather_true = 0.8
    weather = tfp.distributions.Bernoulli(probs=0.8)

    samples = weather.sample(100)
    print(samples)

    theta_weather_mle = tf.reduce_mean(
        tf.cast(samples, dtype=tf.float32), axis=0)
    print(theta_weather_mle)

    alpha = 30.0
    beta = 10.0
    theta_weather_map = (tf.reduce_sum(
        tf.cast(samples, dtype=tf.float32), axis=0) + alpha - 1)/(len(samples) + alpha + beta - 2)
    print(theta_weather_map)
    
    corrupt_data = tf.zeros(len(samples),dtype = tf.float32)
    corrupt_data = tf.concat([corrupt_data, tf.cast(samples, dtype=tf.float32)],axis=0)
    
    print(corrupt_data)
    
    theta_weather_mle = tf.reduce_mean(
        tf.cast(corrupt_data, dtype=tf.float32), axis=0)
    print(theta_weather_mle)

    alpha = 30.0
    beta = 10.0
    theta_weather_map = (tf.reduce_sum(
        tf.cast(corrupt_data, dtype=tf.float32), axis=0) + alpha - 1)/(len(corrupt_data) + alpha + beta - 2)
    print(theta_weather_map)
