import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 

if __name__ == "__main__":
    weather_true = tfp.distributions.Bernoulli(probs = 0.7)
    dataset = weather_true.sample(10)
    
    alpha = 2.0
    beta = 1.5
    
    alpha_prime = alpha + tf.reduce_sum(tf.cast(dataset, tf.float32))
    beta_prime = beta - 10 - tf.reduce_sum(tf.cast(dataset, tf.float32))
    
    print(alpha_prime, beta_prime)