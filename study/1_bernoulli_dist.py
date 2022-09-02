import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 

if __name__ == "__main__":
    weather = tfp.distributions.Bernoulli(probs = 0.8,name = 'weather')
    dataset = weather.sample(10)
    
    print(dataset)
    print(weather)
    print(weather.prob(dataset))
    print(tf.reduce_prod(weather.prob(dataset)))