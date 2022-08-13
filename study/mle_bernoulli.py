import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 

if __name__ == '__main__':
    weather = tfp.distributions.Bernoulli(probs = 0.8,name = 'weather')
    dataset = weather.sample(100)

    print(dataset)

    for i in range(10):
        mean = tf.reduce_mean(tf.cast(weather.sample(100),dtype = tf.float32))
        print(mean)

    print('Here we can see that the mean deviates')
    
    for i in range(10):
        mean = tf.reduce_mean(tf.cast(weather.sample(10000),dtype = tf.float32))
        print(mean)
    print('Here the prob is less deviated')
    
