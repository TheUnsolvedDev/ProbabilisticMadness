import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 

if __name__ == '__main__':
    weather_true = tfp.distributions.Bernoulli(probs = 0.8,name = 'weather_true')
    true_samples = weather_true.sample(100)
    
    theta_fit = tfp.util.TransformedVariable(0.6,bijector = tfp.bijectors.SoftClip(low = 0.0,high = 1.0),name = 'theta')
    neg_log_like = lambda : -tf.reduce_sum(weather_true.log_prob(true_samples))
    
    print(tfp.math.minimize(loss_fn = neg_log_like,optimizer = tf.optimizers.Adam(),num_steps = 1000))
    
    print('Auto diff vs calculated \n================================')
    print(theta_fit.numpy())
    print(tf.reduce_mean(tf.cast(true_samples,dtype = tf.float32)))