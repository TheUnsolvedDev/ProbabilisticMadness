import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 

def happiness_model():
    weather = yield tfp.distributions.JointDistributionCoroutine.Root(
        tfp.distributions.Bernoulli(probs = 0.3,name = 'weather')
    )
    weather_to_happiness = tf.constant([0.6,0.9])
    happiness = yield tfp.distributions.Bernoulli(
        probs = weather_to_happiness[weather],name = 'happiness'
    )
    
if __name__ == '__main__':
    model = tfp.distributions.JointDistributionCoroutineAutoBatched(happiness_model)
    print(model)
    
    print(model.sample(10))