import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 
import matplotlib.pyplot as plt 

if __name__ == "__main__":
    theta_weather = tfp.distributions.Beta(8.0,3.0)
    data = theta_weather.sample(100)
    
    print(data)
    
    weather = tfp.distributions.Bernoulli(probs = theta_weather.sample())
    weather_data = weather.sample(100)
    print(weather_data)