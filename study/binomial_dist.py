import silence_tensorflow.auto
import tensorflow_probability as tfp 
import tensorflow as tf 
from scipy import special 

if __name__ == "__main__":
    n_good_days = tfp.distributions.Binomial(total_count = 7,probs = 0.7)
    
    print(n_good_days,n_good_days.sample())
    print(n_good_days.sample(10))
    
    print(n_good_days.prob(6))
    print(n_good_days.prob(3))