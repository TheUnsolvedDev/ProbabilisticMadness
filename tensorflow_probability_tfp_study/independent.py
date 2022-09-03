import tensorflow as tf
import tensorflow_probability as tfp

if __name__ == '__main__': 
    normal = tfp.distributions.MultivariateNormalDiag(loc = [-1,0.5],scale_diag = [1,1.5])
    batched_normal = tfp.distributions.Normal(loc = [-1,0.5],scale = [1,1.5])
    
    independent_normal = tfp.distributions.Independent(batched_normal,reinterpreted_batch_ndims = 1)
    print(independent_normal)
    print(independent_normal.log_prob([-0.2,1.8]))