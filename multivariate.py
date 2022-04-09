import tensorflow as tf
import tensorflow_probability as tfp

if __name__ == '__main__':
    normal = tfp.distributions.MultivariateNormalDiag(loc = [-1,0.5],scale_diag = [1,1.5])
    print(normal)
    print(normal.sample(3))
    print(normal.log_prob([-0.2,1.8]))
    
    batched_normal = tfp.distributions.Normal(loc = [-1,0.5],scale = [1,1.5])
    print(batched_normal)
    print(batched_normal.sample(3))
    print(batched_normal.log_prob([-0.2,1.8]))