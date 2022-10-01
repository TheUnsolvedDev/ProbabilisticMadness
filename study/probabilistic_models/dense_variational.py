import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_probability as tfp

'''
y_i = alpha + beta*x_i + elpsilon_i
elpsilon_i ~ N(0,sigma^2)
alpha ~ N(0,1)
beta ~ N(0,1)
'''


def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = tf.keras.models.Sequential(
        [tfp.layers.DistributionLambda(
             lambda t: tfp.distributions.MultivariateNormalDiag(
                 loc=tf.zeros(n),
                 scale_diag=tf.ones(n))
        )
            ])
    return prior_model


def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    posterior_model = tf.keras.models.Sequential(
        [tfp.layers.VariableLayer(
             tfp.layers.MultivariateNormalTriL.params_size(n),
             dtype=dtype),
         tfp.layers.MultivariateNormalTriL(n)])
    return posterior_model


if __name__ == '__main__':
    x_train = np.linspace(-1, 1, 1000)[:, np.newaxis]
    y_train = x_train + 0.3*np.random.randn(1000)[:, np.newaxis]
    plt.scatter(x_train, y_train, alpha=0.4)
    plt.show()

    model = tf.keras.models.Sequential(
        [tfp.layers.DenseVariational(
         input_shape=(1,),
         units=1, make_prior_fn=prior, make_posterior_fn=posterior,
         kl_weight=1 / x_train.shape[0],
         kl_use_exact=True)])
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=tf.keras.optimizers.RMSprop(
            learning_rate=0.005))
    model.summary()
    model.fit(x_train,y_train,epochs=500)

    dummy_input = np.array([[0]])
    model_prior = model.layers[0]._prior(dummy_input)
    model_posterior = model.layers[0]._posterior(dummy_input)
    print('Prior mean:',model_prior.mean().numpy())
    print('Prior Variance:',model_prior.variance().numpy())
    print('Posterior mean:',model_posterior.mean().numpy())
    print('Posterior covariance:',model_posterior.covariance().numpy())
