import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework.ops import colocate_with
import tensorflow_probability as tfp

# deterministic model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

# insertion of uncertainty
model_prob = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(1, input_shape=(1,)),
     tfp.layers.DistributionLambda(
         lambda t: tfp.distributions.Normal(loc=t, scale=1))
     ])

model_prob2 = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(2, input_shape=(1,)),
     tfp.layers.DistributionLambda(
         lambda t: tfp.distributions.Normal(
             loc=t[..., : 1],
             scale=tf.math.softplus(t[..., 1:])))
     ])

model_prob3 = tf.keras.models.Sequential(
    [tf.keras.layers.Dense(8, input_shape=(1,),
                           activation='sigmoid'),
     tf.keras.layers.Dense(
         tfp.layers.IndependentNormal.params_size(event_shape=1)),
     tfp.layers.IndependentNormal(event_shape=1)])


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.01)

if __name__ == '__main__':
    x_train = np.linspace(-1, 1, 100)[:, np.newaxis]
    y_train = x_train + 0.3*np.random.randn(100)[:, np.newaxis]

    plt.scatter(x_train, y_train, label='data')
    plt.xlabel('x')
    plt.ylabel('y')

    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=optimizer)
    model.summary()
    model.fit(x_train, y_train, epochs=500)

    plt.plot(
        x_train,
        model.predict(x_train),
        color='r',
        alpha=0.8,
        label='model')
    plt.legend()
    plt.show()

    model_prob.compile(loss=nll, optimizer=optimizer)
    model_prob.summary()
    model_prob.fit(x_train, y_train, epochs=500)

    y_model = model_prob(x_train)
    y_sample = y_model.sample()
    y_hat = y_model.mean()
    y_sd = y_model.stddev()

    y_hat_m2sd = y_hat - 2*y_sd
    y_hat_p2sd = y_hat + 2*y_sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax1.scatter(x_train, y_train, alpha=0.4, label='data')
    ax1.scatter(x_train, y_sample, alpha=0.4, label='model_sample', color='r')
    ax1.legend()
    ax2.scatter(x_train, y_train, alpha=0.4, label='data')
    ax2.plot(x_train, y_hat, color='r', alpha=0.8, label='model $\\mu$')
    ax2.plot(
        x_train,
        y_hat_m2sd,
        color='g',
        alpha=0.8,
        label='model $\\mu \\pm 2 \\sigma$')
    ax2.plot(x_train, y_hat_p2sd, color='g', alpha=0.8)
    ax2.legend()

    plt.show()

    model_prob2.compile(loss=nll, optimizer=optimizer)
    model_prob2.summary()
    model_prob2.fit(x_train, y_train, epochs=500)

    y_model = model_prob2(x_train)
    y_sample = y_model.sample()
    y_hat = y_model.mean()
    y_sd = y_model.stddev()

    y_hat_m2sd = y_hat - 2*y_sd
    y_hat_p2sd = y_hat + 2*y_sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax1.scatter(x_train, y_train, alpha=0.4, label='data')
    ax1.scatter(x_train, y_sample, alpha=0.4, label='model_sample', color='r')
    ax1.legend()
    ax2.scatter(x_train, y_train, alpha=0.4, label='data')
    ax2.plot(x_train, y_hat, color='r', alpha=0.8, label='model $\\mu$')
    ax2.plot(
        x_train,
        y_hat_m2sd,
        color='g',
        alpha=0.8,
        label='model $\\mu \\pm 2 \\sigma$')
    ax2.plot(x_train, y_hat_p2sd, color='g', alpha=0.8)
    ax2.legend()

    plt.show()

    x_train = np.linspace(-1, 1, 1000)[..., np.newaxis]
    y_train = np.power(
        x_train, 3)+0.1*(2+x_train)*np.random.randn(1000)[:, np.newaxis]
    plt.scatter(x_train, y_train, alpha=0.1)
    plt.show()

    model_prob3.compile(loss=nll, optimizer=optimizer)
    model_prob3.summary()
    model_prob3.fit(x_train, y_train, epochs=500)

    y_model = model_prob3(x_train)
    y_sample = y_model.sample()
    y_hat = y_model.mean()
    y_sd = y_model.stddev()

    y_hat_m2sd = y_hat - 2*y_sd
    y_hat_p2sd = y_hat + 2*y_sd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)
    ax1.scatter(x_train, y_train, alpha=0.4, label='data')
    ax1.scatter(x_train, y_sample, alpha=0.4, label='model_sample', color='r')
    ax1.legend()
    ax2.scatter(x_train, y_train, alpha=0.4, label='data')
    ax2.plot(x_train, y_hat, color='r', alpha=0.8, label='model $\\mu$')
    ax2.plot(
        x_train,
        y_hat_m2sd,
        color='g',
        alpha=0.8,
        label='model $\\mu \\pm 2 \\sigma$')
    ax2.plot(x_train, y_hat_p2sd, color='g', alpha=0.8)
    ax2.legend()

    plt.show()
