from keras.saving import experimental
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
from sklearn import preprocessing


def get_data():
    data = pd.read_csv('train.csv')
    x_train = np.array((np.array(data)
                        [:, :-3]+1)/2, dtype=np.float32).reshape(-1, 70, 8)
    y_train = np.array(data)[:, -1]
    label_encoder = preprocessing.LabelEncoder()
    onehot_encoder = preprocessing.OneHotEncoder()
    y_train = np.array(label_encoder.fit_transform(y_train))
    y_train = tf.one_hot(y_train, depth=6).numpy()

    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.uint8)
    return x_train, y_train


def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)


def model():
    def divergence(q, p, _): return tfp.distributions.kl_divergence(q, p)

    mod = tf.keras.models.Sequential([
        tfp.layers.Convolution1DReparameterization(16, 7, activation='relu', input_shape=(70, 8),  # Independent Normal Distribution
                                                   kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                   kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                                         is_singular=False),
                                                   kernel_divergence_fn=divergence,
                                                   bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                                   bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                                                         is_singular=False),
                                                   bias_divergence_fn=divergence
                                                   ),
        tf.keras.layers.MaxPool1D(16),
        tf.keras.layers.Flatten(),
        tfp.layers.DenseReparameterization(tfp.layers.OneHotCategorical.params_size(6), activation=None,
                                           kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                           kernel_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                             is_singular=False),
                         kernel_divergence_fn=divergence,
                         bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                         bias_posterior_fn=tfp.layers.default_mean_field_normal_fn(
                             is_singular=False),
                         bias_divergence_fn=divergence
                         ),
        tfp.layers.OneHotCategorical(6)
    ])
    return mod


def main():
    mod = model()
    mod.summary()
    mod.compile(
        loss=nll,
        optimizer='adam',
        metrics=['accuracy'])
    x_train, y_train = get_data()
    mod.fit(x_train, y_train, epochs=500)
    mod.evaluate(x_train, y_train)


if __name__ == '__main__':
    main()
