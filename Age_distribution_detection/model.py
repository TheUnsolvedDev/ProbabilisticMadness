import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from param import *


def model_cnn():
    inputs = tf.keras.layers.Input(IMAGE_SIZE)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(
        32, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        64, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        128, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(
        256, kernel_size=(3, 3), activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    features = tf.keras.layers.BatchNormalization()(x)
    distribution_params = tf.keras.layers.Dense(
        units=2, activation='relu')(features)
    outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)
    return model


if __name__ == '__main__':
    model = model_cnn()
    a = np.expand_dims(np.ones(IMAGE_SIZE)*255, axis=0)
    model.summary()
    print(model(a))
