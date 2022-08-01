import numpy
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from param import *

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.simplefilter(action='ignore',category=FutureWarning)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)



def model_cnn():
    inputs = tf.keras.layers.Input(IMAGE_SIZE)
    x = tf.keras.layers.Lambda(lambda x:x/255)(inputs)
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
