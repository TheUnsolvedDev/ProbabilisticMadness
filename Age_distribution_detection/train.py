
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

from param import *
from dataset import Dataset
from model import model_cnn


callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='loss', patience=7),
    tf.keras.callbacks.ModelCheckpoint(
        filepath='modelA*.h5', save_weights_only=True, monitor='loss', save_best_only=True),
    tf.keras.callbacks.TensorBoard(
        log_dir='./logs', histogram_freq=1, write_graph=True),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss', factor=0.1, patience=4, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0),
]


def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)


if __name__ == '__main__':
    model = model_cnn()
    data = Dataset()
    train, test = data.get_data()
    kl = negative_loglikelihood
    model.compile(loss=kl, optimizer='adam',
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])
    model.fit(train, epochs = EPOCHS,validation_data = test,callbacks = callbacks)

    model.load_weights('modelA*.h5')

    for elem in train.take(1):
        out = model(elem[0])
        labels = elem[1]

    prediction_distribution = out
    prediction_mean = prediction_distribution.mean().numpy().tolist()
    prediction_stdv = prediction_distribution.stddev().numpy()

    # The 95% CI is computed as mean ± (1.96 * stdv)
    upper = (prediction_mean + (1.96 * prediction_stdv)).tolist()
    lower = (prediction_mean - (1.96 * prediction_stdv)).tolist()
    prediction_stdv = prediction_stdv.tolist()

    for idx in range(20):
        print(
            f"Prediction mean: {round(prediction_mean[idx][0], 2)}, "
            f"stddev: {round(prediction_stdv[idx][0], 2)}, "
            f"95% CI: [{round(upper[idx][0], 2)} - {round(lower[idx][0], 2)}]"
            f" - Actual: {labels[idx]}"
        )
