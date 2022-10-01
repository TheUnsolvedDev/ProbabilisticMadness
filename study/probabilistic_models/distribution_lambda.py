import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib.pyplot as plt


def nll(y_true, y_pred):
    return - y_pred.log_prob(y_true)


# deterministic model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(input_shape=(1,), units=1, activation='sigmoid')
])

# probabilistic model
model_prob = tf.keras.Sequential(
    [tf.keras.layers.Dense(
         input_shape=(1,),
         units=1, activation='sigmoid'),
     tfp.layers.DistributionLambda(
         lambda t: tfp.distributions.Bernoulli(probs=t),
         convert_to_tensor_fn=tfp.distributions.Distribution.sample)
    ])


if __name__ == '__main__':
    x_plot = np.linspace(-5, 5, 1000)[:, np.newaxis]
    y_train = model.predict(x_plot)
    plt.scatter(x_plot, model_prob.predict(x_plot), alpha=0.4)
    plt.plot(x_plot, 1/(1+np.exp(-x_plot)), color='r', alpha=0.8)
    plt.show()

    x = np.array([[0]])
    y_model = model(x)

    model_prob.compile(loss=nll, optimizer='adam')

    epochs = [0]
    training_weights = [model_prob.weights[0].numpy()[0, 0]]
    training_bias = [model_prob.weights[1].numpy()[0]]

    for epoch in range(100):
        model_prob.fit(x_plot, y_train, epochs=1)
        training_weights.append(model_prob.weights[0].numpy()[0, 0])
        training_bias.append(model_prob.weights[1].numpy()[0])
        epochs.append(epoch)
    plt.plot(epochs, training_weights, label='weight')
    plt.plot(epochs, training_bias, label='bias')
    plt.axhline(y = 1,label = 'True Weight',color = 'k',linestyle = ':')
    plt.axhline(y = 0,label = 'True Bias',color = 'k',linestyle = '--')
    plt.legend()
    plt.xlabel('Epochs')
    plt.show()
# y = <x, w> + b + eps
# eps ~  N(0,sigma^2)
# theta = (w,b)
# thata^* = argmax_{theta} P(D|theta)
