from tkinter import N
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

NUMBER_DATA = 2000
ENTRY_DIM = 2
EXIT_DIM = 1


def factors():
    L_obj = yield tfp.distributions.Normal(loc=tf.zeros([ENTRY_DIM,EXIT_DIM]), scale=tf.ones([ENTRY_DIM,EXIT_DIM]))
    F_obj = yield tfp.distributions.Normal(loc=tf.zeros([EXIT_DIM, NUMBER_DATA]), scale=tf.ones([EXIT_DIM, NUMBER_DATA]))
    X_obj = yield tfp.distributions.Normal(loc=tf.matmul(L_obj, F_obj), scale=0.7)


if __name__ == '__main__':
    model = tfp.distributions.JointDistributionCoroutineAutoBatched(factors)

    l, f, x = model.sample()
    new_l = tf.Variable(tf.ones([ENTRY_DIM, EXIT_DIM]))
    new_f = tf.Variable(tf.ones([EXIT_DIM, NUMBER_DATA]))

    def loss(l, f): return model.log_prob(l, f, x)
    tfp.math.minimize(lambda: -loss(new_l, new_f),
                      5000, tf.keras.optimizers.Adam())

    _, _, x_pred = model.sample(value=(new_l, new_f))
    new_x = tf.transpose(x_pred)
    x = tf.transpose(x)

    plt.scatter(x[:, 0], x[:, 1], label='true')
    plt.scatter(new_x[:, 0], new_x[:, 1], label='pred')
    plt.legend()
    plt.show()
