import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    x = np.random.rand(100)
    m_ref = 0.3
    y = m_ref*x
    plt.scatter(x, y)

    noise = np.random.normal(0,0.5,100)*0.1
    y = y + noise
    plt.scatter(x, y)

    m_estimate = np.sum(x*y)/np.sum(x**2)
    print(m_estimate)
    plt.show()
