import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	grade = tfp.distributions.Normal(loc = 2.5,scale = 0.5)
	dataset = grade.sample(100)

	print(grade.prob(5))
	print(grade.prob(2.5))
