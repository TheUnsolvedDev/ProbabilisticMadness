import silence_tensorflow.auto
import tensorflow_probability as tfp
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
	# box muller transform
	U = np.random.rand(1000)
	U_1,U_2 = U[:500],U[500:]

	X_1 = np.sqrt(-2*np.log(U_1))*np.cos(2*np.pi*U_2)
	X_2 = np.sqrt(-2*np.log(U_1))*np.sin(2*np.pi*U_2)
	X = np.concatenate([X_1,X_2],axis = 0)
	print(np.var(X))

	plt.hist(X,bins = 100)
	# plt.show()

	#affine transform
	Y = 3.0+2.0*X
	plt.hist(Y,bins = 100)
	plt.show()
