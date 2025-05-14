import numpy as np

data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', skip_header=1)
X = data[:, :-1].T

bias = np.ones((1, X.shape[1]))
X_augmented = np.vstack((bias, X))

print("Formato de X com dimensão ℝ^{p+1} x N:", X_augmented.shape)