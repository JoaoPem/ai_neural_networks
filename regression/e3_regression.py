import numpy as np
import matplotlib.pyplot as plt

def adaline_train(X, y, learning_rate=0.000001, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    for _ in range(epochs):
        output = X @ weights + bias
        error = y - output
        weights += learning_rate * X.T @ error
        bias += learning_rate * error.sum()
    return weights, bias

def adaline_predict(X, weights, bias):
    return X @ weights + bias

def mlp_train(X, y, hidden_size=10, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    np.random.seed(0)
    W1 = np.random.randn(n_features, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * 0.1
    b2 = np.zeros((1, 1))

    for _ in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        output = z2

        error = y - output
        dW2 = a1.T @ (-2 * error)
        db2 = -2 * error.sum(axis=0, keepdims=True)
        da1 = -2 * error @ W2.T * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ da1
        db1 = da1.sum(axis=0, keepdims=True)

        W2 -= learning_rate * dW2 / n_samples
        b2 -= learning_rate * db2 / n_samples
        W1 -= learning_rate * dW1 / n_samples
        b1 -= learning_rate * db1 / n_samples

    return W1, b1, W2, b2

def mlp_predict(X, W1, b1, W2, b2):
    a1 = np.tanh(X @ W1 + b1)
    return a1 @ W2 + b2

data = np.loadtxt('aerogerador.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

adaline_weights, adaline_bias = adaline_train(X_norm, y)
adaline_output = adaline_predict(X_norm, adaline_weights, adaline_bias)

mlp_W1, mlp_b1, mlp_W2, mlp_b2 = mlp_train(X_norm, y)
mlp_output = mlp_predict(X_norm, mlp_W1, mlp_b1, mlp_W2, mlp_b2)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, s=1, label='True Values', color='blue')
plt.plot(X, adaline_output, label='ADALINE Predictions', color='red')
plt.plot(X, mlp_output, label='MLP Predictions', color='green')
plt.xlabel('Wind Speed')
plt.ylabel('Generated Power')
plt.title('ADALINE vs MLP Predictions')
plt.legend()
plt.show()
