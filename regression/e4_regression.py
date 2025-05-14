import numpy as np
import matplotlib.pyplot as plt

def mlp_train(X, y, hidden_size=10, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    np.random.seed(0)
    W1 = np.random.randn(n_features, hidden_size) * 0.1
    b1 = np.zeros((1, hidden_size))
    W2 = np.random.randn(hidden_size, 1) * 0.1
    b2 = np.zeros((1, 1))

    train_errors = []
    val_errors = []

    for epoch in range(epochs):
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        output = z2

        error = y - output
        train_error = np.mean(np.square(error))
        train_errors.append(train_error)

        dW2 = a1.T @ (-2 * error)
        db2 = -2 * error.sum(axis=0, keepdims=True)
        da1 = -2 * error @ W2.T * (1 - np.tanh(z1) ** 2)
        dW1 = X.T @ da1
        db1 = da1.sum(axis=0, keepdims=True)

        W2 -= learning_rate * dW2 / n_samples
        b2 -= learning_rate * db2 / n_samples
        W1 -= learning_rate * dW1 / n_samples
        b1 -= learning_rate * db1 / n_samples

    return W1, b1, W2, b2, train_errors

def mlp_predict(X, W1, b1, W2, b2):
    a1 = np.tanh(X @ W1 + b1)
    return a1 @ W2 + b2

data = np.loadtxt('aerogerador.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

W1_small, b1_small, W2_small, b2_small, train_errors_small = mlp_train(X_norm, y, hidden_size=2, learning_rate=0.01, epochs=1000)

W1_large, b1_large, W2_large, b2_large, train_errors_large = mlp_train(X_norm, y, hidden_size=50, learning_rate=0.01, epochs=1000)

plt.figure(figsize=(12, 6))
plt.plot(train_errors_small, label='Underfitting (Subdimensionado)', color='blue')
plt.plot(train_errors_large, label='Overfitting (Superdimensionado)', color='red')
plt.xlabel('Epochs')
plt.ylabel('Training Error (MSE)')
plt.title('Learning Curves for Different Network Topologies')
plt.legend()
plt.show()
