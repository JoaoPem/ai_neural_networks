import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def adaline_train(X, y, learning_rate=0.01, epochs=1000):
    n_samples, n_features = X.shape
    weights = np.zeros(n_features)
    bias = 0
    for epoch in range(epochs):
        predictions = X @ weights + bias
        error = y.flatten() - predictions

        weights += learning_rate * X.T @ error / n_samples
        bias += learning_rate * error.sum() / n_samples
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

    for epoch in range(epochs):
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

def calculate_mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

data = np.loadtxt('aerogerador.dat')
X = data[:, 0].reshape(-1, 1)
y = data[:, 1].reshape(-1, 1)

X_mean = X.mean()
X_std = X.std()
X_norm = (X - X_mean) / X_std

R = 250
adaline_mse = []
mlp_mse = []

for i in range(R):
    X_train, X_test, y_train, y_test = train_test_split(X_norm, y, test_size=0.2, random_state=i)
    
    adaline_weights, adaline_bias = adaline_train(X_train, y_train)
    y_pred_adaline = adaline_predict(X_test, adaline_weights, adaline_bias)
    adaline_mse.append(calculate_mse(y_test, y_pred_adaline))
    
    W1, b1, W2, b2 = mlp_train(X_train, y_train, hidden_size=10, learning_rate=0.01, epochs=1000)
    y_pred_mlp = mlp_predict(X_test, W1, b1, W2, b2)
    mlp_mse.append(calculate_mse(y_test, y_pred_mlp))

adaline_mean = np.mean(adaline_mse)
adaline_std = np.std(adaline_mse)
adaline_max = np.max(adaline_mse)
adaline_min = np.min(adaline_mse)

mlp_mean = np.mean(mlp_mse)
mlp_std = np.std(mlp_mse)
mlp_max = np.max(mlp_mse)
mlp_min = np.min(mlp_mse)

print(f"ADALINE - MSE Statistics: Mean={adaline_mean:.4f}, Std={adaline_std:.4f}, Max={adaline_max:.4f}, Min={adaline_min:.4f}")
print(f"MLP - MSE Statistics: Mean={mlp_mean:.4f}, Std={mlp_std:.4f}, Max={mlp_max:.4f}, Min={mlp_min:.4f}")
