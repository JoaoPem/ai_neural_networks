import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('Spiral3d.csv', header=None)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

scaler = StandardScaler()
X = scaler.fit_transform(X)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

class Perceptron:
    def __init__(self, input_dim, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = np.random.randn(input_dim)
        self.bias = np.random.randn(1)

    def activation(self, z):
        return 1 if z >= 0 else 0

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.activation(np.dot(xi, self.weights) + self.bias)
                error = target - output
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

    def predict(self, X):
        return np.array([self.activation(np.dot(xi, self.weights) + self.bias) for xi in X])

class MLP:
    def __init__(self, input_dim, hidden_dim=10, lr=0.01, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.hidden_dim = hidden_dim
        self.weights1 = np.random.randn(input_dim, hidden_dim)
        self.weights2 = np.random.randn(hidden_dim)
        self.bias1 = np.random.randn(hidden_dim)
        self.bias2 = np.random.randn(1)

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        for _ in range(self.epochs):
            for xi, target in zip(X, y):
                z1 = np.dot(xi, self.weights1) + self.bias1
                a1 = self.sigmoid(z1)
                z2 = np.dot(a1, self.weights2) + self.bias2
                output = self.sigmoid(z2)
                error = target - output
                d_z2 = error * output * (1 - output)
                d_z1 = d_z2 * self.weights2 * a1 * (1 - a1)
                self.weights2 += self.lr * d_z2 * a1
                self.weights1 += self.lr * np.outer(xi, d_z1)
                self.bias2 += self.lr * d_z2
                self.bias1 += self.lr * d_z1

    def predict(self, X):
        predictions = []
        for xi in X:
            z1 = np.dot(xi, self.weights1) + self.bias1
            a1 = self.sigmoid(z1)
            z2 = np.dot(a1, self.weights2) + self.bias2
            output = self.sigmoid(z2)
            predictions.append(output)
        return np.array(predictions)

def evaluate_model(model, X, y):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predictions = np.round(predictions)
    return accuracy_score(y_test, predictions)

perceptron = Perceptron(input_dim=X_train.shape[1], lr=0.01, epochs=100)
mlp = MLP(input_dim=X_train.shape[1], hidden_dim=10, lr=0.01, epochs=100)

print("Simple Perceptron Accuracy:", evaluate_model(perceptron, X, y))
print("MLP Accuracy:", evaluate_model(mlp, X, y))
