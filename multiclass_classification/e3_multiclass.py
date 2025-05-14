import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier

data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', skip_header=1)
X = data[:, :6]
y = data[:, 6].astype(int)

encoder = LabelEncoder()
y = encoder.fit_transform(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

class ADALINE:
    def __init__(self, learning_rate=0.001, epochs=200):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.weights = np.zeros(X.shape[1] + 1)
        self.loss_history = []
        
        for _ in range(self.epochs):
            errors = []
            for xi, target in zip(X, y):
                xi = np.insert(xi, 0, 1)
                output = self.activation(xi)
                error = target - output
                errors.append(error**2)
                self.weights += self.learning_rate * error * xi
            self.loss_history.append(np.mean(errors))
        return self
    
    def activation(self, X):
        return np.dot(X, self.weights)
    
    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        return np.where(self.activation(X) >= 0.5, 1, 0)

adaline = ADALINE(learning_rate=0.001, epochs=200)
adaline.fit(X_train, y_train)
y_pred_adaline = adaline.predict(X_test)
acc_adaline = np.mean(y_pred_adaline == y_test)
print(f"Acurácia ADALINE: {acc_adaline:.4f}")

mlp = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    max_iter=50,
    batch_size=16,
    random_state=42
)
mlp.fit(X_train, y_train)
acc_mlp = mlp.score(X_test, y_test)
print(f"Acurácia MLP: {acc_mlp:.4f}")

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(adaline.loss_history)
plt.title('ADALINE - Evolução da Loss')
plt.xlabel('Época')
plt.ylabel('MSE')

plt.subplot(1, 2, 2)
plt.plot(mlp.loss_curve_)
plt.title('MLP - Evolução da Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.tight_layout()
plt.show()