import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

R = 100
test_size = 0.2
class_names = ['NO', 'DH', 'SL']

class ADALINE:
    def __init__(self, learning_rate=0.00005, epochs=150):
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.weights = np.random.normal(0, 0.1, (X.shape[1] + 1, len(self.classes_)))
        for _ in range(self.epochs):
            for i in np.random.permutation(X.shape[0]):
                xi = np.insert(X[i], 0, 1)
                outputs = np.dot(xi, self.weights)
                target = (y[i] == self.classes_).astype(float)
                errors = target - outputs
                self.weights += self.learning_rate * np.outer(xi, errors)
        return self
    
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        outputs = np.dot(X, self.weights)
        return np.argmax(outputs, axis=1)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def create_mlp():
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation='relu',
        solver='adam',
        learning_rate_init=0.0005,
        max_iter=500,
        batch_size=16,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=42
    )

def main():
    data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', skip_header=1, dtype='U')
    X = data[:, :6].astype(float)
    y = data[:, 6]
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    scaler = StandardScaler()
    
    adaline_accuracies = []
    mlp_accuracies = []
    
    for rodada in range(1, R+1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rodada)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        
        adaline = ADALINE()
        adaline.fit(X_train, y_train)
        adaline_accuracies.append(adaline.score(X_test, y_test))
        
        mlp = create_mlp()
        mlp.fit(X_train, y_train)
        mlp_accuracies.append(mlp.score(X_test, y_test))
    
    adaline_stats = {
        'Mean': np.mean(adaline_accuracies),
        'Std Dev': np.std(adaline_accuracies),
        'Max': np.max(adaline_accuracies),
        'Min': np.min(adaline_accuracies)
    }
    
    mlp_stats = {
        'Mean': np.mean(mlp_accuracies),
        'Std Dev': np.std(mlp_accuracies),
        'Max': np.max(mlp_accuracies),
        'Min': np.min(mlp_accuracies)
    }
    
    print(f"{'Model':<10} {'Mean':<10} {'Std Dev':<10} {'Max':<10} {'Min':<10}")
    print("-" * 50)
    print(f"{'ADALINE':<10} {adaline_stats['Mean']:<10.4f} {adaline_stats['Std Dev']:<10.4f} {adaline_stats['Max']:<10.4f} {adaline_stats['Min']:<10.4f}")
    print(f"{'MLP':<10} {mlp_stats['Mean']:<10.4f} {mlp_stats['Std Dev']:<10.4f} {mlp_stats['Max']:<10.4f} {mlp_stats['Min']:<10.4f}")
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    sns.boxplot(data=[adaline_accuracies, mlp_accuracies])
    plt.xticks([0, 1], ['ADALINE', 'MLP'])
    plt.title('Boxplot')
    plt.ylabel('Accuracy')
    
    plt.subplot(1, 2, 2)
    sns.violinplot(data=[adaline_accuracies, mlp_accuracies])
    plt.xticks([0, 1], ['ADALINE', 'MLP'])
    plt.title('Violin Plot')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, R+1), adaline_accuracies, label='ADALINE')
    plt.plot(range(1, R+1), mlp_accuracies, label='MLP')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Accuracy over Rounds')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
