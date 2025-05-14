import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

R = 100
test_size = 0.2
class_names = ['NO', 'DH', 'SL']

def load_data():
    data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', skip_header=1, dtype='U')
    X = data[:, :6].astype(float)
    y = data[:, 6]
    return X, y

def preprocess(X, y):
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X, y, encoder.classes_

class ADALINE:
    def __init__(self, learning_rate=0.00005, epochs=150):
        self.lr = learning_rate
        self.epochs = epochs
    
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.weights = np.random.normal(0, 0.1, (X.shape[1] + 1, len(self.classes_)))
        
        for epoch in range(self.epochs):
            for i in np.random.permutation(X.shape[0]):
                xi = np.insert(X[i], 0, 1)
                outputs = np.dot(xi, self.weights)
                target = (y[i] == self.classes_).astype(float)
                errors = target - outputs
                self.weights += self.lr * np.outer(xi, errors)
        return self
    
    def predict(self, X):
        X = np.c_[np.ones(X.shape[0]), X]
        outputs = np.dot(X, self.weights)
        return np.argmax(outputs, axis=1)

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

def calculate_metrics(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'sensitivity': [],
        'specificity': []
    }
    
    for i in range(len(classes)):
        tp = cm[i,i]
        fn = np.sum(cm[i,:]) - tp
        fp = np.sum(cm[:,i]) - tp
        tn = np.sum(cm) - tp - fn - fp
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics['sensitivity'].append(sensitivity)
        metrics['specificity'].append(specificity)
    
    return metrics

def main():
    X, y = load_data()
    X, y, classes = preprocess(X, y)
    
    results = {
        'ADALINE': {'accuracy': [], 'sensitivity': [[] for _ in classes], 'specificity': [[] for _ in classes]},
        'MLP': {'accuracy': [], 'sensitivity': [[] for _ in classes], 'specificity': [[] for _ in classes]}
    }
    
    for rodada in range(1, R+1):
        print(f"\nRodada {rodada}/{R}")
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=rodada)
        
        try:
            print("Treinando ADALINE...", end=' ')
            adaline = ADALINE(learning_rate=0.00005, epochs=150)
            adaline.fit(X_train, y_train)
            y_pred = adaline.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, classes)
            
            results['ADALINE']['accuracy'].append(metrics['accuracy'])
            for i in range(len(classes)):
                results['ADALINE']['sensitivity'][i].append(metrics['sensitivity'][i])
                results['ADALINE']['specificity'][i].append(metrics['specificity'][i])
            print(f"Acurácia: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Erro no ADALINE: {str(e)}")
            continue
        
        try:
            print("Treinando MLP...", end=' ')
            mlp = create_mlp()
            mlp.fit(X_train, y_train)
            y_pred = mlp.predict(X_test)
            metrics = calculate_metrics(y_test, y_pred, classes)
            
            results['MLP']['accuracy'].append(metrics['accuracy'])
            for i in range(len(classes)):
                results['MLP']['sensitivity'][i].append(metrics['sensitivity'][i])
                results['MLP']['specificity'][i].append(metrics['specificity'][i])
            print(f"Acurácia: {metrics['accuracy']:.4f}")
        except Exception as e:
            print(f"Erro no MLP: {str(e)}")
            continue
    
    def print_results(model_name, results, classes):
        print(f"\n{model_name}:")
        print(f"Acurácia média: {np.mean(results['accuracy']):.4f} ± {np.std(results['accuracy']):.4f}")
        for i, name in enumerate(classes):
            print(f"\nClasse {name}:")
            print(f"Sensibilidade: {np.mean(results['sensitivity'][i]):.4f} ± {np.std(results['sensitivity'][i]):.4f}")
            print(f"Especificidade: {np.mean(results['specificity'][i]):.4f} ± {np.std(results['specificity'][i]):.4f}")
    
    print_results('ADALINE', results['ADALINE'], classes)
    print_results('MLP', results['MLP'], classes)

if __name__ == "__main__":
    main()