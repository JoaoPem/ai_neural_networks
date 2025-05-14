import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

R = 10
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

def compute_confusion_matrix(y_true, y_pred, classes):
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true, pred] += 1
    return cm

def calculate_metrics(y_true, y_pred, classes):
    cm = compute_confusion_matrix(y_true, y_pred, classes)
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

def plot_confusion_matrix(cm, classes, title):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def custom_learning_curve(estimator, X, y, title):
    plt.figure(figsize=(10, 6))
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    
    train_sizes = np.linspace(0.1, 1.0, 5)
    train_scores = []
    test_scores = []
    
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    
    for size in train_sizes:
        n_samples = int(size * len(X_train_full))
        X_train = X_train_full[:n_samples]
        y_train = y_train_full[:n_samples]
        
        X_train = scaler.fit_transform(X_train)
        
        if isinstance(estimator, ADALINE):
            model = ADALINE(learning_rate=estimator.learning_rate, epochs=estimator.epochs)
        else:
            model = create_mlp()
        
        model.fit(X_train, y_train)
        
        train_scores.append(model.score(X_train, y_train))
        test_scores.append(model.score(X_test, y_test))
    
    plt.grid()
    plt.plot(train_sizes, train_scores, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores, 'o-', color="g", label="Validation score")
    plt.legend(loc="best")
    plt.show()

def analyze_results(results, X, y, classes):
    adaline_acc = np.array(results['ADALINE']['accuracy'])
    best_round_adaline = np.argmax(adaline_acc)
    worst_round_adaline = np.argmin(adaline_acc)
    
    mlp_acc = np.array(results['MLP']['accuracy'])
    best_round_mlp = np.argmax(mlp_acc)
    worst_round_mlp = np.argmin(mlp_acc)
    
    print("\n=== Best and Worst ADALINE Cases ===")
    print(f"Best accuracy: {adaline_acc[best_round_adaline]:.4f} (Round {best_round_adaline+1})")
    print(f"Worst accuracy: {adaline_acc[worst_round_adaline]:.4f} (Round {worst_round_adaline+1})")
    
    print("\n=== Best and Worst MLP Cases ===")
    print(f"Best accuracy: {mlp_acc[best_round_mlp]:.4f} (Round {best_round_mlp+1})")
    print(f"Worst accuracy: {mlp_acc[worst_round_mlp]:.4f} (Round {worst_round_mlp+1})")
    
    scaler = StandardScaler()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=best_round_adaline+1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    adaline_best = ADALINE(learning_rate=0.00005, epochs=150)
    adaline_best.fit(X_train, y_train)
    y_pred = adaline_best.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_pred, np.arange(len(classes)))
    plot_confusion_matrix(cm, classes, f"Best ADALINE (Accuracy: {adaline_acc[best_round_adaline]:.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=worst_round_adaline+1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    adaline_worst = ADALINE(learning_rate=0.00005, epochs=150)
    adaline_worst.fit(X_train, y_train)
    y_pred = adaline_worst.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_pred, np.arange(len(classes)))
    plot_confusion_matrix(cm, classes, f"Worst ADALINE (Accuracy: {adaline_acc[worst_round_adaline]:.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=best_round_mlp+1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mlp_best = create_mlp()
    mlp_best.fit(X_train, y_train)
    y_pred = mlp_best.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_pred, np.arange(len(classes)))
    plot_confusion_matrix(cm, classes, f"Best MLP (Accuracy: {mlp_acc[best_round_mlp]:.4f})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=worst_round_mlp+1)
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    mlp_worst = create_mlp()
    mlp_worst.fit(X_train, y_train)
    y_pred = mlp_worst.predict(X_test)
    cm = compute_confusion_matrix(y_test, y_pred, np.arange(len(classes)))
    plot_confusion_matrix(cm, classes, f"Worst MLP (Accuracy: {mlp_acc[worst_round_mlp]:.4f})")
    
    print("\n=== Learning Curves ===")
    custom_learning_curve(adaline_best, X, y, "Best ADALINE - Learning Curve")
    custom_learning_curve(mlp_best, X, y, "Best MLP - Learning Curve")
    custom_learning_curve(adaline_worst, X, y, "Worst ADALINE - Learning Curve")
    custom_learning_curve(mlp_worst, X, y, "Worst MLP - Learning Curve")

def main():
    data = np.genfromtxt('coluna_vertebral.csv', delimiter=',', skip_header=1, dtype='U')
    X = data[:, :6].astype(float)
    y = data[:, 6]
    
    encoder = LabelEncoder()
    y = encoder.fit_transform(y)
    classes = encoder.classes_
    scaler = StandardScaler()
    
    results = {
        'ADALINE': {'accuracy': [], 'sensitivity': [[] for _ in classes], 'specificity': [[] for _ in classes]},
        'MLP': {'accuracy': [], 'sensitivity': [[] for _ in classes], 'specificity': [[] for _ in classes]}
    }
    
    for rodada in range(1, R+1):
        print(f"Round {rodada}/{R}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=rodada)
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        

        adaline = ADALINE()
        adaline.fit(X_train, y_train)
        y_pred = adaline.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, np.arange(len(classes)))
        results['ADALINE']['accuracy'].append(metrics['accuracy'])
        for i in range(len(classes)):
            results['ADALINE']['sensitivity'][i].append(metrics['sensitivity'][i])
            results['ADALINE']['specificity'][i].append(metrics['specificity'][i])
        
        mlp = create_mlp()
        mlp.fit(X_train, y_train)
        y_pred = mlp.predict(X_test)
        metrics = calculate_metrics(y_test, y_pred, np.arange(len(classes)))
        results['MLP']['accuracy'].append(metrics['accuracy'])
        for i in range(len(classes)):
            results['MLP']['sensitivity'][i].append(metrics['sensitivity'][i])
            results['MLP']['specificity'][i].append(metrics['specificity'][i])
    
    analyze_results(results, X, y, classes)

if __name__ == "__main__":
    main()