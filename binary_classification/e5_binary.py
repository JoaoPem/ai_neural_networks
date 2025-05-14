import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
import warnings

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

data = np.genfromtxt('Spiral3d.csv', delimiter=',', skip_header=1)
X = data[:, :3]
y = data[:, 3]

R = 250
models = {
    "Underfitting": MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000, random_state=42),
    "Topologia Ideal": MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000, random_state=42),
    "Overfitting": MLPClassifier(hidden_layer_sizes=(100, 100, 100), max_iter=1000, random_state=42)
}

results = {name: {'acc': [], 'recall': [], 'spec': []} for name in models.keys()}

for i in range(R):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        acc = accuracy_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred, average='macro')
        specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) != 0 else 0.0

        results[name]['acc'].append(acc)
        results[name]['recall'].append(recall)
        results[name]['spec'].append(specificity)

print("\n--- Resultados Médios após 250 Simulações (Monte Carlo) ---")
for name in models.keys():
    acc_mean = np.mean(results[name]['acc'])
    recall_mean = np.mean(results[name]['recall'])
    spec_mean = np.mean(results[name]['spec'])
    
    print(f"\nModelo: {name}")
    print(f"Acurácia Média: {acc_mean:.4f}")
    print(f"Sensibilidade Média: {recall_mean:.4f}")
    print(f"Especificidade Média: {spec_mean:.4f}")
