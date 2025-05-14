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

results = {name: {'acc': [], 'recall': [], 'spec': [], 'cm': []} for name in models}

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
        results[name]['cm'].append(cm)

best_acc = -1
worst_acc = 2
best_model = worst_model = ""
best_cm = worst_cm = None

for name in models:
    for i in range(R):
        acc = results[name]['acc'][i]
        if acc > best_acc:
            best_acc = acc
            best_model = name
            best_cm = results[name]['cm'][i]
        if acc < worst_acc:
            worst_acc = acc
            worst_model = name
            worst_cm = results[name]['cm'][i]

def plot_confusion(cm, title):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap='Blues')

    ax.set_xticks(np.arange(2))
    ax.set_yticks(np.arange(2))
    ax.set_xticklabels(['Previsto 0', 'Previsto 1'])
    ax.set_yticklabels(['Real 0', 'Real 1'])

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

    ax.set_title(title)
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    fig.tight_layout()
    plt.show()

print(f"Melhor acurácia: {best_acc:.4f} - Modelo: {best_model}")
plot_confusion(best_cm, f'Melhor Acurácia - {best_model}')

print(f"Pior acurácia: {worst_acc:.4f} - Modelo: {worst_model}")
plot_confusion(worst_cm, f'Pior Acurácia - {worst_model}')

plt.figure(figsize=(10, 6))
for name in models:
    plt.plot(results[name]['acc'], label=name)
plt.title('Curva de Aprendizado - Acurácia por Rodada')
plt.xlabel('Rodada')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
