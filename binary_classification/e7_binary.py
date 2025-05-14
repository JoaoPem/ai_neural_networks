import numpy as np
import matplotlib.pyplot as plt
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score
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

def gerar_estatisticas(metrica_nome, key):
    print(f"\n--- Tabela de {metrica_nome} ---")
    print(f"{'Modelo':<20} {'Média':>10} {'Desv.Pad':>10} {'Máximo':>10} {'Mínimo':>10}")
    
    tabela = []
    for model in results:
        valores = results[model][key]
        media = np.mean(valores)
        desvio = np.std(valores)
        maximo = np.max(valores)
        minimo = np.min(valores)
        tabela.append((model, valores))

        print(f"{model:<20} {media:>10.4f} {desvio:>10.4f} {maximo:>10.4f} {minimo:>10.4f}")
    
    return tabela

tabela_acc = gerar_estatisticas("Acurácia", "acc")
tabela_rec = gerar_estatisticas("Sensibilidade", "recall")
tabela_spec = gerar_estatisticas("Especificidade", "spec")

def plot_boxplot(tabela, metrica):
    dados = [valores for _, valores in tabela]
    labels = [modelo for modelo, _ in tabela]

    plt.figure(figsize=(8, 6))
    plt.boxplot(dados, labels=labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(color='black'),
                capprops=dict(color='black'))
    
    plt.title(f'Boxplot da {metrica}')
    plt.ylabel(metrica)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

plot_boxplot(tabela_acc, "Acurácia")
plot_boxplot(tabela_rec, "Sensibilidade")
plot_boxplot(tabela_spec, "Especificidade")
