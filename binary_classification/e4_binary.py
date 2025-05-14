import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

data = np.genfromtxt('Spiral3d.csv', delimiter=',', skip_header=1)
X = data[:, :3]
y = data[:, 3]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

def evaluate_model(model, X_test, y_test, model_name="MLP"):
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])

    print(f"\n--- {model_name} ---")
    print(f"Acurácia: {acc:.4f}")
    print(f"Sensibilidade (Recall): {recall:.4f}")
    print(f"Especificidade: {specificity:.4f}")
    print("Matriz de Confusão:\n", cm)

def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 5), random_state=42
    )
    train_mean = train_scores.mean(axis=1)
    test_mean = test_scores.mean(axis=1)

    plt.figure()
    plt.plot(train_sizes, train_mean, label="Treinamento")
    plt.plot(train_sizes, test_mean, label="Validação")
    plt.title(f"Curva de Aprendizado: {title}")
    plt.xlabel("Tamanho do treino")
    plt.ylabel("Acurácia")
    plt.legend()
    plt.grid()
    plt.show()

common_params = {
    "max_iter": 2000,
    "early_stopping": True,
    "validation_fraction": 0.1,
    "n_iter_no_change": 10,
    "learning_rate_init": 0.001,
    "random_state": 42
}

mlp_under = MLPClassifier(hidden_layer_sizes=(2,), **common_params)
mlp_under.fit(X_train, y_train)
evaluate_model(mlp_under, X_test, y_test, "MLP - Underfitting")
plot_learning_curve(mlp_under, X, y, "MLP - Underfitting")

mlp_good = MLPClassifier(hidden_layer_sizes=(10, 10), **common_params)
mlp_good.fit(X_train, y_train)
evaluate_model(mlp_good, X_test, y_test, "MLP - Topologia Ideal")
plot_learning_curve(mlp_good, X, y, "MLP - Topologia Ideal")

mlp_over = MLPClassifier(hidden_layer_sizes=(100, 100, 100), **common_params)
mlp_over.fit(X_train, y_train)
evaluate_model(mlp_over, X_test, y_test, "MLP - Overfitting")
plot_learning_curve(mlp_over, X, y, "MLP - Overfitting")
