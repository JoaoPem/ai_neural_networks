# Projeto de Redes Neurais Artificiais

Este projeto foi desenvolvido como parte da disciplina de Inteligência Artificial. Seu objetivo é aplicar e validar diferentes modelos de redes neurais artificiais (RNAs) — tanto lineares quanto não-lineares — em tarefas de regressão e classificação, incluindo problemas reais e sintéticos.

## 🔧 Bibliotecas Utilizadas

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

## 📁 Estrutura do Projeto

O projeto está organizado em diferentes pastas conforme a tarefa proposta:
├── regressao/
├── classificacao_binaria/
├── classificacao_multiclasse/

---

## 📌 Primeira Etapa — Regressão e Classificação Binária

### 🔷 Tarefa de Regressão

- **Base de dados:** `aerogerador.dat`  
- **Objetivo:** Prever a potência gerada pelo aerogerador a partir da velocidade do vento.
- **Modelos implementados:**
  - ADALINE
  - Perceptron de Múltiplas Camadas (MLP)
- **Procedimentos:**
  - Análise exploratória com gráfico de dispersão.
  - Ajuste de hiperparâmetros com análise de _underfitting_ e _overfitting_.
  - Validação dos modelos com Simulação de Monte Carlo (R = 250)
  - Métrica: Erro quadrático médio (MSE)

### 🔷 Tarefa de Classificação Binária

- **Base de dados:** `Spiral3d.csv`  
- **Objetivo:** Classificar dados sintéticos em duas classes distintas.
- **Modelos implementados:**
  - Perceptron Simples
  - Perceptron de Múltiplas Camadas (MLP)
- **Procedimentos:**
  - Visualização com gráfico de dispersão.
  - Ajuste de hiperparâmetros com análise de _underfitting_ e _overfitting_.
  - Validação com Simulação de Monte Carlo (R = 250)
  - Métricas: Acurácia, Sensibilidade, Especificidade
  - Matrizes de confusão para melhores e piores rodadas
  - Curvas de aprendizado
  - Análise estatística com média, desvio padrão, maior e menor valor para cada métrica.

---

## 📌 Segunda Etapa — Classificação Multiclasse

### 🔷 Tarefa: Classificação de Condições da Coluna Vertebral

- **Base de dados:** `coluna vertebral.csv`  
- **Objetivo:** Classificar pacientes em três categorias:
  - `NO`: Normal
  - `DH`: Hérnia de Disco
  - `SL`: Espondilolistese
- **Modelos implementados:**
  - ADAptive LINear Element (ADALINE)
  - Perceptron de Múltiplas Camadas (MLP)
- **Procedimentos:**
  - Codificação one-hot dos rótulos
  - Validação com Simulação de Monte Carlo (R = 100)
  - Métricas: Acurácia, Sensibilidade, Especificidade
  - Matrizes de confusão para as melhores e piores rodadas
  - Curvas de aprendizado
  - Análise estatística com média, desvio padrão, maior e menor valor das métricas

---

## ⚙️ Considerações Técnicas

- Todos os modelos requerem **normalização dos dados**.
- O critério de convergência utilizado nos treinamentos foi o **número máximo de épocas**.
- Os hiperparâmetros dos modelos foram escolhidos com base em experimentação e análise dos resultados obtidos.

---

## 📊 Exemplos de Visualizações

O projeto utiliza visualizações com `matplotlib` e `seaborn`, incluindo:

- Gráficos de dispersão
- Curvas de aprendizado
- Matrizes de confusão (com `heatmap`)
- Análises estatísticas (Boxplot, Violinplot)

---

## 👨‍💻 Autor

Este projeto foi desenvolvido por João Pedro Monteiro como parte da disciplina de Inteligência Artificial aplicada a Redes Neurais Artificiais na graduação em Ciência da Computação.

---

