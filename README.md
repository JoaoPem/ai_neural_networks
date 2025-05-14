# Artificial Neural Networks Project

This project was developed as part of the Artificial Intelligence course. Its goal is to apply and validate different artificial neural network (ANN) models — both linear and non-linear — in regression and classification tasks, including real-world and synthetic problems.

## 🔧 Libraries Used

- [numpy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)
- [seaborn](https://seaborn.pydata.org/)

## 📁 Project Structure

The project is organized into different folders based on the proposed tasks:
├── regression/
├── binary_classification/
├── multiclass_classification/

---

## 📌 Phase 1 — Regression and Binary Classification

### 🔷 Regression Task

- **Dataset:** `aerogerador.dat`  
- **Goal:** Predict the wind turbine's generated power based on wind speed.  
- **Implemented Models:**
  - ADALINE
  - Multi-Layer Perceptron (MLP)
- **Procedures:**
  - Exploratory analysis with scatter plot
  - Hyperparameter tuning with underfitting and overfitting analysis
  - Model validation using Monte Carlo Simulation (R = 250)
  - Metric: Mean Squared Error (MSE)

### 🔷 Binary Classification Task

- **Dataset:** `Spiral3d.csv`  
- **Goal:** Classify synthetic data into two distinct classes.  
- **Implemented Models:**
  - Simple Perceptron
  - Multi-Layer Perceptron (MLP)
- **Procedures:**
  - Visualization using scatter plots
  - Hyperparameter tuning with underfitting and overfitting analysis
  - Validation using Monte Carlo Simulation (R = 250)
  - Metrics: Accuracy, Sensitivity, Specificity
  - Confusion matrices for best and worst runs
  - Learning curves
  - Statistical analysis with mean, standard deviation, maximum, and minimum for each metric

---

## 📌 Phase 2 — Multiclass Classification

### 🔷 Task: Classification of Vertebral Column Conditions

- **Dataset:** `coluna vertebral.csv`  
- **Goal:** Classify patients into three categories:
  - `NO`: Normal
  - `DH`: Disk Hernia
  - `SL`: Spondylolisthesis
- **Implemented Models:**
  - ADAptive LINear Element (ADALINE)
  - Multi-Layer Perceptron (MLP)
- **Procedures:**
  - One-hot encoding of labels
  - Validation using Monte Carlo Simulation (R = 100)
  - Metrics: Accuracy, Sensitivity, Specificity
  - Confusion matrices for best and worst runs
  - Learning curves
  - Statistical analysis with mean, standard deviation, max and min of each metric

---

## ⚙️ Technical Considerations

- All models require **data normalization**.
- The convergence criterion for training was the **maximum number of epochs**.
- Model hyperparameters were chosen based on experimentation and result analysis.

---

## 📊 Visualization Examples

The project uses visualizations with `matplotlib` and `seaborn`, including:

- Scatter plots
- Learning curves
- Confusion matrices (using `heatmap`)
- Statistical analysis (Boxplot, Violinplot)

---

## 👨‍💻 Author

This project was developed by João Pedro Monteiro as part of the Artificial Intelligence course on Artificial Neural Networks in the Computer Science undergraduate program.
