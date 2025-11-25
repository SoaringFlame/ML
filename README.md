# ML
```markdown
# Machine Learning Educational Project


## 🎯 Project Description

This collection  serves as a complete learning resource for understanding core machine learning concepts:

- **Linear Regression** - Theory, implementation, and application to real datasets
- **Logistic Regression** - Classification problems and marketing campaign prediction
- **Perceptron Algorithm** - Basic neural network implementation from scratch
- **Neural Networks** - Implementation with different activation functions and loss functions

## 📚 Table of Contents


## 🛠 Technologies Used

**Core Technologies:**
- Python 3.8+
- Jupyter Notebook

**Data Science Libraries:**
- NumPy - Numerical computations
- Pandas - Data manipulation and analysis
- Matplotlib - Data visualization
- Seaborn - Statistical data visualization
- scikit-learn - Machine learning algorithms and metrics

**Machine Learning Algorithms:**
- Linear Regression
- Ridge & Lasso Regression
- Logistic Regression
- Support Vector Machines (SVM)
- Perceptron
- Neural Networks with various activation functions


## ✨ Features

### 🔧 Core Features
- **Comprehensive Theory** - Mathematical foundations with detailed explanations
- **From-Scratch Implementations** - Custom implementations of ML algorithms
- **Real Dataset Applications** - Practical examples with real-world data
- **Visualization** - Extensive plotting and data visualization
- **Model Evaluation** - Multiple metrics and performance analysis

### 📈 Educational Features
- Step-by-step mathematical derivations
- Interactive code examples with explanations
- Comparative analysis of different algorithms
- Hands-on exercises and testing
- Problem-solving approaches for common ML challenges

## 📓 Notebooks Overview

### 1. Linear Regression
- Theoretical foundations of linear regression
- Synthetic data generation and visualization
- Model training and evaluation (MSE, MAE)
- Real-world application with Boston housing dataset
- Ridge and Lasso regularization
- Feature engineering demonstrations

### 2. Logistic Regression
- Binary classification theory
- Marketing campaign prediction case study
- Data preprocessing and one-hot encoding
- Model evaluation with accuracy, F1-score, and ROC-AUC
- Comparison with SVM

### 3. Perceptron Implementation
- Perceptron algorithm from scratch
- Custom class implementation with forward/backward passes
- Testing on synthetic and real datasets
- Comparison with scikit-learn's Perceptron
- Gender recognition from voice data

### 4. Neural Networks
- Neural network implementation with sigmoid activation
- Gradient descent optimization
- LogLoss function implementation
- Comparison of different loss functions
- Vanishing gradients problem analysis

## 🚀 Usage Examples

### Linear Regression Example
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Test MSE: ', mean_squared_error(y_test, predictions))
```

### Logistic Regression Example
```python
from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_predicted = model_logistic.predict_proba(X_test)[:, 1]
```

### Custom Perceptron Example
```python
perceptron = Perceptron()
losses = perceptron.fit(X, y, num_epochs=300)
predictions = perceptron.forward_pass(X_test)
```

### Neural Network with LogLoss
```python
neuron = Neuron()
J_values = neuron.fit(X, y)
predictions = neuron.forward_pass(X_test)
```
