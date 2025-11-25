# ML
```markdown
# Machine Learning Educational Project / Образовательный проект по машинному обучению

## 📚 Table of Contents / Содержание
- [English Version](#english-version)
  - [Project Description](#project-description)
  - [Technologies Used](#technologies-used)
  - [Features](#features)
  - [Notebooks Overview](#notebooks-overview)
  - [Usage Examples](#usage-examples)
- [Русская Версия](#russian-version)
  - [Описание проекта](#описание-проекта)
  - [Используемые технологии](#используемые-технологии)
  - [Возможности](#возможности)
  - [Обзор ноутбуков](#обзор-ноутбуков)
  - [Примеры использования](#примеры-использования)

---

```markdown
# Machine Learning Educational Project

## 📊 Project Overview

This is a comprehensive educational project covering fundamental machine learning algorithms and their implementations. The project includes Jupyter notebooks with detailed explanations and practical examples of linear regression, logistic regression, perceptrons, and neural networks.

## 🎯 Project Description

This collection of Jupyter notebooks serves as a complete learning resource for understanding core machine learning concepts:

- **Linear Regression** - Theory, implementation, and application to real datasets
- **Logistic Regression** - Classification problems and marketing campaign prediction
- **Perceptron Algorithm** - Basic neural network implementation from scratch
- **Neural Networks** - Implementation with different activation functions and loss functions

The project is designed for:
- Students learning machine learning fundamentals
- Data science practitioners seeking hands-on examples
- Developers implementing ML algorithms from scratch

## 📚 Table of Contents

- [Technologies Used](#-technologies-used)
- [Installation & Setup](#-installation--setup)
- [Project Structure](#-project-structure)
- [Features](#-features)
- [Notebooks Overview](#-notebooks-overview)
- [Usage Examples](#-usage-examples)
- [License](#-license)
- [FAQ](#-faq)
- [Author](#-author)

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

## 💻 Installation & Setup

### Prerequisites
- Python 3.7+
- Jupyter Notebook/JupyterLab

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd machine-learning-educational-project
```

2. **Create virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install jupyter numpy pandas matplotlib scikit-learn seaborn
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Open and run notebooks sequentially**



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

---

# Russian Version

## 🎯 Описание проекта

Эта коллекция служит полным учебным ресурсом для понимания основных концепций машинного обучения:

- **Линейная регрессия** - Теория, реализация и применение к реальным наборам данных
- **Логистическая регрессия** - Задачи классификации и прогнозирование маркетинговых кампаний
- **Алгоритм перцептрона** - Базовая реализация нейронной сети с нуля
- **Нейронные сети** - Реализация с различными функциями активации и функциями потерь

Проект предназначен для:
- Студентов, изучающих основы машинного обучения
- Практиков в области data science, ищущих практические примеры
- Разработчиков, реализующих алгоритмы ML с нуля

## 🛠 Используемые технологии

**Основные технологии:**
- Python 3.8+
- Jupyter Notebook

**Библиотеки для анализа данных:**
- NumPy - Численные вычисления
- Pandas - Манипуляция и анализ данных
- Matplotlib - Визуализация данных
- Seaborn - Статистическая визуализация данных
- scikit-learn - Алгоритмы машинного обучения и метрики

**Алгоритмы машинного обучения:**
- Линейная регрессия
- Ridge & Lasso регрессия
- Логистическая регрессия
- Метод опорных векторов (SVM)
- Перцептрон
- Нейронные сети с различными функциями активации

## ✨ Возможности

### 🔧 Основные возможности
- **Комплексная теория** - Математические основы с подробными объяснениями
- **Реализации с нуля** - Пользовательские реализации алгоритмов ML
- **Приложения на реальных данных** - Практические примеры с реальными данными
- **Визуализация** - Обширное построение графиков и визуализация данных
- **Оценка моделей** - Множественные метрики и анализ производительности

### 📈 Образовательные возможности
- Пошаговые математические выводы
- Интерактивные примеры кода с объяснениями
- Сравнительный анализ различных алгоритмов
- Практические упражнения и тестирование
- Подходы к решению распространенных проблем ML

## 📓 Обзор ноутбуков

### 1. Линейная регрессия
- Теоретические основы линейной регрессии
- Генерация синтетических данных и визуализация
- Обучение модели и оценка (MSE, MAE)
- Практическое применение с набором данных о ценах на жилье в Бостоне
- Регуляризация Ridge и Lasso
- Демонстрации создания признаков

### 2. Логистическая регрессия
- Теория бинарной классификации
- Кейс прогнозирования маркетинговой кампании
- Предобработка данных и one-hot кодирование
- Оценка модели с точностью, F1-score и ROC-AUC
- Сравнение с SVM

### 3. Реализация перцептрона
- Алгоритм перцептрона с нуля
- Пользовательская реализация класса с прямым/обратным распространением
- Тестирование на синтетических и реальных наборах данных
- Сравнение с перцептроном из scikit-learn
- Распознавание пола по голосовым данным

### 4. Нейронные сети
- Реализация нейронной сети с сигмоидальной активацией
- Оптимизация градиентным спуском
- Реализация функции потерь LogLoss
- Сравнение различных функций потерь
- Анализ проблемы затухающих градиентов

## 🚀 Примеры использования

### Пример линейной регрессии
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print('Test MSE: ', mean_squared_error(y_test, predictions))
```

### Пример логистической регрессии
```python
from sklearn.linear_model import LogisticRegression

model_logistic = LogisticRegression()
model_logistic.fit(X_train, y_train)
y_predicted = model_logistic.predict_proba(X_test)[:, 1]
```

### Пример пользовательского перцептрона
```python
perceptron = Perceptron()
losses = perceptron.fit(X, y, num_epochs=300)
predictions = perceptron.forward_pass(X_test)
```

### Нейронная сеть с LogLoss
```python
neuron = Neuron()
J_values = neuron.fit(X, y)
predictions = neuron.forward_pass(X_test)
```