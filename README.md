# ML

# Machine Learning Educational Project / Образовательный проект по машинному обучению

<div align="center">

![Machine Learning](https://img.shields.io/badge/Machine-Learning-blue)
![Python](https://img.shields.io/badge/Python-3.8%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**Комплексный образовательный проект по основам машинного обучения**

</div>

## 📚 Table of Contents / Содержание

### English Version
- [🌟 Project Overview](#-project-overview)
- [🎯 Project Description](#-project-description)
- [🛠 Technologies Used](#-technologies-used)
- [🚀 Quick Start](#-quick-start)
- [📁 Project Structure](#-project-structure)
- [✨ Features](#-features)
- [📓 Notebooks Overview](#-notebooks-overview)
- [💡 Usage Examples](#-usage-examples)
- [❓ FAQ](#-faq)

### Русская Версия
- [🌟 Обзор проекта](#-обзор-проекта)
- [🎯 Описание проекта](#-описание-проекта)
- [🛠 Используемые технологии](#-используемые-технологии)
- [🚀 Быстрый старт](#-быстрый-старт)
- [📁 Структура проекта](#-структура-проекта)
- [✨ Возможности](#-возможности)
- [📓 Обзор ноутбуков](#-обзор-ноутбуков)
- [💡 Примеры использования](#-примеры-использования)
- [❓ Часто задаваемые вопросы](#-часто-задаваемые-вопросы)

---

# English Version

## 🌟 Project Overview

This is a comprehensive educational project covering fundamental machine learning algorithms and their implementations. The project includes Jupyter notebooks with detailed explanations and practical examples of linear regression, logistic regression, perceptrons, and neural networks.

<div align="center">

```mermaid
graph TD
    A[ML Educational Project] --> B[Linear Regression]
    A --> C[Logistic Regression]
    A --> D[Perceptron]
    A --> E[Neural Networks]
    
    B --> B1[Theory]
    B --> B2[Implementation]
    B --> B3[Real Data]
    
    C --> C1[Classification]
    C --> C2[Marketing Case]
    C --> C3[Evaluation]
    
    D --> D1[From Scratch]
    D --> D2[Custom Class]
    D --> D3[Comparison]
    
    E --> E1[Activation Functions]
    E --> E2[Loss Functions]
    E --> E3[Optimization]

## 🎯 Project Description

This collection of Jupyter notebooks serves as a complete learning resource for understanding core machine learning concepts:

- **📈 Linear Regression** - Theory, implementation, and application to real datasets
- **🎯 Logistic Regression** - Classification problems and marketing campaign prediction
- **🧠 Perceptron Algorithm** - Basic neural network implementation from scratch
- **🔮 Neural Networks** - Implementation with different activation functions and loss functions

### 🎓 Target Audience
- **Students** learning machine learning fundamentals
- **Data Science Practitioners** seeking hands-on examples
- **Developers** implementing ML algorithms from scratch
- **Researchers** exploring basic ML concepts

## 🛠 Technologies Used

### 🔧 Core Technologies
<div align="center">

| Technology | Version | Purpose |
|------------|---------|---------|
| Python | 3.8+ | Core programming language |
| Jupyter Notebook | Latest | Interactive development environment |
| NumPy | 1.21+ | Numerical computations |
| Pandas | 1.3+ | Data manipulation and analysis |
| Matplotlib | 3.4+ | Data visualization |
| Seaborn | 0.11+ | Statistical data visualization |
| scikit-learn | 1.0+ | Machine learning algorithms |

</div>

### 📊 Machine Learning Algorithms
- **Linear Models**: Linear Regression, Ridge, Lasso
- **Classification**: Logistic Regression, SVM, Perceptron
- **Neural Networks**: Custom implementations with various activations
- **Evaluation Metrics**: MSE, MAE, Accuracy, F1-score, ROC-AUC

## 🚀 Quick Start

### ⚡ Prerequisites
- Python 3.7 or higher
- pip package manager
- Git (for cloning repository)

### 📥 Installation Steps

1. **Clone the repository**
```bash
git clone https://github.com/your-username/machine-learning-project.git
cd machine-learning-project
```

2. **Create and activate virtual environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Launch Jupyter Notebook**
```bash
jupyter notebook
```

5. **Explore notebooks in order:**
   - `Linear_Regression.ipynb`
   - `Logistic_Regression.ipynb` 
   - `practice_perceptron.ipynb`
   - `practice_neuron.ipynb`
   - `practice_neuron_logloss.ipynb`

### 🧪 Testing Installation
```python
# Test your installation
import numpy as np
import pandas as pd
import sklearn
print("All packages installed successfully! 🎉")
```

## 📁 Project Structure

```
machine-learning-educational-project/
│
├── 📓 Notebooks/
│   ├── Linear_Regression.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── practice_perceptron.ipynb
│   ├── practice_neuron.ipynb
│   └── practice_neuron_logloss.ipynb
│
├── 📁 data/
│   ├── apples_pears.csv
│   ├── bank.csv
│   ├── bank-additional-full.csv
│   └── voice.csv
│
├── 📄 requirements.txt
├── 📜 LICENSE
└── 📖 README.md
```

## ✨ Features

### 🔧 Core Features
| Feature | Description | Icon |
|---------|-------------|------|
| **Comprehensive Theory** | Mathematical foundations with detailed explanations | 📚 |
| **From-Scratch Implementations** | Custom implementations of ML algorithms | 🛠️ |
| **Real Dataset Applications** | Practical examples with real-world data | 🌍 |
| **Advanced Visualization** | Extensive plotting and data visualization | 📊 |
| **Model Evaluation** | Multiple metrics and performance analysis | ✅ |

### 📈 Educational Features
- **Step-by-step mathematical derivations** with LaTeX formulas
- **Interactive code examples** with detailed explanations
- **Comparative analysis** of different algorithms
- **Hands-on exercises** and testing scenarios
- **Problem-solving approaches** for common ML challenges
- **Visual learning aids** and diagrams

## 📓 Notebooks Overview

### 1. 📈 Linear Regression
<div align="center">

| Topic | Description | Skills |
|-------|-------------|--------|
| **Theory** | Mathematical foundations | Mathematics |
| **Synthetic Data** | Data generation & visualization | NumPy, Matplotlib |
| **Model Training** | Implementation & evaluation | scikit-learn |
| **Real Application** | Boston housing dataset | Pandas, Data Analysis |
| **Regularization** | Ridge & Lasso techniques | Model Optimization |
</div>

### 2. 🎯 Logistic Regression
- **Binary classification theory** with probability interpretation
- **Marketing campaign prediction** case study
- **Data preprocessing** and one-hot encoding techniques
- **Model evaluation** with accuracy, F1-score, and ROC-AUC
- **Comparison with SVM** for classification tasks

### 3. 🧠 Perceptron Implementation
- **Perceptron algorithm** from first principles
- **Custom class implementation** with forward/backward passes
- **Testing on diverse datasets** including synthetic and real data
- **Performance comparison** with scikit-learn's Perceptron
- **Gender recognition** from voice characteristics

### 4. 🔮 Neural Networks
- **Neural network implementation** with sigmoid activation
- **Gradient descent optimization** techniques
- **LogLoss function implementation** and analysis
- **Comparison of different loss functions** (MSE vs LogLoss)
- **Vanishing gradients problem** and solutions

## 💡 Usage Examples

### 📈 Linear Regression Example
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate performance
mse = mean_squared_error(y_test, predictions)
print(f'Test MSE: {mse:.4f}')

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Linear Regression: Actual vs Predicted')
plt.show()
```

### 🎯 Logistic Regression Example
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Initialize and train model
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_train, y_train)

# Get probability predictions
y_pred_proba = model_logistic.predict_proba(X_test)[:, 1]
y_pred = model_logistic.predict(X_test)

# Comprehensive evaluation
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### 🧠 Custom Perceptron Example
```python
from sklearn.metrics import accuracy_score

# Initialize custom perceptron
perceptron = Perceptron()
losses = perceptron.fit(X_train, y_train, num_epochs=300)

# Make predictions
predictions = perceptron.forward_pass(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f'Perceptron Accuracy: {accuracy:.4f}')

# Plot training progress
plt.plot(losses)
plt.title('Perceptron Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
```

### 🔮 Neural Network with LogLoss
```python
# Initialize neural network with LogLoss
neuron = Neuron()
J_values = neuron.fit(X, y, num_epochs=5000)

# Generate predictions
predictions = neuron.forward_pass(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Evaluate performance
accuracy = accuracy_score(y_test, binary_predictions)
print(f'Neural Network Accuracy: {accuracy:.4f}')

# Visualize learning curve
plt.figure(figsize=(10, 6))
plt.plot(J_values)
plt.title('Neural Network Training (LogLoss)')
plt.xlabel('Iteration')
plt.ylabel('LogLoss')
plt.grid(True)
plt.show()
```

---

# Русская Версия

## 🌟 Обзор проекта

Это комплексный образовательный проект, охватывающий фундаментальные алгоритмы машинного обучения и их реализации. Проект включает Jupyter notebooks с подробными объяснениями и практическими примерами линейной регрессии, логистической регрессии, перцептронов и нейронных сетей.

## 🎯 Описание проекта

Эта коллекция Jupyter notebooks служит полным учебным ресурсом для понимания основных концепций машинного обучения:

- **📈 Линейная регрессия** - Теория, реализация и применение к реальным наборам данных
- **🎯 Логистическая регрессия** - Задачи классификации и прогнозирование маркетинговых кампаний
- **🧠 Алгоритм перцептрона** - Базовая реализация нейронной сети с нуля
- **🔮 Нейронные сети** - Реализация с различными функциями активации и функциями потерь

### 🎓 Целевая аудитория
- **Студенты**, изучающие основы машинного обучения
- **Практики Data Science**, ищущие практические примеры
- **Разработчики**, реализующие алгоритмы ML с нуля
- **Исследователи**, изучающие базовые концепции ML

## 🛠 Используемые технологии

### 🔧 Основные технологии
<div align="center">

| Технология | Версия | Назначение |
|------------|---------|------------|
| Python | 3.8+ | Основной язык программирования |
| Jupyter Notebook | Последняя | Интерактивная среда разработки |
| NumPy | 1.21+ | Численные вычисления |
| Pandas | 1.3+ | Манипуляция и анализ данных |
| Matplotlib | 3.4+ | Визуализация данных |
| Seaborn | 0.11+ | Статистическая визуализация |
| scikit-learn | 1.0+ | Алгоритмы машинного обучения |

</div>

### 📊 Алгоритмы машинного обучения
- **Линейные модели**: Линейная регрессия, Ridge, Lasso
- **Классификация**: Логистическая регрессия, SVM, Перцептрон
- **Нейронные сети**: Пользовательские реализации с различными активациями
- **Метрики оценки**: MSE, MAE, Accuracy, F1-score, ROC-AUC

## 🚀 Быстрый старт

### ⚡ Предварительные требования
- Python 3.7 или выше
- Менеджер пакетов pip
- Git (для клонирования репозитория)

### 📥 Шаги установки

1. **Клонируйте репозиторий**
```bash
git clone https://github.com/your-username/machine-learning-project.git
cd machine-learning-project
```

2. **Создайте и активируйте виртуальное окружение**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/MacOS
python3 -m venv venv
source venv/bin/activate
```

3. **Установите зависимости**
```bash
pip install -r requirements.txt
```

4. **Запустите Jupyter Notebook**
```bash
jupyter notebook
```

5. **Изучайте notebooks по порядку:**
   - `Linear_Regression.ipynb`
   - `Logistic_Regression.ipynb`
   - `practice_perceptron.ipynb`
   - `practice_neuron.ipynb`
   - `practice_neuron_logloss.ipynb`

### 🧪 Проверка установки
```python
# Проверьте установку
import numpy as np
import pandas as pd
import sklearn
print("Все пакеты успешно установлены! 🎉")
```

## 📁 Структура проекта

```
образовательный-проект-ml/
│
├── 📓 Ноутбуки/
│   ├── Linear_Regression.ipynb
│   ├── Logistic_Regression.ipynb
│   ├── practice_perceptron.ipynb
│   ├── practice_neuron.ipynb
│   └── practice_neuron_logloss.ipynb
│
├── 📁 данные/
│   ├── apples_pears.csv
│   ├── bank.csv
│   ├── bank-additional-full.csv
│   └── voice.csv
│
├── 📄 requirements.txt
├── 📜 LICENSE
└── 📖 README.md
```

## ✨ Возможности

### 🔧 Основные возможности
| Возможность | Описание | Иконка |
|-------------|----------|--------|
| **Комплексная теория** | Математические основы с подробными объяснениями | 📚 |
| **Реализации с нуля** | Пользовательские реализации алгоритмов ML | 🛠️ |
| **Приложения на реальных данных** | Практические примеры с реальными данными | 🌍 |
| **Продвинутая визуализация** | Обширное построение графиков и визуализация | 📊 |
| **Оценка моделей** | Множественные метрики и анализ производительности | ✅ |

### 📈 Образовательные возможности
- **Пошаговые математические выводы** с формулами LaTeX
- **Интерактивные примеры кода** с подробными объяснениями
- **Сравнительный анализ** различных алгоритмов
- **Практические упражнения** и сценарии тестирования
- **Подходы к решению** распространенных проблем ML
- **Визуальные помощники** и диаграммы для обучения

## 📓 Обзор ноутбуков

### 1. 📈 Линейная регрессия
<div align="center">

| Тема | Описание | Навыки |
|------|----------|--------|
| **Теория** | Математические основы | Математика |
| **Синтетические данные** | Генерация и визуализация | NumPy, Matplotlib |
| **Обучение модели** | Реализация и оценка | scikit-learn |
| **Реальное применение** | Набор данных о жилье в Бостоне | Pandas, Анализ данных |
| **Регуляризация** | Методы Ridge и Lasso | Оптимизация моделей |
</div>

### 2. 🎯 Логистическая регрессия
- **Теория бинарной классификации** с вероятностной интерпретацией
- **Кейс прогнозирования маркетинговой кампании**
- **Техники предобработки данных** и one-hot кодирования
- **Оценка модели** с точностью, F1-score и ROC-AUC
- **Сравнение с SVM** для задач классификации

### 3. 🧠 Реализация перцептрона
- **Алгоритм перцептрона** с первых принципов
- **Пользовательская реализация класса** с прямым/обратным распространением
- **Тестирование на различных наборах данных** включая синтетические и реальные
- **Сравнение производительности** с перцептроном из scikit-learn
- **Распознавание пола** по характеристикам голоса

### 4. 🔮 Нейронные сети
- **Реализация нейронной сети** с сигмоидальной активацией
- **Техники оптимизации градиентным спуском**
- **Реализация функции потерь LogLoss** и анализ
- **Сравнение различных функций потерь** (MSE vs LogLoss)
- **Проблема затухающих градиентов** и решения

## 💡 Примеры использования

### 📈 Пример линейной регрессии
```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Создание и обучение модели
model = LinearRegression()
model.fit(X_train, y_train)

# Прогнозирование
predictions = model.predict(X_test)

# Оценка производительности
mse = mean_squared_error(y_test, predictions)
print(f'Test MSE: {mse:.4f}')

# Визуализация результатов
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Фактические значения')
plt.ylabel('Предсказанные значения')
plt.title('Линейная регрессия: Фактические vs Предсказанные')
plt.show()
```

### 🎯 Пример логистической регрессии
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

# Инициализация и обучение модели
model_logistic = LogisticRegression(random_state=42)
model_logistic.fit(X_train, y_train)

# Получение вероятностных предсказаний
y_pred_proba = model_logistic.predict_proba(X_test)[:, 1]
y_pred = model_logistic.predict(X_test)

# Комплексная оценка
print("Classification Report:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")
```

### 🧠 Пример пользовательского перцептрона
```python
from sklearn.metrics import accuracy_score

# Инициализация пользовательского перцептрона
perceptron = Perceptron()
losses = perceptron.fit(X_train, y_train, num_epochs=300)

# Прогнозирование
predictions = perceptron.forward_pass(X_test)

# Расчет точности
accuracy = accuracy_score(y_test, predictions)
print(f'Точность перцептрона: {accuracy:.4f}')

# График процесса обучения
plt.plot(losses)
plt.title('Потери при обучении перцептрона')
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.show()
```

### 🔮 Нейронная сеть с LogLoss
```python
# Инициализация нейронной сети с LogLoss
neuron = Neuron()
J_values = neuron.fit(X, y, num_epochs=5000)

# Генерация предсказаний
predictions = neuron.forward_pass(X_test)
binary_predictions = (predictions > 0.5).astype(int)

# Оценка производительности
accuracy = accuracy_score(y_test, binary_predictions)
print(f'Точность нейронной сети: {accuracy:.4f}')

# Визуализация кривой обучения
plt.figure(figsize=(10, 6))
plt.plot(J_values)
plt.title('Обучение нейронной сети (LogLoss)')
plt.xlabel('Итерация')
plt.ylabel('LogLoss')
plt.grid(True)
plt.show()
```
