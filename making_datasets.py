import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Загружаем датасет Iris
iris = load_iris()
X = iris.data
y = iris.target

# Разбиваем датасет на train и test три раза по-разному
for i in range(3):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    
    # Сохраняем train и test данные в pkl файлы
    with open(f'x_train_{i+1}.pkl', 'wb') as f:
        pickle.dump(X_train.tolist(), f)
        
    with open(f'x_test_{i+1}.pkl', 'wb') as f:
        pickle.dump(X_test.tolist(), f)

    # Сохраняем train и test данные в pkl файлы
    with open(f'y_train_{i+1}.pkl', 'wb') as f:
        pickle.dump(y_train.tolist(), f)
        
    with open(f'y_test_{i+1}.pkl', 'wb') as f:
        pickle.dump(y_test.tolist(), f)

# Создаем гиперпараметры для моделей Random Forest и Logistic Regression
rf_params = [
    {"n_estimators": 50, "max_depth": None},
    {"n_estimators": 100, "max_depth": 10},
    {"n_estimators": 200, "max_depth": 20}
]

lr_params = [
    {"C": 1.0, "penalty": "l2"},
    {"C": 0.01, "penalty": "l1"},
    {"C": 10.0, "solver": "liblinear"}
]

# Сохраняем гиперпараметры в отдельные pkl файлы
for i, params in enumerate(rf_params):
    with open(f'random_forest_params_{i+1}.pkl', 'wb') as f:
        pickle.dump(params, f)

for i, params in enumerate(lr_params):
    with open(f'logistic_regression_params_{i+1}.pkl', 'wb') as f:
        pickle.dump(params, f)
