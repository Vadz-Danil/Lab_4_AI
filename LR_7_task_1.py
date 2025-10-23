import pickle
import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt

# Завантаження даних
input_file = 'data_singlevar_regr.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбиття даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training

X_train = X[:num_training]
y_train = y[:num_training]
X_test = X[num_training:]
y_test = y[num_training:]

# Створення та навчання моделі лінійної регресії
regressor = linear_model.LinearRegression()
regressor.fit(X_train, y_train)

# Прогноз на тестових даних
y_test_pred = regressor.predict(X_test)

# Побудова графіка
plt.scatter(X_test, y_test, color='green')
plt.plot(X_test, y_test_pred, color='black', linewidth=4)
plt.xticks(())
plt.yticks(())
plt.show()

# Виведення метрик якості
print("Linear regression performance: ")
print("Mean absolute error = ",
      round(sm.mean_absolute_error(y_test, y_test_pred), 2))
print("Mean squared error = ",
      round(sm.mean_squared_error(y_test, y_test_pred), 2))
print("Median absolute error = ",
      round(sm.median_absolute_error(y_test, y_test_pred), 2))
print("Explain variance score = ",
      round(sm.explained_variance_score(y_test, y_test_pred), 2))
print("R2 score = ",
      round(sm.r2_score(y_test, y_test_pred), 2))

# Збереження моделі у файл
output_model_file = 'model.pkl'
with open(output_model_file, 'wb') as f:
    pickle.dump(regressor, f)

# Завантаження моделі з файлу
with open(output_model_file, 'rb') as f:
    regressor_model = pickle.load(f)

# Прогноз за допомогою завантаженої моделі
y_test_pred_new = regressor_model.predict(X_test)

# Виведення нової метрики якості
print("\nNew mean absolute error =",
      round(sm.mean_absolute_error(y_test, y_test_pred_new), 2))