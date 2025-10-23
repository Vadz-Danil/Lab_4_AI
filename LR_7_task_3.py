import numpy as np
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, explained_variance_score, r2_score
import matplotlib.pyplot as plt

# Завантаження даних
data = np.loadtxt('data_multivar_regr.txt', delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбиття даних на навчальний та тестовий набори
num_training = int(0.8 * len(X))
num_test = len(X) - num_training
X_train, X_test = X[:num_training], X[num_training:]
y_train, y_test = y[:num_training], y[num_training:]

# Створення та навчання моделі лінійної регресії
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)
y_test_pred_linear = linear_regressor.predict(X_test)

# Виведення метрик якості лінійної регресії
print("Linear Regressor performance:")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred_linear), 2))
print("Mean squared error =", round(mean_squared_error(y_test, y_test_pred_linear), 2))
print("Median absolute error =", round(median_absolute_error(y_test, y_test_pred_linear), 2))
print("Explained variance score =", round(explained_variance_score(y_test, y_test_pred_linear), 2))
print("R2 score =", round(r2_score(y_test, y_test_pred_linear), 2))

# Створення та навчання поліноміального регресора (ступінь 10)
degree = 10
polyreg = make_pipeline(PolynomialFeatures(degree), LinearRegression())
polyreg.fit(X_train, y_train)
y_test_pred_poly = polyreg.predict(X_test)

# Виведення метрик якості поліноміальної регресії
print("\nPolynomial Regressor (degree 10) performance:")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred_poly), 2))
print("Mean squared error =", round(mean_squared_error(y_test, y_test_pred_poly), 2))
print("Median absolute error =", round(median_absolute_error(y_test, y_test_pred_poly), 2))
print("Explained variance score =", round(explained_variance_score(y_test, y_test_pred_poly), 2))
print("R2 score =", round(r2_score(y_test, y_test_pred_poly), 2))

# Тестування на вибірковій точці [7.66, 6.29, 5.66]
sample_point = np.array([[7.66, 6.29, 5.66]])
expected_value = 41.35

# Прогноз для вибіркової точки лінійним регресором
linear_pred = linear_regressor.predict(sample_point)
print("\nPrediction for sample point [7.66, 6.29, 5.66] (Linear):", round(linear_pred[0], 2))
print("Difference from expected value (41.35):", round(abs(linear_pred[0] - expected_value), 2))

# Прогноз для вибіркової точки поліноміальним регресором
poly_pred = polyreg.predict(sample_point)
print("Prediction for sample point [7.66, 6.29, 5.66] (Polynomial):", round(poly_pred[0], 2))
print("Difference from expected value (41.35):", round(abs(poly_pred[0] - expected_value), 2))

# Збереження моделей
with open('linear_model_task_3.pkl', 'wb') as f:
    pickle.dump(linear_regressor, f)
with open('poly_model_task_3.pkl', 'wb') as f:
    pickle.dump(polyreg, f)

# Завантаження моделей для перевірки
with open('linear_model_task_3.pkl', 'rb') as f:
    loaded_linear_model = pickle.load(f)
with open('poly_model_task_3.pkl', 'rb') as f:
    loaded_poly_model = pickle.load(f)

# Прогноз завантаженими моделями
y_test_pred_linear_loaded = loaded_linear_model.predict(X_test)
y_test_pred_poly_loaded = loaded_poly_model.predict(X_test)

# Перевірка метрик для завантажених моделей
print("\nLoaded Linear Regressor performance:")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred_linear_loaded), 2))
print("\nLoaded Polynomial Regressor performance:")
print("Mean absolute error =", round(mean_absolute_error(y_test, y_test_pred_poly_loaded), 2))