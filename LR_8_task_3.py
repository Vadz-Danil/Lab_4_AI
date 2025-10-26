import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# === Варіант 6: Генерація даних ===
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)
y = 2 * np.sin(X.flatten()) + np.random.uniform(-0.6, 0.6, m)


# === Функція для побудови кривих навчання ===
def plot_learning_curves(model, X, y, title):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val, y_val_predict))

    plt.figure(figsize=(10, 6))
    plt.plot(np.sqrt(train_errors), "r-", linewidth=2, label="Обучающий набор")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Проверочный набор")
    plt.title(title, fontsize=14)
    plt.xlabel("Размер обучающего набора")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 3)
    plt.tight_layout()
    plt.show()


# === 1. Криві навчання для лінійної регресії ===
print("Будуємо криві навчання для лінійної моделі...")
lin_reg = LinearRegression()
plot_learning_curves(lin_reg, X, y, "Криві навчання для лінійної моделі")

# === 2. Криві навчання для поліноміальної регресії (ступінь 10) ===
print("Будуємо криві навчання для поліноміальної моделі 10-го ступеня...")
polynomial_regression_10 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_10, X, y, "Криві навчання для поліноміальної моделі (ступінь 10)")

# === 3. Криві навчання для поліноміальної регресії (ступінь 2) ===
print("Будуємо криві навчання для поліноміальної моделі 2-го ступеня...")
polynomial_regression_2 = Pipeline([
    ("poly_features", PolynomialFeatures(degree=2, include_bias=False)),
    ("lin_reg", LinearRegression()),
])
plot_learning_curves(polynomial_regression_2, X, y, "Криві навчання для поліноміальної моделі (ступінь 2)")