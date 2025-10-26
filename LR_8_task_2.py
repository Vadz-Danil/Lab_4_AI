import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# === Варіант 6 ===
m = 100
X = np.linspace(-3, 3, m).reshape(-1, 1)  # reshape для sklearn
y = 2 * np.sin(X.flatten()) + np.random.uniform(-0.6, 0.6, m)

# --- 1. Візуалізація згенерованих даних ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=20, label='Дані (y = 2·sin(x) + шум)')
plt.title('Варіант 6: Згенеровані дані')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 2. Лінійна регресія ---
lin_reg = LinearRegression()
lin_reg.fit(X, y)
y_pred_lin = lin_reg.predict(X)

# --- 3. Поліноміальна регресія (ступінь 2) ---
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print("X[0] =", X[0])
print("X_poly[0] =", X_poly[0])

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)
y_pred_poly = poly_reg.predict(X_poly)

# --- 4. Сортування для плавної лінії ---
X_sorted = np.sort(X, axis=0)
X_sorted_poly = poly_features.transform(X_sorted)
y_pred_poly_sorted = poly_reg.predict(X_sorted_poly)

# --- 5. Графік обох моделей ---
plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='blue', s=20, label='Дані')
plt.plot(X_sorted, lin_reg.predict(X_sorted), color='green', linewidth=2, label='Лінійна регресія')
plt.plot(X_sorted, y_pred_poly_sorted, color='red', linewidth=2, label='Поліноміальна регресія (ступінь 2)')
plt.title('Варіант 6: Лінійна vs Поліноміальна регресія')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# --- 6. Виведення коефіцієнтів ---
print("\n=== Модель лінійної регресії ===")
print(f"y = {lin_reg.coef_[0]:.4f}·x + {lin_reg.intercept_:.4f}")

print("\n=== Модель поліноміальної регресії (ступінь 2) ===")
a, b = poly_reg.coef_  # x², x
c = poly_reg.intercept_
print(f"y = {a:.4f}·x² + {b:.4f}·x + {c:.4f}")

# --- 7. Оцінка якості ---
print("\n=== Оцінка якості моделей ===")
print("Лінійна регресія:")
print(f"  R² = {r2_score(y, y_pred_lin):.4f}")
print(f"  MSE = {mean_squared_error(y, y_pred_lin):.4f}")

print("Поліноміальна регресія (ступінь 2):")
print(f"  R² = {r2_score(y, y_pred_poly):.4f}")
print(f"  MSE = {mean_squared_error(y, y_pred_poly):.4f}")

# --- 8. Математична модель варіанту ---
print("\n=== Модель генерації даних (ваш варіант) ===")
print("y = 2·sin(x) + шум ~ U(-0.6, 0.6)")

print("\n=== Отримана поліноміальна модель ===")
print(f"y ≈ {a:.4f}·x² + {b:.4f}·x + {c:.4f}")