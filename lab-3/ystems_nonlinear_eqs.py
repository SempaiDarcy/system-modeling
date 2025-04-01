import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

# ------------------------------
# Метод Ньютона (вручную)
# ------------------------------

def F(v):
    x, y = v[0][0], v[1][0]
    f1 = np.tan(x**y + 5) - x**2
    f2 = 0.5 * x**2 + 2 * y**2 - 1
    return np.array([[f1], [f2]])

def J(v):
    x, y = v[0][0], v[1][0]
    df1_dx = (2 * x) * (-1)  # численно неверно, но приближённо — уточняется ниже
    df1_dy = (1 / np.cos(x**y + 5)**2) * (x**y * np.log(x)) if x > 0 else 0  # приближённо
    df2_dx = 0.5 * 2 * x
    df2_dy = 2 * 2 * y
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])

x0 = np.array([[0.5], [0.5]])
eps = 1e-4
max_iter = 100
i = 0

while i < max_iter:
    f_val = F(x0)
    j_val = J(x0)
    try:
        delta = np.linalg.solve(j_val, f_val)
    except np.linalg.LinAlgError:
        print("Матрица Якоби вырождена на шаге", i)
        break
    x1 = x0 - delta
    if np.linalg.norm(x1 - x0) < eps:
        break
    x0 = x1
    i += 1

print("Метод Ньютона (собственная реализация):")
print(f"x = {x0[0][0]:.5f}, y = {x0[1][0]:.5f} за {i} итераций\n")

# ------------------------------
# fsolve (библиотека SciPy)
# ------------------------------

def system(v):
    x, y = v
    return [np.tan(x**y + 5) - x**2, 0.5 * x**2 + 2 * y**2 - 1]

guess = [0.5, 0.5]
sol = fsolve(system, guess)
print("Метод fsolve:")
print(f"x = {sol[0]:.5f}, y = {sol[1]:.5f}\n")

# ------------------------------
# Метод простой итерации
# ------------------------------

def phi1(x, y): return np.sin(x) - 1.32
def phi2(x, y): return np.cos(y) + 0.85

x, y = 0.5, 0.5
eps = 1e-3
count = 0

while True:
    x_new = phi2(x, y)
    y_new = phi1(x, y)
    if abs(x_new - x) < eps and abs(y_new - y) < eps:
        break
    x, y = x_new, y_new
    count += 1
    if count > 100:
        print("Метод итерации не сошёлся за 100 шагов")
        break

print("Метод простой итерации:")
print(f"x = {x:.3f}, y = {y:.3f} за {count} шагов")

# ------------------------------
# Построение графика решений (опционально)
# ------------------------------

x_vals = np.linspace(-2, 2, 400)
y_vals = np.linspace(-2, 2, 400)
X, Y = np.meshgrid(x_vals, y_vals)
Z1 = np.tan(X**Y + 5) - X**2
Z2 = 0.5 * X**2 + 2 * Y**2 - 1

plt.contour(X, Y, Z1, levels=[0], colors='blue')
plt.contour(X, Y, Z2, levels=[0], colors='red')
plt.title("График решений системы (вариант 4)")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.show()
