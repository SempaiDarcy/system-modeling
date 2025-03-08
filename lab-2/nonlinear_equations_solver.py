import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import bisect, newton, fsolve

# Уравнение из варианта 4: x^3 - 4x + 2 = 0
def f(x):
    return x**3 - 4*x + 2

# 1️⃣ Графическое представление уравнения
def plot_function():
    x_vals = np.linspace(-3, 3, 400)
    y_vals = f(x_vals)

    plt.figure(figsize=(8, 5))
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)
    plt.plot(x_vals, y_vals, label=r"$x^3 - 4x + 2 = 0$", color='blue')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('График функции для определения корней')
    plt.legend()
    plt.grid(True)
    plt.show()

# 2️⃣ Метод бисекции (деление отрезка пополам)
def bisection_method(f, a, b, eps=0.0001):
    if f(a) * f(b) > 0:
        raise ValueError("На интервале нет корня (или их четное количество)")

    while abs(b - a) > eps:
        c = (a + b) / 2  # Середина отрезка
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c

    return (a + b) / 2

# 3️⃣ Метод хорд
def secant_method(f, x0, x1, tol=1e-4, max_iter=100):
    for _ in range(max_iter):
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))  # Формула метода хорд
        if abs(x2 - x1) < tol:
            return x2
        x0, x1 = x1, x2
    raise ValueError("Метод хорд не сошелся")

# 4️⃣ Метод Ньютона (касательных)
def newton_method(f, x0, tol=1e-4):
    return newton(f, x0, tol=tol)

# 5️⃣ Метод fsolve (дополнительная проверка)
def find_root_fsolve():
    root = fsolve(f, 1)
    return root[0]

# 6️⃣ Нахождение всех корней полинома
def find_polynomial_roots():
    coefficients = [1, 0, -4, 2]  # Коэффициенты уравнения x^3 - 4x + 2
    roots = np.roots(coefficients)  # Все корни полинома
    return roots

# Выбор интервала корня по графику
a, b = -3, -1

# Построение графика
plot_function()

# Решение методом бисекции
root_bisect = bisection_method(f, a, b)
print(f"Метод бисекции: x ≈ {root_bisect:.5f}")

# Решение методом хорд
root_secant = secant_method(f, a, b)
print(f"Метод хорд: x ≈ {root_secant:.5f}")

# Решение методом Ньютона
root_newton = newton_method(f, -2)
print(f"Метод Ньютона: x ≈ {root_newton:.5f}")

# Решение методом fsolve
root_fsolve = find_root_fsolve()
print(f"Метод fsolve: x ≈ {root_fsolve:.5f}")

# Вычисление всех корней полинома
roots_poly = find_polynomial_roots()
print("Все корни полинома x^3 - 4x + 2:")
print(roots_poly)
