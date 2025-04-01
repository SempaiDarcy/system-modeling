import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Задание 1: Система ОДУ
# dx/dt = -y/t, dy/dt = -x/t, x(0) = 1, y(0) = 1
# ------------------------------

print("Решение системы методом Эйлера и Рунге-Кутта (задание 1)")

def system(t, x, y):
    dxdt = -y / t if t != 0 else 0
    dydt = -x / t if t != 0 else 0
    return dxdt, dydt

# Метод Эйлера
t_vals = np.arange(0.1, 1.01, 0.1)
x_euler = [1]
y_euler = [1]
t0 = 0.1

for t in t_vals[:-1]:
    dx, dy = system(t0, x_euler[-1], y_euler[-1])
    x_euler.append(x_euler[-1] + 0.1 * dx)
    y_euler.append(y_euler[-1] + 0.1 * dy)
    t0 += 0.1

# Метод Рунге-Кутты 4-го порядка
x_rk = [1]
y_rk = [1]
t0 = 0.1
t_vals = np.arange(0.1, 1.01, 0.1)

for t in t_vals[:-1]:
    x_n, y_n = x_rk[-1], y_rk[-1]
    
    k1x, k1y = system(t0, x_n, y_n)
    k2x, k2y = system(t0 + 0.05, x_n + 0.05 * k1x, y_n + 0.05 * k1y)
    k3x, k3y = system(t0 + 0.05, x_n + 0.05 * k2x, y_n + 0.05 * k2y)
    k4x, k4y = system(t0 + 0.1, x_n + 0.1 * k3x, y_n + 0.1 * k3y)

    x_rk.append(x_n + (0.1 / 6)*(k1x + 2*k2x + 2*k3x + k4x))
    y_rk.append(y_n + (0.1 / 6)*(k1y + 2*k2y + 2*k3y + k4y))
    t0 += 0.1

print("\nМетод Эйлера:")
for i, t in enumerate(t_vals):
    print(f"t = {t:.1f}, x = {x_euler[i]:.5f}, y = {y_euler[i]:.5f}")

print("\nМетод Рунге-Кутта:")
for i, t in enumerate(t_vals):
    print(f"t = {t:.1f}, x = {x_rk[i]:.5f}, y = {y_rk[i]:.5f}")

plt.plot(t_vals, x_euler, 'bo--', label='x (Эйлер)')
plt.plot(t_vals, y_euler, 'ro--', label='y (Эйлер)')
plt.plot(t_vals, x_rk, 'b-', label='x (РК4)')
plt.plot(t_vals, y_rk, 'r-', label='y (РК4)')
plt.xlabel("t")
plt.ylabel("Значения")
plt.title("Решение системы ОДУ")
plt.legend()
plt.grid(True)
plt.show()

# ------------------------------
# Задание 2: ОДУ dy/dx = x + cos(y / sqrt(10)), y(0.6) = 4.2
# ------------------------------

print("\nРешение ОДУ методом Эйлера и Рунге-Кутта (задание 2)")

def f(x, y):
    return x + np.cos(y / np.sqrt(10))

x_vals = np.arange(0.6, 0.81, 0.05)
y_euler = [4.2]
y_rk = [4.2]

# Метод Эйлера
for x in x_vals[:-1]:
    y_euler.append(y_euler[-1] + 0.05 * f(x, y_euler[-1]))

# Метод Рунге-Кутты 4-го порядка
for x in x_vals[:-1]:
    y_n = y_rk[-1]
    k1 = f(x, y_n)
    k2 = f(x + 0.025, y_n + 0.025 * k1)
    k3 = f(x + 0.025, y_n + 0.025 * k2)
    k4 = f(x + 0.05, y_n + 0.05 * k3)
    y_rk.append(y_n + (0.05 / 6)*(k1 + 2*k2 + 2*k3 + k4))

print("\nМетод Эйлера:")
for i, x in enumerate(x_vals):
    print(f"x = {x:.2f}, y = {y_euler[i]:.5f}")

print("\nМетод Рунге-Кутта:")
for i, x in enumerate(x_vals):
    print(f"x = {x:.2f}, y = {y_rk[i]:.5f}")

plt.plot(x_vals, y_euler, 'go--', label='Эйлер')
plt.plot(x_vals, y_rk, 'g-', label='Рунге-Кутта')
plt.xlabel("x")
plt.ylabel("y")
plt.title("Решение ОДУ")
plt.legend()
plt.grid(True)
plt.show()
