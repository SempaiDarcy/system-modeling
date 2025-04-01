import numpy as np

# ------------------------------
# Метод простой итерации
# ------------------------------

# Система:
# sin(x) - y - 1.32 = 0  =>  y = sin(x) - 1.32
# cos(y) - x + 0.85 = 0 =>  x = cos(y) + 0.85

def phi1(x, y): return np.sin(x) - 1.32
def phi2(x, y): return np.cos(y) + 0.85

x, y = 0.5, 0.5
eps = 1e-3
count = 0

print("Метод простой итерации для системы:")
print("1. sin(x) - y - 1.32 = 0")
print("2. cos(y) - x + 0.85 = 0\n")

while True:
    x_new = phi2(x, y)
    y_new = phi1(x, y)
    print(f"Шаг {count+1:2d}: x = {x_new:.5f}, y = {y_new:.5f}")
    if abs(x_new - x) < eps and abs(y_new - y) < eps:
        break
    x, y = x_new, y_new
    count += 1
    if count > 100:
        print("Метод итерации не сошёлся за 100 шагов.")
        break

print(f"\nРешение найдено за {count} шагов:")
print(f"x = {x:.3f}, y = {y:.3f}")
