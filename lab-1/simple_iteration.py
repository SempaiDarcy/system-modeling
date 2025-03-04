import numpy as np

def simple_iteration(A, B, eps=0.0001, max_iter=100):
    n = len(B)
    X = np.zeros(n)  # Начальное приближение
    X_new = np.zeros(n)

    # Проверка условия сходимости (диагональное преобладание)
    for i in range(n):
        if abs(A[i, i]) < sum(abs(A[i, j]) for j in range(n) if j != i):
            raise ValueError("Метод не сходится: нет диагонального преобладания")

    # Итерационный процесс
    for _ in range(max_iter):
        for i in range(n):
            sum_ax = sum(A[i, j] * X[j] for j in range(n) if j != i)  # Вычисление суммы по строке без диагонального элемента
            X_new[i] = (B[i] - sum_ax) / A[i, i]  # Итерационная формула

        # Проверка на достижение заданной точности
        if np.linalg.norm(X_new - X, ord=np.inf) < eps:
            return X_new
        X[:] = X_new  # Обновление значений

    raise ValueError("Метод не сошелся за указанное количество итераций")

# Входные данные (из варианта 4)
A = np.array([[11, 1, 2],
              [1, -12, 3],
              [2, 3, 13]], dtype=float)
B = np.array([11, 12, 13], dtype=float)

# Решение
solution = simple_iteration(A, B)

# Вывод результатов
print("Решение системы уравнений методом простой итерации:")
print(f"x1 = {solution[0]:.4f}, x2 = {solution[1]:.4f}, x3 = {solution[2]:.4f}")