import numpy as np

def gaussian_elimination(A, B):
    A = A.astype(float)
    B = B.astype(float)
    n = len(B)

    # Прямой ход метода Гаусса
    for i in range(n):
        # Поиск максимального элемента в столбце для выбора главного элемента
        max_row = i + np.argmax(np.abs(A[i:n, i]))

        # Переставляем строки, если необходимо
        if i != max_row:
            A[[i, max_row]] = A[[max_row, i]].copy()
            B[[i, max_row]] = B[[max_row, i]].copy()

        # Приведение главного элемента к 1
        pivot = A[i, i]
        A[i] = A[i] / pivot
        B[i] = B[i] / pivot

        # Обнуление элементов ниже главного
        for j in range(i + 1, n):
            factor = A[j, i]
            A[j] -= factor * A[i]
            B[j] -= factor * B[i]

    # Обратный ход метода Гаусса
    X = np.zeros(n)
    for i in range(n - 1, -1, -1):
        X[i] = (B[i] - np.sum(A[i, i + 1:] * X[i + 1:])) / A[i, i]

    return X

# Входные данные (исправленная матрица A)
A = np.array([[1, -1, 1],
              [5, -4, 3],
              [2, 1, 1]], dtype=float)

B = np.array([-4, -12, 11], dtype=float)

# Решение
solution = gaussian_elimination(A, B)

# Контроль точности
precision = np.allclose(np.dot(A, solution), B)

# Вывод результатов
print("Решение системы уравнений:")
print(f"x1 = {solution[0]:.2f}, x2 = {solution[1]:.2f}, x3 = {solution[2]:.2f}")
print(f"Контроль точности: {'успешен' if precision else 'не пройден'}")
