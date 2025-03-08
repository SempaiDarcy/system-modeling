import numpy as np
from itertools import permutations


# Функция проверки диагонального преобладания
def check_diagonal_dominance(A):
    n = A.shape[0]
    for i in range(n):
        diagonal_element = abs(A[i, i])
        sum_non_diagonal = sum(abs(A[i, j]) for j in range(n) if j != i)
        if diagonal_element < sum_non_diagonal:
            return False  # Нет диагонального преобладания
    return True  # Диагональное преобладание есть


# Исходные данные (матрица A и вектор B)
A_original = np.array([[11, 1, 2],
                       [1, -12, 3],
                       [2, 3, 13]], dtype=float)

B_original = np.array([11, 12, 13], dtype=float)

# Перестановка строк для достижения диагонального преобладания
best_A, best_B = None, None
for perm in permutations(range(3)):  # Проверяем все перестановки строк
    A_permuted = A_original[list(perm), :]
    B_permuted = B_original[list(perm)]
    if check_diagonal_dominance(A_permuted):  # Если нашли диагонально преобладающую матрицу
        best_A, best_B = A_permuted, B_permuted
        break

if best_A is None:
    raise ValueError("Не удалось достичь диагонального преобладания. Метод может не сойтись.")


# Метод простой итерации
def simple_iteration(A, B, eps=1e-6, max_iter=1000):
    """
    Метод простой итерации для решения СЛАУ.

    Параметры:
      A - матрица коэффициентов (диагонально преобладающая)
      B - вектор свободных членов
      eps - заданная точность
      max_iter - максимальное количество итераций

    Возвращает:
      X - решение системы
      итерации - количество итераций до сходимости
    """
    n = len(B)
    X = np.zeros(n)  # Начальное приближение (нулевой вектор)
    X_new = np.zeros(n)

    # Итерационный процесс
    for iteration in range(max_iter):
        for i in range(n):
            sum_ax = sum(
                A[i, j] * X[j] for j in range(n) if j != i)  # Вычисляем сумму элементов строки без диагонального
            X_new[i] = (B[i] - sum_ax) / A[i, i]  # Итерационная формула

        # Проверка на достижение заданной точности
        if np.linalg.norm(X_new - X, ord=np.inf) < eps:
            return X_new, iteration + 1  # Возвращаем решение и число итераций

        X[:] = X_new  # Обновляем значения

    raise ValueError("Метод не сошелся за указанное количество итераций")


# Запуск метода на переставленной системе
solution_iteration, iteration_count = simple_iteration(best_A, best_B)

# Вывод результатов
print("Решение системы уравнений методом простой итерации:")
print(f"x₁ = {solution_iteration[0]:.6f}, x₂ = {solution_iteration[1]:.6f}, x₃ = {solution_iteration[2]:.6f}")
print(f"Метод сошелся за {iteration_count} итераций.")
