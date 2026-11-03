"""
gauss.py  —  CAPA MATEMÁTICA
Resolución de sistemas lineales A·x = b por eliminación de Gauss
con pivotaje parcial. La usa newton() en cada iteración.
"""
import numpy as np


def gauss(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Resuelve A x = b.
      1. Eliminación: se hacen ceros debajo de la diagonal, columna a columna.
         En cada columna se elige como pivote la fila con el mayor |A[i, k]|
         (pivotaje parcial), para no dividir por números pequeños.
      2. Sustitución hacia atrás: se despeja x de la última fila a la primera.
    """
    tipo = np.result_type(np.asarray(A), np.asarray(b), float)   # real o complejo
    A = np.array(A, dtype=tipo)              # copias: no se modifica lo que entra
    b = np.array(b, dtype=tipo)
    n = len(b)

    for k in range(n - 1):
        p = k + int(np.argmax(np.abs(A[k:, k])))     # fila del pivote
        if abs(A[p, k]) < 1e-300:
            raise ValueError("Matriz singular: no hay pivote no nulo.")
        if p != k:                                    # intercambio de filas
            A[[k, p]] = A[[p, k]]
            b[[k, p]] = b[[p, k]]
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]

    x = np.zeros(n, dtype=tipo)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - A[i, i + 1:] @ x[i + 1:]) / A[i, i]
    return x
