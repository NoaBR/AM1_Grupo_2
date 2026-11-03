"""
jacobiano.py  —  CAPA MATEMÁTICA
Jacobiano numérico de cualquier función vectorial G(X), por diferencias centradas.
Sirve para cualquier problema físico: no sabe nada de las ecuaciones concretas.
"""
from typing import Callable

import numpy as np


def jacobiano(G: Callable[[np.ndarray], np.ndarray], X: np.ndarray,
              h: float = 1e-6) -> np.ndarray:
    """
    J[:, j] = ( G(X + h e_j) - G(X - h e_j) ) / (2h)      error O(h²)
    Devuelve J (N x N) con J[i, j] = ∂G_i / ∂X_j.
    h ≈ 1e-6 equilibra el error de truncación (h²) y el de redondeo (ε/h).
    """
    X = np.asarray(X)
    tipo = complex if np.iscomplexobj(X) else float     # admite estados complejos
    X = X.astype(tipo)
    N = len(X)
    J = np.zeros((N, N), dtype=tipo)
    for j in range(N):
        dX = np.zeros(N, dtype=tipo)
        dX[j] = h
        J[:, j] = (G(X + dX) - G(X - dX)) / (2 * h)
    return J
