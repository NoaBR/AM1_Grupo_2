"""
newton.py  —  CAPA MATEMÁTICA
Método de Newton-Raphson para sistemas no lineales G(X) = 0.
Usa el Jacobiano numérico (jacobiano.py) y la eliminación de Gauss (gauss.py).
Lo usan los esquemas implícitos (Euler inverso y Crank-Nicolson).
"""
from typing import Callable

import numpy as np

from .gauss import gauss
from .jacobiano import jacobiano


def newton(G: Callable[[np.ndarray], np.ndarray], X0: np.ndarray,
           tol: float = 1e-12, max_iter: int = 50) -> np.ndarray:
    """
    Itera  J(X_k) · ΔX = -G(X_k),   X_{k+1} = X_k + ΔX
    hasta que la corrección sea despreciable: |ΔX| <= tol·(1 + |X|).
    Cerca de la solución converge de forma cuadrática (los dígitos
    correctos se duplican en cada iteración).
    """
    X = np.array(X0, dtype=complex if np.iscomplexobj(X0) else float)
    for _ in range(max_iter):
        dX = gauss(jacobiano(G, X), -G(X))
        X = X + dX
        if np.linalg.norm(dX) <= tol * (1 + np.linalg.norm(X)):
            return X
    return X        # sin converger del todo: se devuelve la última aproximación
