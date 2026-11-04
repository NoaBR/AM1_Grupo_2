"""
esquemas_temporales.py  —  CAPA MATEMÁTICA
Solo integración temporal: los esquemas (Euler, Euler inverso, Crank-Nicolson,
RK4), el problema de Cauchy y la malla temporal (partition).
No sabe nada de la física concreta: todo recibe F.
Los esquemas implícitos resuelven su ecuación con newton() (Newton + Gauss).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from .newton import newton

# Tipo de la función del problema: F(U, t) -> dU/dt
FuncionF = Callable[[np.ndarray, float], np.ndarray]


def euler(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    return U + At * F(U, t)


def euler_inverso(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    """U_{n+1} = U_n + At F(U_{n+1})   (implícito: G(X) = 0 se resuelve con Newton)."""
    def G(X: np.ndarray) -> np.ndarray:
        return X - U - At * F(X, t + At)

    return newton(G, euler(F, U, At, t))     # Euler como estimación inicial


def crank_nicholson(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    """U_{n+1} = U_n + At/2 [F(U_n) + F(U_{n+1})]   (implícito: se resuelve con Newton)."""
    a = U + At / 2 * F(U, t)                 # parte explícita, se calcula una vez

    def G(X: np.ndarray) -> np.ndarray:
        return X - a - At / 2 * F(X, t + At)

    return newton(G, euler(F, U, At, t))


def rk4(F: FuncionF, U: np.ndarray, At: float, t: float) -> np.ndarray:
    k1 = F(U, t)
    k2 = F(U + At * k1 / 2, t + At / 2)
    k3 = F(U + At * k2 / 2, t + At / 2)
    k4 = F(U + At * k3, t + At)
    return U + At * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def cauchy_problem(F: FuncionF, U0: np.ndarray, t: np.ndarray,
                   scheme: Callable) -> np.ndarray:
    """Integra dU/dt = F(U, t) en la malla t. Devuelve U (N+1 x Nv)."""
    N = len(t) - 1
    U = np.zeros((N + 1, len(U0)))
    U[0, :] = U0
    for n in range(N):
        U[n + 1, :] = scheme(F, U[n, :], t[n + 1] - t[n], t[n])
    return U


def partition(a: float, b: float, N: int) -> np.ndarray:
    """N+1 puntos equiespaciados entre a y b (N segmentos)."""
    return np.array([a + i * (b - a) / N for i in range(N + 1)])
