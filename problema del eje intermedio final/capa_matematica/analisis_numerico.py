"""
analisis_numerico.py  —  CAPA MATEMÁTICA
Estabilidad absoluta (región |R| <= 1 calculada con el propio esquema), error por
extrapolación de Richardson, ratio de convergencia y deriva de invariantes.
No sabe nada de la física: todo recibe F o z como argumento.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy import array, zeros, linspace, float64

from .esquemas_temporales import cauchy_problem, FuncionF

# ---------------------------------------------------------------------------
# Estabilidad absoluta, calculada directamente con el esquema (vale para CUALQUIERA)
#   Se aplica un paso del esquema al problema modelo dU/dt = w·U con Δt = 1
#   y U = 1: el resultado es el factor de amplificación r = R(w).
#   El esquema es estable en w si |r| <= 1.
#   No hace falta conocer la fórmula de R(z) de cada esquema.
# ---------------------------------------------------------------------------
def factor_amplificacion(Scheme: Callable, w: complex) -> float:
    """|R(w)|: un paso de Scheme sobre dU/dt = w·U, con U = 1 y Δt = 1."""
    try:
        with np.errstate(all='ignore'):
            r = Scheme(F=lambda U, t: w * U, U=array([1.0 + 0j]), At=1., t=0.)
        return float(abs(r[0]))
    except (ValueError, ZeroDivisionError, OverflowError):
        return float('inf')            # p. ej. Euler inverso en w = 1 (sistema singular)


def Stability_Region(Scheme: Callable, N: int, x0: float, xf: float,
                     y0: float, yf: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    |R(w)| en una malla N x N del plano complejo w = x + i·y.
    rho[i, j] corresponde a (x[i], y[j]); para contourf hay que trasponer: transpose(rho).
    Adaptación del código de clase a la firma de nuestros esquemas,
    Scheme(F, U, At, t): por eso se llama con At en vez de dt.
    """
    x, y = linspace(x0, xf, N), linspace(y0, yf, N)
    rho = zeros((N, N), dtype=float64)

    for i in range(N):
        for j in range(N):
            w = complex(x[i], y[j])
            rho[i, j] = factor_amplificacion(Scheme, w)

    return rho, x, y


def amplificacion_maxima(Scheme: Callable, J: np.ndarray,
                         dt: float) -> tuple[float, np.ndarray]:
    """
    Autovalores λ_i de J -> z_i = λ_i Δt -> max |R(z_i)|, con R calculado aplicando
    el propio esquema (factor_amplificacion). Si es <= 1 el paso es estable.
    Devuelve (max|R|, z).

    Los autovalores nulos se ignoran en el máximo: R(0) = 1 en todos los
    esquemas consistentes (son direcciones neutras). "Nulo" con margen relativo,
    porque con un Jacobiano numérico el 0 sale como ~1e-10.
    """
    lam = np.linalg.eigvals(J)
    z = lam * dt
    no_nulos = np.abs(lam) > 1e-6 * max(np.abs(lam).max(), 1e-300)
    if not no_nulos.any():
        return 1.0, z
    return max(factor_amplificacion(Scheme, complex(zi)) for zi in z[no_nulos]), z


# ---------------------------------------------------------------------------
# Error de Richardson y ratio de convergencia
# ---------------------------------------------------------------------------
def richardson_error_temporal(F: FuncionF, U0: np.ndarray, t: np.ndarray,
                              scheme: Callable, q: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Error de la solución con malla t, estimado con una malla el doble de fina:
        E(t_n) = (U_2N(t_n) - U_N(t_n)) / (1 - 1/2^q)
    Devuelve (U_N, E), ambos (N+1 x Nv).
    """
    t2 = np.zeros(2 * len(t) - 1)
    t2[0::2] = t
    t2[1::2] = (t[:-1] + t[1:]) / 2
    U1 = cauchy_problem(F, U0, t, scheme)
    U2 = cauchy_problem(F, U0, t2, scheme)
    return U1, (U2[0::2] - U1) / (1 - 1 / 2**q)


def ratio_convergencia(F: FuncionF, U0: np.ndarray, T: float, N_values: list[int],
                       scheme: Callable) -> tuple[float, np.ndarray]:
    """
    d(N) = |U_2N(T) - U_N(T)| ~ C / N^q  ->  q = -pendiente de log10 d frente a log10 N.
    Descarta los puntos en el error de redondeo (d < 1e-11) y, si sobran,
    el primero (malla aún no asintótica).  Devuelve (q, d).
    """
    d = []
    for N in N_values:
        U_N = cauchy_problem(F, U0, np.linspace(0, T, N + 1), scheme)
        U_2N = cauchy_problem(F, U0, np.linspace(0, T, 2 * N + 1), scheme)
        d.append(np.linalg.norm(U_2N[-1] - U_N[-1]))
    d = np.array(d)

    with np.errstate(all='ignore'):
        logN, logd = np.log10(np.array(N_values, dtype=float)), np.log10(d)
    validos = np.isfinite(logd) & (logd > -11)
    if validos.sum() >= 3:
        validos[np.argmax(validos)] = False
    if validos.sum() < 2:
        return float('nan'), d
    pendiente, _ = np.polyfit(logN[validos], logd[validos], 1)
    return -pendiente, d


# ---------------------------------------------------------------------------
# Deriva de invariantes
# ---------------------------------------------------------------------------
def calcular_deriva(U_simulado: np.ndarray,
                    funcion_invariante: Callable[[np.ndarray, list[float]], float],
                    parametros_fisicos: list[float]) -> np.ndarray:
    """Deriva absoluta |I(t) - I(0)| de cualquier invariante a lo largo de la solución."""
    valor_referencia = funcion_invariante(U_simulado[0, :], parametros_fisicos)
    deriva = np.zeros(len(U_simulado))
    for i in range(len(U_simulado)):
        deriva[i] = abs(funcion_invariante(U_simulado[i, :], parametros_fisicos)
                        - valor_referencia)
    return deriva
