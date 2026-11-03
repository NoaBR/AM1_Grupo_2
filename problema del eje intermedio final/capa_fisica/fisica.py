"""
fisica.py  —  CAPA FÍSICA: el modelo
Ecuaciones de Euler del sólido rígido libre y sus dos integrales primeras
(energía 2T y momento angular L²). Es el problema del eje intermedio en sí.
La geometría que se deduce de estas leyes (elipsoides, separatriz,
polhodias) está en poinsot.py.

No importa nada de la capa numérica ni de la GUI: solo física.
El Jacobiano NO está aquí: lo calcula la capa matemática (jacobiano.py)
para cualquier sistema de ecuaciones.
Convención: omega = [ω1, ω2, ω3] en ejes cuerpo, I = [I1, I2, I3].
"""
import numpy as np


def euler_equations(omega: np.ndarray, I: list[float], t: float) -> np.ndarray:
    """dω/dt = F(ω, t) para el sólido rígido libre (ecuaciones de Euler)."""
    domega_dt = np.zeros(3)
    domega_dt[0] = (I[1] - I[2]) * omega[1] * omega[2] / I[0]
    domega_dt[1] = (I[2] - I[0]) * omega[2] * omega[0] / I[1]
    domega_dt[2] = (I[0] - I[1]) * omega[0] * omega[1] / I[2]
    return domega_dt


def invariant_energy(omega: np.ndarray, I: list[float]) -> float:
    """2T = I1 ω1² + I2 ω2² + I3 ω3²   (devuelve 2T, no T)."""
    return I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2


def invariant_angular_momentum(omega: np.ndarray, I: list[float]) -> float:
    """L² = I1² ω1² + I2² ω2² + I3² ω3²   (devuelve L², no L)."""
    return I[0]**2 * omega[0]**2 + I[1]**2 * omega[1]**2 + I[2]**2 * omega[2]**2
