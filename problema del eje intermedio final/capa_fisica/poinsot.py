"""
poinsot.py  —  CAPA FÍSICA: geometría de las soluciones
Lo que se DEDUCE del modelo (fisica.py) sin integrar nada: la construcción
geométrica de Poinsot.

    cociente_D        D = L²/2T, decide qué curva recorre ω
    semiejes_poinsot  radios de los elipsoides de energía y de momento
    separatriz        frontera entre giros alrededor del eje menor y del mayor
    polhodia          curvas cerradas que recorre ω para un D dado

Todo depende solo de los momentos de inercia y de los invariantes, y son
resultados exactos: no usa ningún método numérico ni sabe nada de la GUI.
Convención: omega = [ω1, ω2, ω3] en ejes cuerpo, I = [I1, I2, I3].
"""
from __future__ import annotations

import numpy as np

from .fisica import invariant_energy, invariant_angular_momentum


def cociente_D(omega: np.ndarray, I: list[float]) -> float:
    """
    D = L² / 2T: decide qué polhodia recorre ω.
        I_min < D < I_int  -> gira alrededor del eje menor
        D = I_int          -> separatriz
        I_int < D < I_max  -> gira alrededor del eje mayor
    """
    return invariant_angular_momentum(omega, I) / invariant_energy(omega, I)


def semiejes_poinsot(I: list[float], dos_T0: float,
                     L0: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Semiejes de las dos superficies invariantes en el espacio (ω1, ω2, ω3):
        elipsoide de energía   2T = Σ Ii ωi²    ->  semiejes sqrt(2T / Ii)
        elipsoide de momento   L² = Σ Ii² ωi²   ->  semiejes L / Ii
    OJO: L0 es el MÓDULO del momento angular (no L²).
    """
    I = np.asarray(I, dtype=float)
    return np.sqrt(dos_T0 / I), L0 / I


def separatriz(I: list[float], dos_T0: float,
               num_puntos: int = 300) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Separatriz crítica sobre el elipsoide de energía 2T0, calculada de forma
    analítica (sin integrar nada).

    Es la polhodia con L² = 2 I_int T, la que pasa por los dos puntos de silla
    (rotación pura sobre el eje intermedio, ±Ω_s con Ω_s = sqrt(2T0 / I_int)).
    Está formada por dos elipses planas contenidas en los planos
        ω_max = ± k ω_min,     k = sqrt( I_min (I_int - I_min) / (I_max (I_max - I_int)) )
    y parametrizadas por θ ∈ [0, 2π]:
        ω_min = A cos θ,   ω_int = Ω_s sin θ,   ω_max = ± C cos θ
        A = Ω_s sqrt( I_int (I_max - I_int) / (I_min (I_max - I_min)) )
        C = Ω_s sqrt( I_int (I_int - I_min) / (I_max (I_max - I_min)) )

    Funciona aunque los momentos no estén ordenados (I1 < I2 < I3 no hace falta):
    se busca qué eje es el intermedio. Si dos momentos son iguales no hay eje
    intermedio y devuelve None.

    Devuelve (lazo_1, lazo_2), cada uno un array (num_puntos x 3).
    """
    I = np.asarray(I, dtype=float)
    i_min, i_int, i_max = np.argsort(I)
    Ia, Ib, Ic = I[i_min], I[i_int], I[i_max]
    if Ib - Ia < 1e-9 * Ic or Ic - Ib < 1e-9 * Ic:
        return None

    Omega_s = np.sqrt(dos_T0 / Ib)
    A = Omega_s * np.sqrt(Ib * (Ic - Ib) / (Ia * (Ic - Ia)))
    C = Omega_s * np.sqrt(Ib * (Ib - Ia) / (Ic * (Ic - Ia)))
    theta = np.linspace(0, 2 * np.pi, num_puntos)

    lazos = []
    for signo in (1, -1):
        lazo = np.zeros((num_puntos, 3))
        lazo[:, i_min] = A * np.cos(theta)
        lazo[:, i_int] = Omega_s * np.sin(theta)
        lazo[:, i_max] = signo * C * np.cos(theta)
        lazos.append(lazo)
    return lazos[0], lazos[1]


def polhodia(I: list[float], dos_T0: float, D: float,
             num_puntos: int = 200) -> list[np.ndarray]:
    """
    Polhodia de nivel D = L²/2T sobre el elipsoide de energía 2T0, de forma analítica.
    Son las curvas cerradas que recorre ω. Devuelve una lista con las dos ramas
    (una alrededor de cada polo del eje que rodea), cada una (num_puntos x 3).

    Si I_min < D < I_int rodea el eje menor. Eliminando ω_min de las dos
    conservaciones queda una elipse en (ω_int, ω_max):
        I_int (I_int - I_min) ω_int² + I_max (I_max - I_min) ω_max² = 2T (D - I_min)
    y ω_min se despeja de la energía (dos signos = dos ramas).
    Si I_int < D < I_max rodea el eje mayor, con los papeles cambiados.
    Devuelve [] si D no está en ninguno de esos intervalos.
    """
    I = np.asarray(I, dtype=float)
    i_min, i_int, i_max = np.argsort(I)
    Ia, Ib, Ic = I[i_min], I[i_int], I[i_max]
    theta = np.linspace(0, 2 * np.pi, num_puntos)

    if Ia < D < Ib:              # alrededor del eje menor
        polo, e1, e2 = i_min, i_int, i_max
        r = dos_T0 * (D - Ia)
        a = np.sqrt(r / (Ib * (Ib - Ia)))
        b = np.sqrt(r / (Ic * (Ic - Ia)))
    elif Ib < D < Ic:            # alrededor del eje mayor
        polo, e1, e2 = i_max, i_min, i_int
        r = dos_T0 * (Ic - D)
        a = np.sqrt(r / (Ia * (Ic - Ia)))
        b = np.sqrt(r / (Ib * (Ic - Ib)))
    else:
        return []

    ramas = []
    for signo in (1, -1):
        curva = np.zeros((num_puntos, 3))
        curva[:, e1] = a * np.cos(theta)
        curva[:, e2] = b * np.sin(theta)
        resto = dos_T0 - I[e1] * curva[:, e1]**2 - I[e2] * curva[:, e2]**2
        curva[:, polo] = signo * np.sqrt(np.maximum(resto, 0) / I[polo])
        ramas.append(curva)
    return ramas
