"""
calculo.py  —  ORQUESTADOR (puente entre capas)
Junta la capa física con la capa matemática. No dibuja nada.
Es el único módulo de la GUI que llama a las dos capas a la vez.
"""
from typing import Callable

import numpy as np

from capa_fisica import fisica as f
from capa_fisica import poinsot as po
from capa_matematica import esquemas_temporales as et
from capa_matematica import analisis_numerico as an
from capa_matematica.jacobiano import jacobiano
from capa_matematica.rotaciones import rotacion
from .ajustes import ESQUEMAS, TF, TF_E, T_CONV, N_CONV, w, estado, w_e, estado_e


# =====================================================================
# Pestaña 1: teorema del eje intermedio
# =====================================================================
def orientacion(t: np.ndarray, U: np.ndarray) -> np.ndarray:
    """
    Actitud del satélite: matrices de rotación R_k (cuerpo -> inercial).
    Cada paso compone un giro incremental de vector ω_medio·Δt (Rodrigues).
    Devuelve un array (N+1 x 3 x 3).
    """
    R = [np.eye(3)]
    for k in range(len(t) - 1):
        omega_medio = 0.5 * (U[k] + U[k + 1])
        R.append(R[-1] @ rotacion(omega_medio * (t[k + 1] - t[k])))
    return np.array(R)


def generar_malla_elipsoide(a: float, b: float, c: float,
                            num_puntos: int = 50) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Malla (X, Y, Z) de un elipsoide de semiejes (a, b, c)."""
    u = np.linspace(0, 2 * np.pi, num_puntos)
    v = np.linspace(0, np.pi, num_puntos)
    U, V = np.meshgrid(u, v)
    return a * np.cos(U) * np.sin(V), b * np.sin(U) * np.sin(V), c * np.cos(V)


def superficies_poinsot(I: list[float], dos_T0: float, L0: float,
                        num_puntos: int = 50) -> tuple[tuple, tuple]:
    """Mallas de los elipsoides de energía y de momento (L0 = módulo de L)."""
    ejes_T, ejes_L = po.semiejes_poinsot(I, dos_T0, L0)
    return (generar_malla_elipsoide(*ejes_T, num_puntos),
            generar_malla_elipsoide(*ejes_L, num_puntos))


def ejecutar_simulacion(I: list[float], w0: np.ndarray, dt: float,
                        esquema_func: Callable, TF: float = 20.0
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Integra ω(t), calcula la deriva de 2T y L² y la actitud.
    Independiente de la GUI. Devuelve (t, U, dT, dL, R).
    """
    t = et.partition(0, TF, int(TF / dt))

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    with np.errstate(all='ignore'):          # silencia avisos si un esquema diverge
        U = et.cauchy_problem(F, w0, t, esquema_func)
        dT = an.calcular_deriva(U, f.invariant_energy, I)
        dL = an.calcular_deriva(U, f.invariant_angular_momentum, I)

    # Si el esquema "explota" (p. ej. Euler con Δt grande) nos quedamos con la parte válida
    valido = np.isfinite(U).all(axis=1)
    if not valido.all():
        n = max(int(valido.argmin()), 2)
        t, U, dT, dL = t[:n], U[:n], dT[:n], dL[:n]

    return t, U, dT, dL, orientacion(t, U)


def resolver_teorema() -> None:
    """Lee los controles de la pestaña 1, lanza la simulación y guarda todo en `estado`."""
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    w0 = np.array([w['w1'].val, w['w2'].val, w['w3'].val])
    dt = w['dt'].val
    esquema = ESQUEMAS[w['esquema'].value_selected][0]

    t, U, dT, dL, R = ejecutar_simulacion(I, w0, dt, esquema, TF)
    estado.update(I=I, w0=w0, t=t, U=U, dT=dT, dL=dL, R=R, t_anim=0.0)


# =====================================================================
# Pestaña 2: estabilidad numérica y convergencia
# =====================================================================
def leer_parametros() -> tuple[list[float], np.ndarray, float, str]:
    """Lee la pestaña 2 y construye la condición inicial: giro Ω sobre el eje elegido + ε en los otros."""
    I = [w_e['I1'].val, w_e['I2'].val, w_e['I3'].val]
    puesto = ['eje menor', 'eje intermedio', 'eje mayor'].index(w_e['eje'].value_selected)
    eje = int(np.argsort(I)[puesto])
    w0 = np.full(3, w_e['eps'].val)
    w0[eje] = w_e['Omega'].val
    return I, w0, w_e['dt'].val, w_e['esquema'].value_selected


def equilibrios(I: list[float], Omega: float) -> list[np.ndarray]:
    """Rotación pura de velocidad Ω sobre el eje menor, intermedio y mayor (en ese orden)."""
    lista = []
    for eje in np.argsort(I):
        W = np.zeros(3)
        W[eje] = Omega
        lista.append(W)
    return lista


def resolver_estabilidad() -> None:
    """Simula con el esquema elegido, estima el error y analiza la estabilidad."""
    I, w0, dt, nombre = leer_parametros()
    scheme, q = ESQUEMAS[nombre]
    t = et.partition(0, TF_E, int(round(TF_E / dt)))   # la pestaña 2 sigue con 20 s
    dt = t[1] - t[0]                                  # paso real de la malla

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    # 1. Error de Richardson a lo largo del tiempo
    with np.errstate(all='ignore'):
        U, E = an.richardson_error_temporal(F, w0, t, scheme, q)
    valido = np.isfinite(U).all(axis=1) & np.isfinite(E).all(axis=1)
    if not valido.all():                              # el esquema ha divergido
        n = max(int(valido.argmin()), 2)
        t, U, E = t[:n], U[:n], E[:n]

    # El sistema es autónomo: el Jacobiano solo depende de ω (t da igual)
    def F_omega(omega: np.ndarray) -> np.ndarray:
        return F(omega, 0.0)

    # 2. Estabilidad lineal en los tres equilibrios: z = λ·Δt
    #    J se calcula con la función general jacobiano() de la capa matemática
    eq = []
    for W in equilibrios(I, w_e['Omega'].val):
        maxR, z = an.amplificacion_maxima(scheme, jacobiano(F_omega, W), dt)
        eq.append((maxR, z))

    # 3. Nube de z = λ·Δt a lo largo de la trayectoria (el Jacobiano cambia con ω)
    idx = np.linspace(0, len(U) - 1, min(150, len(U))).astype(int)
    z_tray = np.concatenate([np.linalg.eigvals(jacobiano(F_omega, U[k])) * dt for k in idx])

    estado_e.update(I=I, w0=w0, dt=dt, nombre=nombre, t=t, U=U, E=E,
                    eq=eq, z_tray=z_tray, divergio=not valido.all())


def calcular_convergencia() -> None:
    """Ratio de convergencia de los cuatro esquemas con la condición inicial actual."""
    I, w0, _, _ = leer_parametros()

    def F(omega: np.ndarray, t_val: float) -> np.ndarray:
        return f.euler_equations(omega, I, t_val)

    resultados = {}
    with np.errstate(all='ignore'):
        for nombre, (scheme, q_teo) in ESQUEMAS.items():
            q, d = an.ratio_convergencia(F, w0, T_CONV, N_CONV, scheme)
            resultados[nombre] = (q, d, q_teo)
    estado_e['conv'] = resultados
    estado_e['conv_param'] = (tuple(I), tuple(w0))
