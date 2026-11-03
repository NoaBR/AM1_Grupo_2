"""
modelo_satelite.py
Geometría del satélite (cuerpo + paneles solares) hecha con cajas.

Las dimensiones del cuerpo dependen de los momentos de inercia: se dibuja la
CAJA HOMOGÉNEA EQUIVALENTE, la que tiene exactamente esos tres momentos.
Para una caja de masa m y semilados (a, b, c) en los ejes 1, 2, 3:
    I1 = m/3·(b² + c²),   I2 = m/3·(a² + c²),   I3 = m/3·(a² + b²)
y despejando:
    a² ∝ I2 + I3 − I1,    b² ∝ I1 + I3 − I2,    c² ∝ I1 + I2 − I3
El eje de MENOR inercia es el lado MÁS LARGO. Los paneles solares son decorativos.
"""
import numpy as np

from .ajustes import ROJO

TAMANO_MAX: float = 0.5        # semilado más largo del cuerpo, en el dibujo
GROSOR_MIN: float = 0.05       # semilado mínimo, para que una placa no desaparezca


def caja(centro: list[float], semilados: list[float]) -> np.ndarray:
    """Las 6 caras (4 vértices cada una) de un ortoedro."""
    vertices = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)])
    vertices = vertices * semilados + centro
    caras = [[0, 1, 3, 2], [4, 5, 7, 6],      # caras x = -1, +1
             [0, 1, 5, 4], [2, 3, 7, 6],      # caras y = -1, +1
             [0, 2, 6, 4], [1, 3, 7, 5]]      # caras z = -1, +1
    return vertices[caras]


def semilados_caja(I: list[float]) -> tuple[np.ndarray, bool]:
    """
    Semilados (a, b, c) de la caja homogénea con momentos de inercia I, escalados
    para el dibujo. Devuelve también si I es realizable por un sólido rígido:
    hace falta que cada momento no supere la suma de los otros dos.
    """
    I = np.asarray(I, dtype=float)
    cuadrados = I.sum() - 2 * I              # (I2+I3−I1, I1+I3−I2, I1+I2−I3)
    # ">= 0" con margen: el caso límite (p. ej. I3 = I1 + I2) es una placa plana, que sí existe
    realizable = bool(np.all(cuadrados >= -1e-9 * I.sum()))
    s = np.sqrt(np.maximum(cuadrados, 0.0))
    s = TAMANO_MAX * s / max(s.max(), 1e-12)
    return np.maximum(s, GROSOR_MIN), realizable


def geometria_satelite(I: list[float]) -> tuple[np.ndarray, list[str], bool]:
    """
    Caras del satélite en ejes cuerpo: cuerpo con las proporciones de la caja
    equivalente + dos paneles solares a lo largo del eje 2.
    Devuelve (caras, colores, realizable).
    """
    a, b, c = semilados_caja(I)[0]
    realizable = semilados_caja(I)[1]
    largo = 0.4                                      # semilongitud de cada panel
    panel = [0.6 * a, largo, 0.015]
    caras = np.concatenate([
        caja([0, 0, 0], [a, b, c]),
        caja([0, -(b + largo + 0.03), 0], panel),
        caja([0, b + largo + 0.03, 0], panel),
    ])
    # Colores: cara x = rojo, y = verde, z = azul (como los ejes 1, 2, 3); paneles azul oscuro
    colores = [ROJO] * 2 + ['#69db7c'] * 2 + ['#4dabf7'] * 2 + ['#1e40af'] * 12
    return caras, colores, realizable
