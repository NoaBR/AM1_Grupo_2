"""
rotaciones.py  —  CAPA MATEMÁTICA
Matrices de rotación. La usa la GUI para calcular la orientación del satélite
a partir de ω(t): cada paso se compone un giro de vector ω·Δt.
"""
import numpy as np


def rotacion(v: np.ndarray) -> np.ndarray:
    """Matriz de rotación de ángulo |v| alrededor del eje v (fórmula de Rodrigues)."""
    ang = np.linalg.norm(v)
    if ang < 1e-12:
        return np.eye(3)
    k = v / ang
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * (K @ K)
