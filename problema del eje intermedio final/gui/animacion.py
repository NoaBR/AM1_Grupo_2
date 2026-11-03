"""
animacion.py  —  pestaña 1
animar(): se llama cada 30 ms y mueve los elementos ya dibujados
(punto del espacio de fases, satélite, cursores de tiempo).

Por qué ahora el giro es fluido:
  1. BLITTING. Antes, en cada fotograma matplotlib volvía a dibujar la figura
     entera (elipsoides, curvas, gráficas...): unos 170-200 ms por fotograma,
     es decir, 5 fotogramas por segundo, y por eso el satélite iba "a saltos".
     Ahora el fondo (todo lo que no se mueve) se guarda como imagen cada vez
     que la figura se redibuja, y en cada fotograma solo se restaura esa imagen
     y se pintan encima los pocos elementos que se mueven.
  2. RELOJ REAL. El tiempo simulado avanza con el reloj del ordenador
     (velocidad × segundos reales), no un número fijo de pasos por fotograma.
  3. INTERPOLACIÓN. El estado se interpola entre los dos pasos de la malla:
     ω de forma lineal y la orientación con un giro de Rodrigues parcial.
     Así el satélite no salta de un paso de la integración al siguiente.
"""
from __future__ import annotations

import time

import numpy as np
import matplotlib.pyplot as plt

from capa_matematica.rotaciones import rotacion
from .ajustes import estado, w, pestanas, TRAZA_SEG

# Elementos que se mueven (se pintan encima del fondo guardado)
ANIMADOS: list = ['recorrido', 'punto', 'sat', 'flecha_w', 'traza', 'reloj',
                  'cursor_T', 'cursor_L']


def marcar_animados() -> None:
    """
    Los elementos animados no se pintan en los redibujados normales (animated=True),
    para que no queden "congelados" en el fondo. En pausa se pintan normal,
    así salen también si se guarda la figura.
    """
    for clave in ANIMADOS:
        if clave in estado:
            estado[clave].set_animated(not estado['pausa'])


# >>> EVENTO RECIBIDO: 'draw_event', la figura se acaba de redibujar entera
#                     (lo provoca cualquier cambio: slider, botón, girar la vista...)
def guardar_fondo(evento: object) -> None:
    """Se ejecuta tras cada redibujado completo: guarda el fondo y pinta lo animado."""
    lienzo = plt.gcf().canvas
    try:
        estado['fondo'] = lienzo.copy_from_bbox(lienzo.figure.bbox)
    except Exception:                       # backend sin blitting: se redibuja entero
        estado['fondo'] = None
    if pestanas['activa'] == 0 and not estado['pausa']:
        pintar_animados()


def pintar_animados() -> None:
    """Pinta los elementos animados encima del lienzo y actualiza la pantalla."""
    lienzo = plt.gcf().canvas
    if 'sat' not in estado:
        return
    estado['sat'].do_3d_projection()        # el polígono 3D se proyecta a mano
    ejes = []
    for clave in ANIMADOS:
        artista = estado[clave]
        artista.axes.draw_artist(artista)
        if artista.axes not in ejes:
            ejes.append(artista.axes)
    for a in ejes:
        lienzo.blit(a.bbox)


def estado_interpolado(ta: float) -> tuple[int, np.ndarray, np.ndarray]:
    """Devuelve (k, ω(ta), R(ta)) con t[k] <= ta < t[k+1]."""
    t, U, R = estado['t'], estado['U'], estado['R']
    k = int(np.clip(np.searchsorted(t, ta, side='right') - 1, 0, len(t) - 2))
    s = (ta - t[k]) / (t[k + 1] - t[k])
    omega = (1 - s) * U[k] + s * U[k + 1]
    R_ta = R[k] @ rotacion(0.5 * (U[k] + U[k + 1]) * (ta - t[k]))   # giro parcial del paso k
    return k, omega, R_ta


def actualizar_animados(ta: float) -> None:
    """Coloca cada elemento animado en el instante ta."""
    k, omega, R_ta = estado_interpolado(ta)
    t, U = estado['t'], estado['U']

    # espacio de fases: punto actual y tramo ya recorrido
    estado['punto'].set_data_3d([omega[0]], [omega[1]], [omega[2]])
    rec = np.vstack((U[:k + 1], omega))
    estado['recorrido'].set_data_3d(rec[:, 0], rec[:, 1], rec[:, 2])

    # satélite: se giran sus vértices con la orientación interpolada
    estado['sat'].set_verts(list(estado['geom'] @ R_ta.T))
    wk = R_ta @ omega * estado['escala_w']                  # ω expresada en ejes inerciales
    estado['flecha_w'].set_data_3d([0, wk[0]], [0, wk[1]], [0, wk[2]])
    k0 = int(np.searchsorted(t, ta - TRAZA_SEG))            # traza de los últimos segundos
    p = np.vstack((estado['punta'][k0:k + 1], R_ta[:, estado['eje_traza']] * 1.2))
    estado['traza'].set_data_3d(p[:, 0], p[:, 1], p[:, 2])
    estado['reloj'].set_text(f't = {ta:5.2f} s')

    # cursores de tiempo en las gráficas de deriva
    estado['cursor_T'].set_xdata([ta, ta])
    estado['cursor_L'].set_xdata([ta, ta])


# >>> EVENTO RECIBIDO: tic del temporizador (cada 30 ms). No lo provoca el
#                     usuario: es el reloj que mueve la animación.
def animar() -> None:
    """Se llama cada 30 ms con el temporizador de la figura."""
    ahora = time.perf_counter()
    dt_real = min(ahora - estado.get('reloj_real', ahora), 0.1)   # evita saltos tras una pausa
    estado['reloj_real'] = ahora
    if ('R' not in estado or estado['pausa'] or pestanas['activa'] != 0
            or len(estado['t']) < 2):
        return

    t = estado['t']
    velocidad = w['vel'].val * (2 if estado['x2'] else 1)      # botón ×2
    estado['t_anim'] = (estado['t_anim'] + velocidad * dt_real) % t[-1]
    actualizar_animados(estado['t_anim'])

    lienzo = plt.gcf().canvas
    if estado.get('fondo') is None:          # sin fondo guardado: redibujado normal
        lienzo.draw_idle()
        return
    lienzo.restore_region(estado['fondo'])
    pintar_animados()
