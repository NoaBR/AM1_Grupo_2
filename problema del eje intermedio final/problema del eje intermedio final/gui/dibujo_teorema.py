"""
dibujo_teorema.py  —  pestaña 1
Funciones que dibujan cada panel:
    dibujar_fases()    espacio de fases 3D (mapa de Poinsot)
    dibujar_actitud()  satélite 3D
    dibujar_derivas()  errores ΔT(t) y ΔL(t)
Se llaman cada vez que cambia un parámetro. Dejan en `estado` los elementos
vacíos que luego animar() irá moviendo.
"""
from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from capa_fisica import fisica as f
from capa_fisica import poinsot as po
from .calculo import superficies_poinsot
from .ajustes import (ax, estado, FONDO, PANEL, TEXTO, GRIS, ROJO, AZUL, VERDE,
                      AMARILLO, ROSA, CIAN, NMESH, NOMBRE_EJE)
from .modelo_satelite import geometria_satelite


def estilo_3d(a: Axes, titulo: str) -> None:
    a.set_facecolor(PANEL)
    a.set_title(titulo, loc='left', color='white', fontweight='bold', fontsize=11)
    for eje in (a.xaxis, a.yaxis, a.zaxis):
        eje.set_pane_color((1, 1, 1, 0))
    a.tick_params(labelsize=7, colors=GRIS)


def estilo_2d(a: Axes, titulo: str, color: str) -> None:
    a.set_facecolor(PANEL)
    a.set_title(titulo, loc='left', color=color, fontweight='bold', fontsize=11)
    a.grid(True, color='#26324a', linestyle=':')
    a.tick_params(labelsize=8, colors=GRIS)
    for borde in a.spines.values():
        borde.set_color('#26324a')


def clasificar(I: list[float], D: float) -> tuple[str, str]:
    """Texto y color que explican qué movimiento corresponde a D = L²/2T."""
    Ia, Ib, Ic = np.sort(I)
    if Ib - Ia < 1e-9 * Ic or Ic - Ib < 1e-9 * Ic:
        return 'dos inercias iguales: no hay eje intermedio', GRIS
    if abs(D - Ib) < 0.01 * (Ic - Ia):
        return 'D ≈ I_int: sobre la separatriz → VOLTEO periódico', AMARILLO
    if D < Ib:
        return 'I_mín < D < I_int: gira alrededor del eje menor (centro)', CIAN
    return 'I_int < D < I_máx: gira alrededor del eje mayor (centro)', ROSA


def dibujar_fases() -> None:
    """
    Mapa de fases en (ω1, ω2, ω3):
      · elipsoide de energía (rojo) y de momento (malla azul) de ESTA trayectoria
      · polhodias de referencia: curvas cerradas alrededor del eje menor (cian)
        y del mayor (rosa) sobre el mismo elipsoide de energía
      · separatriz (verde), que separa las dos familias
      · equilibrios: centros (cian / rosa) y puntos de silla (blanco con borde rojo).
        Son los vértices del elipsoide de energía.
      · trayectoria simulada (blanco) y estado actual (amarillo)
    """
    a = ax['fases']
    elev, azim = a.elev, a.azim
    a.clear()
    estilo_3d(a, 'ESPACIO DE FASES  (ω₁, ω₂, ω₃)')
    I, w0, U, ver = estado['I'], estado['w0'], estado['U'], estado['ver']
    i_min, i_int, i_max = np.argsort(I)
    Ia, Ib, Ic = np.sort(I)

    # Constantes del movimiento de ESTA trayectoria
    dos_T0 = f.invariant_energy(w0, I)                       # 2T
    L0 = np.sqrt(f.invariant_angular_momentum(w0, I))        # |L| (módulo, no L²)
    D = po.cociente_D(w0, I)                                  # L² / 2T
    ejes_T, ejes_L = po.semiejes_poinsot(I, dos_T0, L0)
    malla_T, malla_L = superficies_poinsot(I, dos_T0, L0, num_puntos=NMESH)
    lim = 1.05 * max(ejes_T.max(), ejes_L.max())

    # 1. Ejes principales, con su nombre según la inercia
    puesto = {i_min: 'menor', i_int: 'interm.', i_max: 'mayor'}
    for j in range(3):
        p = np.zeros((2, 3))
        p[:, j] = [-lim, lim]
        a.plot(p[:, 0], p[:, 1], p[:, 2], color=GRIS, lw=0.7, ls=':')
        a.text(*(p[1] * 1.05), f'ω{j + 1} ({puesto[j]})', color=GRIS, fontsize=7)

    # 2. Superficies invariantes
    if ver['energia']:
        a.plot_surface(*malla_T, color=ROJO, alpha=0.13, linewidth=0, shade=False)
    if ver['momento']:
        a.plot_wireframe(*malla_L, color=AZUL, alpha=0.25, linewidth=0.5,
                         rstride=3, cstride=3)

    # 3. Polhodias de referencia sobre el mismo elipsoide de energía
    if ver['polhodias'] and Ib - Ia > 1e-9 * Ic and Ic - Ib > 1e-9 * Ic:
        for fr in (0.25, 0.55, 0.85):
            for curva in po.polhodia(I, dos_T0, Ia + fr * (Ib - Ia), 160):
                a.plot(curva[:, 0], curva[:, 1], curva[:, 2], color=CIAN, lw=0.9, alpha=0.6)
            for curva in po.polhodia(I, dos_T0, Ic - fr * (Ic - Ib), 160):
                a.plot(curva[:, 0], curva[:, 1], curva[:, 2], color=ROSA, lw=0.9, alpha=0.6)

    # 4. Separatriz
    if ver['separatriz']:
        lazos = po.separatriz(I, dos_T0)
        if lazos is not None:
            for lazo in lazos:
                a.plot(lazo[:, 0], lazo[:, 1], lazo[:, 2], color=VERDE, lw=2.5, ls='--')

    # 5. Equilibrios: vértices del elipsoide de energía
    for j, color, borde, marca in ((i_min, CIAN, 'black', 'o'), (i_max, ROSA, 'black', 'o'),
                                   (i_int, 'white', ROJO, 'X')):
        p = np.zeros((2, 3))
        p[:, j] = [ejes_T[j], -ejes_T[j]]
        a.plot(p[:, 0], p[:, 1], p[:, 2], marca, color=color, mec=borde, mew=1.5, ms=8)

    # 6. Trayectoria simulada y elementos animados
    a.plot(U[:, 0], U[:, 1], U[:, 2], color='white', alpha=0.45, lw=1.2)
    estado['recorrido'], = a.plot([], [], [], color='white', lw=2)
    estado['punto'], = a.plot([], [], [], 'o', color=AMARILLO, mec='black', ms=9)

    # 7. Texto explicativo y leyenda
    texto, color = clasificar(I, D)
    a.text2D(0.01, 0.06, f'D = L²/2T = {D:.3f}    I = {Ia:.2f} · {Ib:.2f} · {Ic:.2f}',
             transform=a.transAxes, color=TEXTO, fontsize=8)
    a.text2D(0.01, 0.01, texto, transform=a.transAxes, color=color, fontsize=8,
             fontweight='bold')
    leyenda = [Patch(color=ROJO, alpha=0.4, label='elipsoide energía 2T'),
               Line2D([], [], color=AZUL, label='elipsoide momento L²'),
               Line2D([], [], color=CIAN, label='giros eje menor'),
               Line2D([], [], color=ROSA, label='giros eje mayor'),
               Line2D([], [], color=VERDE, ls='--', lw=2, label='separatriz'),
               Line2D([], [], marker='X', ls='', color='white', mec=ROJO, label='puntos de silla'),
               Line2D([], [], color='white', label='trayectoria'),
               Line2D([], [], marker='o', ls='', color=AMARILLO, label='estado actual')]
    a.legend(handles=leyenda, loc='upper right', fontsize=6.5, frameon=False)

    a.set_xlim(-lim, lim); a.set_ylim(-lim, lim); a.set_zlim(-lim, lim)
    a.set_xlabel('ω₁'); a.set_ylabel('ω₂'); a.set_zlabel('ω₃')
    a.set_box_aspect((1, 1, 1), zoom=1.3)
    a.view_init(elev, azim)


def dibujar_actitud() -> None:
    """Panel del satélite: se crean los elementos; animar() los mueve."""
    a = ax['actitud']
    a.clear()
    a.set_axis_off()
    I, w0, U = estado['I'], estado['w0'], estado['U']

    # Título que avisa de si el giro inicial es estable o no
    eje = int(np.argmax(np.abs(w0)))                 # eje en torno al que gira al principio
    puesto = list(np.argsort(I)).index(eje)          # 0 = menor inercia, 1 = intermedio, 2 = mayor
    texto, color = [('estable (eje de menor inercia)', VERDE),
                    ('INESTABLE (eje intermedio)', ROJO),
                    ('estable (eje de mayor inercia)', VERDE)][puesto]
    a.set_title('ACTITUD DEL SATÉLITE', loc='left', color='white',
                fontweight='bold', fontsize=11)
    a.text2D(0.02, 0.95, f'Giro inicial en el eje {eje + 1}: {texto}', transform=a.transAxes,
             color=color, fontweight='bold', fontsize=9)

    # Cuerpo con las proporciones de la caja equivalente a estas inercias
    caras, colores, realizable = geometria_satelite(I)
    estado['geom'] = caras
    if not realizable:
        a.text2D(0.02, 0.89, 'Aviso: un sólido rígido necesita que cada inercia no\n'
                 'supere la suma de las otras dos', transform=a.transAxes,
                 color=AMARILLO, fontsize=8)
    estado['sat'] = Poly3DCollection(caras, facecolors=colores,
                                     edgecolors=FONDO, linewidths=0.5, alpha=0.95)
    a.add_collection3d(estado['sat'])

    # L es constante en el espacio inercial: se dibuja fijo
    L = np.array(I) * U[0]
    L = 1.4 * L / np.linalg.norm(L)
    a.plot([0, L[0]], [0, L[1]], [0, L[2]], color=AMARILLO, lw=2.5, ls='--', label='L (fijo)')

    estado['eje_traza'] = eje
    estado['flecha_w'], = a.plot([], [], [], color='white', lw=2.5, label='ω')
    estado['traza'], = a.plot([], [], [], color=ROSA, lw=1.5, label=f'eje {eje + 1} (traza)')
    estado['punta'] = estado['R'][:, :, eje] * 1.2      # posición de la punta de ese eje
    estado['escala_w'] = 1.2 / max(np.linalg.norm(U, axis=1).max(), 1e-12)
    estado['reloj'] = a.text2D(0.03, 0.04, '', transform=a.transAxes, color=TEXTO)

    a.set_xlim(-1.5, 1.5); a.set_ylim(-1.5, 1.5); a.set_zlim(-1.5, 1.5)
    a.set_box_aspect((1, 1, 1), zoom=1.5)
    a.legend(loc='lower right', fontsize=8, frameon=False)


def dibujar_derivas() -> None:
    """Errores ΔT(t) y ΔL(t) en escala logarítmica."""
    t = estado['t']
    for clave, dato, titulo, color in (('T', estado['dT'], 'DERIVA DE ENERGÍA  ΔT(t)', ROJO),
                                       ('L', estado['dL'], 'DERIVA DE MOMENTO  ΔL(t)', AZUL)):
        a = ax[clave]
        a.clear()
        estilo_2d(a, titulo, color)
        a.semilogy(t, np.maximum(dato, 1e-17), color=color, lw=2)   # 1e-17 evita log(0)
        a.fill_between(t, 1e-17, np.maximum(dato, 1e-17), color=color, alpha=0.15)
        a.set_ylabel('error absoluto', color=GRIS)
        a.set_xlim(0, t[-1])
        estado['cursor_' + clave] = a.axvline(0, color='white', lw=1, alpha=0.7)
    ax['L'].set_xlabel('Tiempo [s]', color=GRIS)


def vista(eje: int | None) -> None:
    """Orienta la cámara del espacio de fases: mirando por un eje (0, 1, 2) o en 3D (None)."""
    a = ax['fases']
    if eje is None:
        a.view_init(25, -60)
    else:
        a.view_init(*[(0, 0), (0, 90), (90, -90)][eje])
