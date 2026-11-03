"""
panel_teorema.py  —  pestaña 1
Sliders, botones, selector de integrador y casillas del mapa de fases,
y la función al_cambiar_teorema(), que se ejecuta cada vez que el usuario toca un control.
"""
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, CheckButtons, RadioButtons, Slider

from .ajustes import w, estado, ESQUEMAS, FONDO, TEXTO, AMARILLO, CIAN, MORADO, VERDE, VEL_ANIM
from .calculo import resolver_teorema
from .dibujo_teorema import dibujar_fases, dibujar_actitud, dibujar_derivas, vista
from .animacion import marcar_animados, actualizar_animados


# >>> EVENTO RECIBIDO: mover un slider (I₁, I₂, I₃, ω₁(0), ω₂(0), ω₃(0), Δt)
#                     o elegir otro integrador. Matplotlib llama a esta función
#                     y le pasa el valor nuevo (número o texto), que no se usa:
#                     se leen todos los controles de nuevo en resolver_teorema().
def al_cambiar_teorema(_: float | str | None = None) -> None:
    """Se ejecuta cada vez que el usuario toca un control de la física o del integrador."""
    if np.linalg.norm([w['w1'].val, w['w2'].val, w['w3'].val]) < 1e-9:
        return                                   # con ω(0) = 0 no hay movimiento
    resolver_teorema()
    dibujar_fases()
    dibujar_actitud()
    dibujar_derivas()
    marcar_animados()
    actualizar_animados(estado['t_anim'])
    plt.gcf().canvas.draw_idle()


def crear_slider(rect: list[float], nombre: str, vmin: float, vmax: float,
                 inicial: float, color: str, formato: str = '%.2f',
                 al_mover: Callable | None = al_cambiar_teorema) -> Slider:
    s = Slider(plt.axes(rect, facecolor='#26324a'), nombre, vmin, vmax,
               valinit=inicial, valfmt=formato, color=color,
               track_color='#2b3652')
    s.label.set_color(TEXTO)
    s.valtext.set_color(color)
    if al_mover is not None:
        s.on_changed(al_mover)           # >>> CONEXIÓN: al mover el slider -> al_mover()
    return s


# >>> EVENTO RECIBIDO: clic en «eje menor», «eje interm.» o «eje mayor»
def preset(puesto: int) -> None:
    """Pone ω(0) casi alineada con un eje: 0 = menor inercia, 1 = intermedio, 2 = mayor."""
    eje = int(np.argsort([w['I1'].val, w['I2'].val, w['I3'].val])[puesto])
    valores = [0.01, 0.01, 0.01]
    valores[eje] = 1.0
    for nombre, v in zip(('w1', 'w2', 'w3'), valores):
        w[nombre].eventson = False               # evita recalcular tres veces
        w[nombre].set_val(v)
        w[nombre].eventson = True
    al_cambiar_teorema()


# >>> EVENTO RECIBIDO: clic en «Pausa» / «Reanudar»
def pausar(_: object) -> None:
    estado['pausa'] = not estado['pausa']
    w['b_pausa'].label.set_text('▶ Reanudar' if estado['pausa'] else '|| Pausa')
    marcar_animados()                 # en pausa se dibujan normal (salen al guardar la figura)
    plt.gcf().canvas.draw_idle()


# >>> EVENTO RECIBIDO: clic en «×2»: dobla la velocidad de la animación o la
#                     devuelve a la normal. El botón se ilumina mientras está activo.
def duplicar_velocidad(_: object) -> None:
    estado['x2'] = not estado['x2']
    b = w['b_x2']
    b.color = VERDE if estado['x2'] else '#26324a'
    b.ax.set_facecolor(b.color)
    b.label.set_color('black' if estado['x2'] else TEXTO)
    b.label.set_fontweight('bold' if estado['x2'] else 'normal')
    plt.gcf().canvas.draw_idle()


# >>> EVENTO RECIBIDO: clic en «vista eje interm.» o «vista 3D»
def cambiar_vista(eje_intermedio: bool) -> None:
    """Mira el mapa de fases a lo largo del eje intermedio (se ve la X) o en 3D."""
    I = [w['I1'].val, w['I2'].val, w['I3'].val]
    vista(int(np.argsort(I)[1]) if eje_intermedio else None)
    plt.gcf().canvas.draw_idle()


# >>> EVENTO RECIBIDO: marcar o desmarcar una casilla del mapa de fases
#                     (Energía 2T, Momento L², Polhodias, Separatriz).
#                     Matplotlib le pasa el texto de la casilla pulsada.
def al_marcar(etiqueta: str) -> None:
    """Muestra u oculta una capa del mapa de fases (no hace falta recalcular)."""
    clave = CAPAS[etiqueta]
    estado['ver'][clave] = not estado['ver'][clave]
    estado['indicadores'][clave].set_visible(estado['ver'][clave])   # círculo verde on/off
    dibujar_fases()
    marcar_animados()
    actualizar_animados(estado['t_anim'])
    plt.gcf().canvas.draw_idle()


def crear_opciones(tipo: type, *args, **kwargs) -> RadioButtons | CheckButtons:
    """
    Crea RadioButtons o CheckButtons SIN su blitting interno (useblit=False).
    Con blitting, matplotlib pinta los círculos aunque su pestaña esté oculta,
    y se superponen con los de la otra pestaña.
    """
    try:
        return tipo(*args, useblit=False, **kwargs)
    except TypeError:                      # matplotlib < 3.7 no tiene useblit
        return tipo(*args, **kwargs)


# Casillas del mapa de fases: texto que se ve -> clave en estado['ver']
CAPAS: dict = {'Energía 2T': 'energia', 'Momento L²': 'momento',
               'Polhodias': 'polhodias', 'Separatriz': 'separatriz'}


def crear_indicadores(ax_check: plt.Axes) -> None:
    """
    Un círculo verde a la derecha de cada casilla: se ve si la capa está
    activa y se oculta si no. Se guardan en estado['indicadores'].
    """
    estado['indicadores'] = {}
    for etiqueta in w['capas'].labels:
        clave = CAPAS[etiqueta.get_text()]
        y = etiqueta.get_position()[1]                 # misma altura que el texto
        circulo, = ax_check.plot([0.92], [y], 'o', color=VERDE, ms=9,
                                 mec='black', mew=0.8, transform=ax_check.transAxes)
        circulo.set_visible(estado['ver'][clave])
        estado['indicadores'][clave] = circulo


def crear_boton(rect: list[float], texto: str) -> Button:
    b = Button(plt.axes(rect), texto, color='#26324a', hovercolor='#34456a')
    b.label.set_color(TEXTO)
    b.label.set_fontsize(8.5)
    return b


def crear_controles_teorema() -> None:
    # Títulos de cada grupo
    for x, titulo, color in ((0.06, 'MOMENTOS DE INERCIA', AMARILLO),
                             (0.38, 'VELOCIDAD ANGULAR INICIAL', CIAN),
                             (0.70, 'INTEGRADOR · ANIMACIÓN · MAPA', MORADO)):
        plt.gcf().text(x, 0.285, titulo, color=color, fontweight='bold', fontsize=10)

    # Sliders: columna 1 (inercias) y columna 2 (ω inicial)
    w['I1'] = crear_slider([0.09, 0.22, 0.22, 0.03], 'I₁', 0.1, 5.0, 1.0, AMARILLO)
    w['I2'] = crear_slider([0.09, 0.17, 0.22, 0.03], 'I₂', 0.1, 5.0, 2.0, AMARILLO)
    w['I3'] = crear_slider([0.09, 0.12, 0.22, 0.03], 'I₃', 0.1, 5.0, 3.0, AMARILLO)
    w['w1'] = crear_slider([0.41, 0.22, 0.22, 0.03], 'ω₁(0)', -5.0, 5.0, 0.01, CIAN)
    w['w2'] = crear_slider([0.41, 0.17, 0.22, 0.03], 'ω₂(0)', -5.0, 5.0, 1.0, CIAN)
    w['w3'] = crear_slider([0.41, 0.12, 0.22, 0.03], 'ω₃(0)', -5.0, 5.0, 0.01, CIAN)

    # Columna 3: paso, velocidad de la animación, integrador y capas del mapa
    w['dt'] = crear_slider([0.73, 0.22, 0.22, 0.03], 'Δt', 0.005, 0.2, 0.05, MORADO, '%.3f')
    w['vel'] = crear_slider([0.73, 0.17, 0.22, 0.03], 'vel. anim.', 0.1, 3.0, VEL_ANIM,
                            MORADO, '%.2f×', al_mover=None)   # solo la lee animar()

    ax_radio = plt.axes([0.70, 0.01, 0.12, 0.14], facecolor=FONDO)
    for borde in ax_radio.spines.values():
        borde.set_visible(False)
    w['esquema'] = crear_opciones(RadioButtons, ax_radio, list(ESQUEMAS),
                                  active=3, activecolor=MORADO)
    for etiqueta in w['esquema'].labels:
        etiqueta.set_color(TEXTO)
    w['esquema'].on_clicked(al_cambiar_teorema)       # >>> CONEXIÓN: elegir integrador

    ax_check = plt.axes([0.84, 0.01, 0.13, 0.14], facecolor=FONDO)
    for borde in ax_check.spines.values():
        borde.set_visible(False)
    w['capas'] = crear_opciones(CheckButtons, ax_check,
                                list(CAPAS), [True, True, True, True])
    for etiqueta in w['capas'].labels:
        etiqueta.set_color(TEXTO)
    w['capas'].on_clicked(al_marcar)                  # >>> CONEXIÓN: casillas del mapa
    crear_indicadores(ax_check)

    # Botones: pausa, vistas del mapa y atajos de condiciones iniciales
    w['b_pausa'] = crear_boton([0.09, 0.04, 0.06, 0.05], '|| Pausa')
    w['b_pausa'].on_clicked(pausar)                                   # >>> CONEXIÓN
    w['b_x2'] = crear_boton([0.155, 0.04, 0.035, 0.05], '×2')
    w['b_x2'].on_clicked(duplicar_velocidad)                          # >>> CONEXIÓN
    w['b_vista2'] = crear_boton([0.195, 0.04, 0.075, 0.05], 'vista eje interm.')
    w['b_vista2'].on_clicked(lambda _: cambiar_vista(True))           # >>> CONEXIÓN
    w['b_vista3d'] = crear_boton([0.275, 0.04, 0.045, 0.05], 'vista 3D')
    w['b_vista3d'].on_clicked(lambda _: cambiar_vista(False))         # >>> CONEXIÓN
    w['b_min'] = crear_boton([0.41, 0.04, 0.07, 0.05], 'eje menor')
    w['b_mid'] = crear_boton([0.49, 0.04, 0.07, 0.05], 'eje interm.')
    w['b_max'] = crear_boton([0.57, 0.04, 0.07, 0.05], 'eje mayor')
    w['b_min'].on_clicked(lambda _: preset(0))                        # >>> CONEXIÓN
    w['b_mid'].on_clicked(lambda _: preset(1))                        # >>> CONEXIÓN
    w['b_max'].on_clicked(lambda _: preset(2))                        # >>> CONEXIÓN
