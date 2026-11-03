"""
panel_estabilidad.py  —  pestaña 2
Sliders, selectores y botón, y la función al_cambiar_estabilidad(),
que se ejecuta cada vez que el usuario toca un control.
"""
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons, Slider

from .ajustes import w_e as w, ESQUEMAS, FONDO, TEXTO, AMARILLO, CIAN, MORADO
from .calculo import resolver_estabilidad, calcular_convergencia
from .dibujo_estabilidad import (dibujar_regiones, dibujar_zoom, dibujar_richardson,
                    dibujar_convergencia, dibujar_diagnostico)


# >>> EVENTO RECIBIDO: mover un slider (I₁, I₂, I₃, Ω, ε, Δt) o elegir otro
#                     esquema / eje de giro. El valor nuevo no se usa: se leen
#                     todos los controles en resolver_estabilidad().
def al_cambiar_estabilidad(_: float | str | None = None) -> None:
    """Recalcula la simulación, el error y la estabilidad, y redibuja."""
    resolver_estabilidad()
    dibujar_regiones()
    dibujar_zoom()
    dibujar_richardson()
    dibujar_convergencia()          # solo cambia el aviso de «parámetros anteriores»
    dibujar_diagnostico()
    plt.gcf().canvas.draw_idle()


# >>> EVENTO RECIBIDO: clic en «Calcular convergencia»
def al_pulsar_convergencia(_: object) -> None:
    """La convergencia es cara (muchas integraciones): solo se calcula al pulsar el botón."""
    w['b_conv'].label.set_text('Calculando...')
    plt.gcf().canvas.draw()
    plt.pause(0.01)
    calcular_convergencia()
    w['b_conv'].label.set_text('Calcular convergencia (≈10 s)')
    dibujar_convergencia()
    plt.gcf().canvas.draw_idle()


def crear_slider(rect: list[float], nombre: str, vmin: float, vmax: float,
                 inicial: float, color: str, formato: str = '%.2f') -> Slider:
    s = Slider(plt.axes(rect, facecolor='#26324a'), nombre, vmin, vmax,
               valinit=inicial, valfmt=formato, color=color, track_color='#2b3652')
    s.label.set_color(TEXTO)
    s.valtext.set_color(color)
    s.on_changed(al_cambiar_estabilidad)        # >>> CONEXIÓN: al mover el slider
    return s


def crear_radio(rect: list[float], opciones: list[str], activo: int,
                color: str) -> RadioButtons:
    a = plt.axes(rect, facecolor=FONDO)
    for borde in a.spines.values():
        borde.set_visible(False)
    try:                                   # sin blitting interno: si no, sus círculos
        r = RadioButtons(a, opciones, active=activo, activecolor=color,
                         useblit=False)    # se pintan aunque la pestaña esté oculta
    except TypeError:                      # matplotlib < 3.7 no tiene useblit
        r = RadioButtons(a, opciones, active=activo, activecolor=color)
    for etiqueta in r.labels:
        etiqueta.set_color(TEXTO)
    r.on_clicked(al_cambiar_estabilidad)        # >>> CONEXIÓN: al elegir una opción
    return r


def crear_controles_estabilidad() -> None:
    for x, titulo, color in ((0.06, 'MOMENTOS DE INERCIA', AMARILLO),
                             (0.38, 'CONDICIÓN INICIAL Y PASO', CIAN),
                             (0.70, 'INTEGRADOR  ·  EJE DE GIRO', MORADO)):
        plt.gcf().text(x, 0.285, titulo, color=color, fontweight='bold', fontsize=10)

    w['I1'] = crear_slider([0.09, 0.22, 0.22, 0.03], 'I₁', 0.1, 5.0, 1.0, AMARILLO)
    w['I2'] = crear_slider([0.09, 0.17, 0.22, 0.03], 'I₂', 0.1, 5.0, 2.0, AMARILLO)
    w['I3'] = crear_slider([0.09, 0.12, 0.22, 0.03], 'I₃', 0.1, 5.0, 3.0, AMARILLO)

    w['Omega'] = crear_slider([0.41, 0.22, 0.22, 0.03], 'Ω', 0.1, 5.0, 1.0, CIAN)
    w['eps'] = crear_slider([0.41, 0.17, 0.22, 0.03], 'ε', 0.0, 0.3, 0.05, CIAN, '%.3f')
    w['dt'] = crear_slider([0.41, 0.12, 0.22, 0.03], 'Δt', 0.005, 1.0, 0.05, CIAN, '%.3f')

    w['esquema'] = crear_radio([0.70, 0.04, 0.13, 0.20], list(ESQUEMAS), 3, MORADO)
    w['eje'] = crear_radio([0.84, 0.08, 0.13, 0.16],
                           ['eje menor', 'eje intermedio', 'eje mayor'], 1, MORADO)

    w['b_conv'] = Button(plt.axes([0.09, 0.03, 0.22, 0.05]), 'Calcular convergencia (≈10 s)',
                         color='#26324a', hovercolor='#34456a')
    w['b_conv'].label.set_color(TEXTO)
    w['b_conv'].on_clicked(al_pulsar_convergencia)      # >>> CONEXIÓN: clic en el botón
