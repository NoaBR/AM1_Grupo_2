"""
pestanas.py
Pestañas hechas con botones: cada pestaña es un grupo de paneles y widgets.
Cambiar de pestaña = mostrar un grupo y ocultar el otro (set_visible)
y desactivar los widgets de la pestaña oculta (set_active).

OJO: ocultar no basta. Los botones de matplotlib (Button) reciben el clic
aunque su panel esté oculto, y como los de las dos pestañas ocupan el mismo
sitio, el botón oculto "robaba" el clic al visible. Por eso se desactivan.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Widget

from .ajustes import pestanas, w, w_e, TEXTO, GRIS, MORADO


# >>> EVENTO RECIBIDO: clic en uno de los botones de pestaña de arriba
def cambiar_pestana(i: int) -> None:
    """Muestra la pestaña i y oculta las demás."""
    pestanas['activa'] = i
    for j, grupo in enumerate(pestanas['grupos']):
        for elemento in grupo:
            elemento.set_visible(j == i)
    for j, widgets in enumerate((w, w_e)):          # solo responde la pestaña visible
        for widget in widgets.values():
            # Widget.set_active (la versión base): en RadioButtons y CheckButtons
            # widget.set_active(n) significa "marca la opción n", que no es lo que queremos
            Widget.set_active(widget, j == i)
    for j, b in enumerate(pestanas['botones']):
        activo = j == i
        b.color = MORADO if activo else '#26324a'
        b.ax.set_facecolor(b.color)
        b.label.set_color('black' if activo else TEXTO)
        b.label.set_fontweight('bold' if activo else 'normal')
    plt.gcf().canvas.draw_idle()


def crear_pestanas(nombres: list[str]) -> None:
    """Crea un botón por pestaña en la parte de arriba de la ventana."""
    for i, nombre in enumerate(nombres):
        b = Button(plt.axes([0.04 + 0.235 * i, 0.895, 0.225, 0.04]), nombre,
                   color='#26324a', hovercolor='#34456a')
        b.label.set_color(TEXTO)
        b.label.set_fontsize(10)
        b.on_clicked(lambda _, i=i: cambiar_pestana(i))     # >>> CONEXIÓN: clic en la pestaña
        pestanas['botones'].append(b)


def registrar_grupo(i: int, ejes_antes: list, textos_antes: list) -> None:
    """Guarda en la pestaña i todo lo que se ha creado desde ejes_antes / textos_antes."""
    fig = plt.gcf()
    pestanas['grupos'][i] += [a for a in fig.axes if a not in ejes_antes]
    pestanas['grupos'][i] += [t for t in fig.texts if t not in textos_antes]
