# gui_principal.py
# ---------------------------------------------------------------------------
#  Interfaz con dos pestañas:
#    1. TEOREMA DEL EJE INTERMEDIO: espacio de fases, satélite y derivas
#    2. ESTABILIDAD NUMÉRICA: regiones |R(z)| <= 1, autovalores, Richardson y convergencia
#
#  Se lanza desde main.py (carpeta de arriba).
#
#  Módulos de la GUI (funciones sencillas, sin clases):
#    ajustes.py             colores, constantes y diccionarios compartidos
#    calculo.py             orquestador: junta la capa física y la matemática
#    modelo_satelite.py     geometría del satélite
#    dibujo_teorema.py      paneles de la pestaña 1
#    animacion.py           animar(): mueve el satélite y el punto de fases (pestaña 1)
#    panel_teorema.py       controles de la pestaña 1 y al_cambiar_teorema()
#    dibujo_estabilidad.py  paneles de la pestaña 2
#    panel_estabilidad.py   controles de la pestaña 2 y al_cambiar_estabilidad()
#    pestanas.py            botones de pestaña y cambiar_pestana()
#
#  MAPA DE EVENTOS (busca ">>> EVENTO" y ">>> CONEXIÓN" en los archivos):
#    · CONEXIÓN = la línea donde se dice "cuando pase X, llama a la función Y"
#                 (on_changed, on_clicked, mpl_connect, add_callback)
#    · EVENTO RECIBIDO = la función Y, que es la que se ejecuta cuando pasa X
#
#    Qué hace el usuario               Se conecta en          Lo recibe
#    -------------------------------   --------------------   --------------------------
#    Pestaña 1: mover slider / esquema panel_teorema.py       al_cambiar_teorema()
#    Pestaña 1: botones de eje         panel_teorema.py       preset()
#    Pestaña 1: Pausa / Reanudar       panel_teorema.py       pausar()
#    Pestaña 1: botón ×2               panel_teorema.py       duplicar_velocidad()
#    Pestaña 1: vista interm. / 3D     panel_teorema.py       cambiar_vista()
#    Pestaña 1: casillas del mapa      panel_teorema.py       al_marcar()
#    Pestaña 2: mover slider / opción  panel_estabilidad.py   al_cambiar_estabilidad()
#    Pestaña 2: Calcular convergencia  panel_estabilidad.py   al_pulsar_convergencia()
#    Cambiar de pestaña                pestanas.py            cambiar_pestana()
#    (automático) figura redibujada    gui_principal.py       guardar_fondo()
#    (automático) reloj cada 30 ms     gui_principal.py       animar()
# ---------------------------------------------------------------------------
import matplotlib.pyplot as plt

from .ajustes import ax, ax_e, FONDO, PANEL, TEXTO
from .animacion import animar, guardar_fondo
from .panel_teorema import crear_controles_teorema, al_cambiar_teorema
from .panel_estabilidad import crear_controles_estabilidad, al_cambiar_estabilidad
from .pestanas import crear_pestanas, registrar_grupo, cambiar_pestana


def crear_pestana_teorema(fig: plt.Figure) -> None:
    """Pestaña 1: fases | actitud | (deriva T arriba, deriva L abajo)."""
    gs = fig.add_gridspec(2, 3, left=0.04, right=0.98, top=0.85, bottom=0.36,
                          wspace=0.15, hspace=0.35, width_ratios=[1.25, 1.15, 1])
    ax['fases'] = fig.add_subplot(gs[:, 0], projection='3d')
    ax['actitud'] = fig.add_subplot(gs[:, 1], projection='3d')
    ax['T'] = fig.add_subplot(gs[0, 2])
    ax['L'] = fig.add_subplot(gs[1, 2])
    crear_controles_teorema()


def crear_pestana_estabilidad(fig: plt.Figure) -> None:
    """Pestaña 2: región completa | zoom | convergencia ; Richardson | diagnóstico."""
    gs = fig.add_gridspec(2, 3, left=0.05, right=0.98, top=0.85, bottom=0.35,
                          wspace=0.18, hspace=0.42, height_ratios=[2.0, 1])
    ax_e['region'] = fig.add_subplot(gs[0, 0])
    ax_e['zoom'] = fig.add_subplot(gs[0, 1])
    ax_e['conv'] = fig.add_subplot(gs[0, 2])
    ax_e['richardson'] = fig.add_subplot(gs[1, 0:2])
    ax_e['diag'] = fig.add_subplot(gs[1, 2])
    crear_controles_estabilidad()


def lanzar_gui() -> plt.Figure:
    plt.style.use('dark_background')
    plt.rcParams.update({'font.family': 'monospace', 'font.size': 9,
                         'figure.facecolor': FONDO, 'axes.facecolor': PANEL,
                         'text.color': TEXTO, 'axes.labelcolor': TEXTO,
                         'grid.color': '#2b3652', 'axes.edgecolor': '#2b3652'})

    fig = plt.figure(figsize=(16, 9), facecolor=FONDO)
    fig.text(0.04, 0.955, 'DINÁMICA DE ACTITUD SATELITAL · TEOREMA DEL EJE INTERMEDIO',
             fontsize=17, fontweight='bold', color='white')
    crear_pestanas(['1 · Teorema del eje intermedio', '2 · Estabilidad numérica'])

    # Cada pestaña registra los paneles, widgets y textos que crea
    for i, crear in enumerate((crear_pestana_teorema, crear_pestana_estabilidad)):
        ejes_antes, textos_antes = list(fig.axes), list(fig.texts)
        crear(fig)
        registrar_grupo(i, ejes_antes, textos_antes)

    al_cambiar_teorema()            # primer cálculo y dibujo de las dos pestañas
    al_cambiar_estabilidad()
    cambiar_pestana(0)

    # Animación con blitting: tras cada redibujado se guarda el fondo, y un
    # temporizador llama a animar() cada 30 ms. Hay que guardarlo en una variable.
    fig.canvas.mpl_connect('draw_event', guardar_fondo)    # >>> CONEXIÓN: tras cada redibujado
    fig.temporizador = fig.canvas.new_timer(interval=30)
    fig.temporizador.add_callback(animar)                  # >>> CONEXIÓN: cada 30 ms
    fig.temporizador.start()
    plt.show()
    return fig
