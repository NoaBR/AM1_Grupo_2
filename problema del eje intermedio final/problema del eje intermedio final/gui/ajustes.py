"""
ajustes.py
Colores, constantes y los diccionarios compartidos por todos los módulos de la GUI.

Pestaña 1 (teorema):      ax,   w,   estado
Pestaña 2 (estabilidad):  ax_e, w_e, estado_e
Pestañas:                 pestanas

Como son diccionarios, cuando un módulo escribe en ellos el resto lo ve.
"""
from capa_matematica import esquemas_temporales as et

FONDO, PANEL, TEXTO, GRIS = '#0b1020', '#131a2e', '#e2e8f0', '#7c8aa5'
ROJO, AZUL, VERDE, AMARILLO, ROSA, CIAN, MORADO = (
    '#ff5a5f', '#4aa8ff', '#a3e635', '#fbbf24', '#f472b6', '#4dd0e1', '#b388ff')

# nombre -> (esquema, orden teórico q). La región de estabilidad se calcula
# aplicando el propio esquema (analisis_numerico.Stability_Region).
ESQUEMAS: dict = {
    'Euler':          (et.euler,           1),
    'Euler inverso':  (et.euler_inverso,   1),
    'Crank-Nicolson': (et.crank_nicholson, 2),
    'RK4':            (et.rk4,             4),
}
N_REGION: int = 120          # malla N x N de la región completa (se calcula una vez)
N_ZOOM: int = 60             # malla del zoom (se recalcula al cambiar parámetros)
COLOR_ESQUEMA: dict = {'Euler': ROJO, 'Euler inverso': AZUL,
                       'Crank-Nicolson': VERDE, 'RK4': MORADO}

# Colores de los tres equilibrios: 0 = eje menor, 1 = intermedio, 2 = mayor
COLOR_EJE: list = [CIAN, AMARILLO, ROSA]
NOMBRE_EJE: list = ['eje menor', 'eje intermedio', 'eje mayor']

# ---------------- Pestaña 1: teorema ----------------
TF: float = 50.0               # duración de la simulación de la pestaña 1 [s]
                               # (con I = 1, 2, 3 da para ver el volteo de ida, t ≈ 11 s,
                               #  y el de vuelta, t ≈ 30 s)
VEL_ANIM: float = 0.6          # segundos simulados por segundo real (valor inicial del slider)
NMESH: int = 25                # resolución de las superficies de Poinsot
TRAZA_SEG: float = 12.0        # segundos de traza del eje que se muestran en el satélite

# ---------------- Pestaña 2: estabilidad ----------------
TF_E: float = 20.0                              # duración de la simulación de la pestaña 2 [s]
T_CONV: float = 20.0                            # tiempo final del estudio de convergencia
N_CONV: list = [200, 400, 800, 1600, 3200]      # pasos de las mallas de convergencia
LIM_REGION: tuple = ((-4.0, 2.0), (-3.5, 3.5))  # ventana del plano z en el panel completo

# ---------------- Diccionarios compartidos ----------------
ax: dict = {}          # pestaña 1: paneles
w: dict = {}           # pestaña 1: widgets
estado: dict = {'pausa': False, 'x2': False,
                'ver': {'energia': True, 'momento': True,
                        'polhodias': True, 'separatriz': True}}

ax_e: dict = {}        # pestaña 2: paneles
w_e: dict = {}         # pestaña 2: widgets
estado_e: dict = {}

pestanas: dict = {'activa': 0, 'grupos': [[], []], 'botones': []}
