"""
dibujo_estabilidad.py  —  pestaña 2
Funciones que dibujan cada panel de la figura:
    dibujar_regiones()      regiones de estabilidad |R(z)| <= 1 (vista completa)
    dibujar_zoom()          zoom alrededor de los z = λ·Δt del problema
    dibujar_richardson()    error estimado E(t) por Richardson
    dibujar_convergencia()  log|U_2N - U_N| frente a log N, con el orden q
    dibujar_diagnostico()   resumen en texto
"""
import numpy as np
from matplotlib.axes import Axes
from matplotlib.lines import Line2D

from capa_matematica import analisis_numerico as an
from .ajustes import (ax_e as ax, estado_e as estado, ESQUEMAS, COLOR_ESQUEMA, COLOR_EJE, NOMBRE_EJE,
                     PANEL, TEXTO, GRIS, AMARILLO, N_CONV, LIM_REGION, N_REGION, N_ZOOM)


def estilo_2d(a: Axes, titulo: str, color: str = 'white') -> None:
    a.set_facecolor(PANEL)
    a.set_title(titulo, loc='left', color=color, fontweight='bold', fontsize=11)
    a.grid(True, color='#26324a', linestyle=':')
    a.tick_params(labelsize=8, colors=GRIS)
    for borde in a.spines.values():
        borde.set_color('#26324a')


_REGIONES: dict = {}     # caché: (esquema, ventana, N) -> (rho, x, y)


def _malla(nombre: str, x_lim: tuple, y_lim: tuple, N: int) -> tuple:
    """
    Región |R| <= 1 calculada con Stability_Region (aplicando el propio esquema).
    Se guarda en caché: la ventana completa no cambia, así que se calcula una sola vez.
    """
    clave = (nombre, tuple(np.round(x_lim, 6)), tuple(np.round(y_lim, 6)), N)
    if clave not in _REGIONES:
        _REGIONES[clave] = an.Stability_Region(ESQUEMAS[nombre][0], N, *x_lim, *y_lim)
    return _REGIONES[clave]


def _region(a: Axes, nombre: str, x_lim: tuple, y_lim: tuple, rellenar: bool,
            N: int = N_REGION) -> None:
    """Dibuja |R(z)| <= 1 de un esquema: relleno si es el elegido, solo el borde si no."""
    rho, x, y = _malla(nombre, x_lim, y_lim, N)
    X, Y, absR = x, y, np.transpose(rho)          # rho[i, j] es (x[i], y[j])
    color = COLOR_ESQUEMA[nombre]
    if rellenar:
        a.contourf(X, Y, absR, levels=[0, 1], colors=[color], alpha=0.25)
        a.contour(X, Y, absR, levels=[1], colors=[color], linewidths=2)
    else:
        a.contour(X, Y, absR, levels=[1], colors=[color], linewidths=1,
                  linestyles='--', alpha=0.6)


def _puntos_z(a: Axes, tamano: float) -> None:
    """z = λ·Δt de la trayectoria (nube blanca) y de los tres equilibrios."""
    z = estado['z_tray']
    a.plot(z.real, z.imag, '.', color='white', ms=2, alpha=0.35)
    for puesto, (_, z_eq) in enumerate(estado['eq']):
        a.plot(z_eq.real, z_eq.imag, 'o', color=COLOR_EJE[puesto], ms=tamano,
               mec='black', mew=0.5)


def dibujar_regiones() -> None:
    a = ax['region']
    a.clear()
    estilo_2d(a, 'REGIONES DE ESTABILIDAD  |R(z)| ≤ 1')
    (x0, x1), (y0, y1) = LIM_REGION
    elegido = estado['nombre']
    for nombre in ESQUEMAS:
        if nombre != elegido:
            _region(a, nombre, (x0, x1), (y0, y1), rellenar=False)
    _region(a, elegido, (x0, x1), (y0, y1), rellenar=True)
    _puntos_z(a, 6)
    a.axhline(0, color=GRIS, lw=0.6); a.axvline(0, color=GRIS, lw=0.6)
    a.set_xlim(x0, x1); a.set_ylim(y0, y1); a.set_aspect('equal', adjustable='box')
    a.set_xlabel('Re(z)', color=GRIS); a.set_ylabel('Im(z)', color=GRIS)
    leyenda = [Line2D([], [], color=COLOR_ESQUEMA[n], lw=2 if n == elegido else 1,
                      ls='-' if n == elegido else '--', label=n) for n in ESQUEMAS]
    a.legend(handles=leyenda, loc='lower left', fontsize=7, frameon=False)


def dibujar_zoom() -> None:
    """Zoom alrededor de los z = λ·Δt: aquí se ve si caen dentro o fuera de la región."""
    a = ax['zoom']
    a.clear()
    estilo_2d(a, 'ZOOM  z = λ·Δt  DEL PROBLEMA')
    todos = np.concatenate([estado['z_tray']] + [z for _, z in estado['eq']])
    L = max(1.4 * np.abs(todos).max(), 0.02)
    _region(a, estado['nombre'], (-L, L), (-L, L), rellenar=True, N=N_ZOOM)
    _puntos_z(a, 9)
    a.axhline(0, color=GRIS, lw=0.6); a.axvline(0, color=GRIS, lw=0.6)
    a.set_xlim(-L, L); a.set_ylim(-L, L); a.set_aspect('equal', adjustable='box')
    a.set_xlabel('Re(z)', color=GRIS)
    leyenda = [Line2D([], [], marker='o', ls='', color=COLOR_EJE[p], label=NOMBRE_EJE[p])
               for p in range(3)]
    leyenda.append(Line2D([], [], marker='.', ls='', color='white', label='trayectoria'))
    a.legend(handles=leyenda, loc='lower left', fontsize=7, frameon=False)


def dibujar_richardson() -> None:
    a = ax['richardson']
    a.clear()
    nombre = estado['nombre']
    color = COLOR_ESQUEMA[nombre]
    estilo_2d(a, f'ERROR DE RICHARDSON  |E(t)|  ·  {nombre}, Δt = {estado["dt"]:.3f}', color)
    t, E = estado['t'], np.linalg.norm(estado['E'], axis=1)
    a.semilogy(t, np.maximum(E, 1e-17), color=color, lw=2)
    a.fill_between(t, 1e-17, np.maximum(E, 1e-17), color=color, alpha=0.15)
    a.set_ylim(bottom=max(np.maximum(E, 1e-17).min() / 10, 1e-17))
    a.set_xlabel('Tiempo [s]', color=GRIS); a.set_ylabel('error estimado', color=GRIS)
    if estado['divergio']:
        a.text(0.98, 0.9, 'EL ESQUEMA HA DIVERGIDO', transform=a.transAxes, ha='right',
               color='#ff5a5f', fontweight='bold')


def dibujar_convergencia() -> None:
    a = ax['conv']
    a.clear()
    estilo_2d(a, 'CONVERGENCIA  log|U₂ₙ − Uₙ|')
    a.set_xlabel('log₁₀ N', color=GRIS)
    if 'conv' not in estado:
        a.text(0.5, 0.5, 'Pulsa «Calcular convergencia»', transform=a.transAxes,
               ha='center', va='center', color=GRIS)
        return
    logN = np.log10(N_CONV)
    for nombre, (q, d, q_teo) in estado['conv'].items():
        with np.errstate(all='ignore'):
            a.plot(logN, np.log10(d), 'o-', color=COLOR_ESQUEMA[nombre], lw=1.5, ms=4,
                   label=f'{nombre}: q = {q:.2f} ({q_teo})')
    a.legend(loc='lower left', fontsize=7, frameon=False)
    if estado['conv_param'] != (tuple(estado['I']), tuple(estado['w0'])):
        a.text(0.98, 0.95, 'calculado con parámetros anteriores', transform=a.transAxes,
               ha='right', va='top', color=AMARILLO, fontsize=8)


def dibujar_diagnostico() -> None:
    a = ax['diag']
    a.clear()
    a.set_facecolor(PANEL)
    a.set_xticks([]); a.set_yticks([])
    for borde in a.spines.values():
        borde.set_color('#26324a')
    a.set_title('DIAGNÓSTICO', loc='left', color='white', fontweight='bold', fontsize=11)

    nombre = estado['nombre']
    a.text(0.04, 0.88, f'{nombre}   Δt = {estado["dt"]:.3f}', transform=a.transAxes,
           color=COLOR_ESQUEMA[nombre], fontweight='bold')
    a.text(0.04, 0.76, 'Estabilidad lineal en cada equilibrio:', transform=a.transAxes,
           color=GRIS, fontsize=8)
    for puesto, (maxR, z) in enumerate(estado['eq']):
        # λ real positivo: silla (margen relativo por el ruido del Jacobiano numérico)
        fisica_inestable = np.max(z.real) > 1e-6 * max(np.abs(z).max(), 1e-300)
        if fisica_inestable and maxR <= 1 + 1e-9:
            # λ real > 0 (la física crece) pero los dos autovalores caen dentro de la región
            veredicto = 'decae (falsa estabilidad)'
        elif fisica_inestable:
            veredicto = 'crece (inestable físico)'
        elif abs(maxR - 1) < 1e-9:
            veredicto = 'neutro: conserva'
        elif maxR < 1:
            veredicto = 'amortigua (disipa)'
        else:
            veredicto = 'AMPLIFICA (num.)'
        a.text(0.04, 0.64 - 0.11 * puesto,
               f'{NOMBRE_EJE[puesto]:15s} max|R| = {maxR:.6f}  {veredicto}',
               transform=a.transAxes, color=COLOR_EJE[puesto], fontsize=8)
    texto = ('Ejes menor y mayor: el valor exacto es |R| = 1\n'
             '(giro estable que ni crece ni decae).\n'
             'Eje intermedio: |R| > 1 es la física correcta.')
    a.text(0.04, 0.08, texto, transform=a.transAxes, color=GRIS, fontsize=7.5)