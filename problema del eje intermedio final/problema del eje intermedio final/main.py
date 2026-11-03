"""
main.py  —  ARCHIVO QUE SE EJECUTA (desde VS Code: botón ▶ o F5)

Problema del eje intermedio, organizado en tres capas:
    capa_fisica/       el problema: fisica.py (ecuaciones de Euler e invariantes
                       2T y L²) y poinsot.py (D, elipsoides, separatriz, polhodias)
    capa_matematica/   esquemas temporales, Jacobiano (para cualquier sistema),
                       Newton y Gauss (esquemas implícitos),
                       estabilidad R(z), Richardson y convergencia
    gui/               interfaz con dos pestañas (teorema y estabilidad numérica)

Este archivo solo lanza la interfaz. Ejecútalo desde esta carpeta
para que Python encuentre los tres paquetes.
"""
from gui.gui_principal import lanzar_gui

if __name__ == '__main__':
    lanzar_gui()
