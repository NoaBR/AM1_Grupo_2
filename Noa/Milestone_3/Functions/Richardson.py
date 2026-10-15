from numpy import array, zeros, ones, linspace, sin, cos, pi

def error_richardson(order: int, coarse: array, fine: array) -> float:

     E = (fine - coarse) / (2**order - 1)

     return E
