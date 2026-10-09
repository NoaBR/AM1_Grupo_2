from numpy import array

def RK4_function(F: callable, U0: array, Dt: float) -> array:
    k1 = (U0, )
    k2 = (U0 + 0.5 * Dt * F(*k1), )
    k3 = (U0 + 0.5 * Dt * F(*k2), )
    k4 = (U0 + Dt * F(*k3), )
    return U0 + (Dt / 6) * (F(*k1) + 2 * F(*k2) + 2 * F(*k3) + F(*k4))