from numpy import array
from scipy.optimize import fsolve

def Inverse_Euler_Function( Dt: float, F: callable, U0: array) -> array:

    f = lambda x: x - U0 - Dt * F(x)
    root = fsolve(f, U0)
    return root