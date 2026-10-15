from numpy import array, zeros
from numpy.linalg import norm, solve
from scipy.optimize import fsolve
from Milestone_2.Functions.non_linear_solver import Jacobian
from Milestone_2.Functions.non_linear_solver import newton_function

def Euler_Function(F: callable, U0: array, Dt: float) -> array:


    return U0 + Dt * F(U0)

def Crank_function(F: callable, U0: array, Dt: float) -> array:
    Y = U0
    while norm(Y - U0 - Dt/2*(F(Y) + F(U0))) > 1e-10:
        R = Y - U0 - Dt/2*(F(Y) + F(U0))
        Y = Y - R
        print(norm(R))
    return  Y

def Inverse_Euler_Function( F: callable, U0: array,  Dt: float) -> array:

    f = lambda x: x - U0 - Dt * F(x)
    root = fsolve(f, U0)
    return root

def RK4_function(F: callable, U0: array, Dt: float) -> array:
    k1 = (U0, )
    k2 = (U0 + 0.5 * Dt * F(*k1), )
    k3 = (U0 + 0.5 * Dt * F(*k2), )
    k4 = (U0 + Dt * F(*k3), )
    return U0 + (Dt / 6) * (F(*k1) + 2 * F(*k2) + 2 * F(*k3) + F(*k4))

def Crank_Nicolson_function(F: callable, U0: array, Dt: float) -> array:
    def G(x):
        return x - U0 - Dt/2*(F(x) + F(U0))

    return newton_function(G, U0)





