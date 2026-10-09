from numpy import array, zeros
from numpy.linalg import norm, solve
from scipy.optimize import fsolve

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

def newton_function(G: callable, U0: array) -> array:
    x = U0.copy()
    for _ in range(50):
        residual = G(x)
        if norm(residual) < 1e-10:
            return x
        J = Jacobian(G, x, h=1e-8)
        x = x - solve(J, residual)

    raise RuntimeError("Newton's method did not converge after 50 iterations")

def Jacobian(G: callable, x: array, h: float) -> array:
    n = len(x)
    J = zeros((n, n))
    for i in range(n):
        e = zeros(n)
        e[i] = 1
        x_plus = x + h * e
        x_minus = x - h * e
        J[:, i] = (G(x_plus) - G(x_minus)) / (2 * h)
    return J


def oscilador(U: array) -> array:
    return array([U[1], -U[0]])

def partition(a: float, b: float, N: int) -> list:
    return [a + i*(b-a)/N for i in range(N+1)]