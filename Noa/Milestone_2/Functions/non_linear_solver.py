from numpy import array, zeros
from numpy.linalg import norm, solve
from Jacobian import Jacobian

def newton_function(G: callable, U0: array) -> array:
    x = U0.copy()
    for _ in range(50):
        residual = G(x)
        if norm(residual) < 1e-10:
            return x
        J = Jacobian(G, x, h=1e-8)
        x = x - solve(J, residual)

    raise RuntimeError("Newton's method did not converge after 50 iterations")

