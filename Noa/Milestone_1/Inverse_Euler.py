from numpy import array, zeros
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

N = 1000
Dt = 0.01
Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])


for n in range(0, N):
    f = lambda x: x - U[n, :] - Dt * F(x)
    root = fsolve(f, U[n, :])
    U[n + 1, :] = root

plt.plot(U[:, 0], U[:, 1])
plt.show()