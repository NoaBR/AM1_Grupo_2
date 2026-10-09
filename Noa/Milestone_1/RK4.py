from numpy import array, zeros
import matplotlib.pyplot as plt

N = 1000
Dt = 0.01

Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])

for n in range(0,N):
    k1 = (U[n, :], )
    k2 = (U[n, :] + 0.5 * Dt * F(*k1), )
    k3 = (U[n, :] + 0.5 * Dt * F(*k2), )
    k4 = (U[n, :] + Dt * F(*k3), )
    U[n + 1, :] = U[n, :] + (Dt / 6) * (F(*k1) + 2 * F(*k2) + 2 * F(*k3) + F(*k4))


plt.plot(U[:, 0], U[:, 1])
plt.show()