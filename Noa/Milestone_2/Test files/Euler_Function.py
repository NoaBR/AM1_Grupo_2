from numpy import array, zeros
import matplotlib.pyplot as plt


def Euler_Function( Dt: float, F: callable, U0: array) -> array:


    return U0 + Dt * F(U0)


N = 1000
Dt = 0.01
Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])


for n in range(0, N):
    U[n + 1, :] = Euler_Function(Dt, F, U[n, :])

plt.plot(U[:, 0], U[:, 1])
plt.show()

