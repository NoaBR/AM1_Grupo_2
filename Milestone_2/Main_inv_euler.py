from numpy import array, zeros
import matplotlib.pyplot as plt
from Functions.Inv_Euler_f import Inverse_Euler_Function

N = 10000
Dt = 0.001
Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])

for n in range(0, N):
    U[n + 1, :] = Inverse_Euler_Function(Dt, F, U[n, :])

plt.plot(U[:, 0], U[:, 1])
plt.show()
