from numpy import array, zeros
import matplotlib.pyplot as plt
from Functions.Crank_f import Crank_function

N = 1000
Dt = 0.1

Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])

for n in range(0,N):
 U[n + 1, :] = Crank_function(F, U[n, :], Dt)



plt.plot(U[:, 0], U[:, 1])
plt.show()