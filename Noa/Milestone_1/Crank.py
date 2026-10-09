from numpy import array, zeros
import matplotlib.pyplot as plt
from numpy.linalg import norm   
N = 1000
Dt = 0.1

Nv = 2

U = zeros((N+1, Nv))
U[0, :] = array([1, 0])

def F(U: array) -> array:
    return array([U[1], -U[0]])

for n in range(0,N):
    Y = U[n, :]
    while norm(Y - U[n,:] - Dt/2*(F(Y) + F(U[n,:]))) > 1e-10:
        R = Y - U[n,:] - Dt/2*(F(Y) + F(U[n,:]))
        Y = Y - R
        print(norm(R))
    U[n+1, :] = Y



plt.plot(U[:, 0], U[:, 1])
plt.show()