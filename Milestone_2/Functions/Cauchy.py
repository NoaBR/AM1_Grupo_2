from numpy import array, zeros
import matplotlib.pyplot as plt
from .esquemas_temporales import Crank_Nicolson_function, Euler_Function,  Inverse_Euler_Function, RK4_function, Crank_function


def Cauchy(F:callable, U0:  array, scheme:str, t:array) -> None:

     Nv = len(U0)
     N = len(t) - 1


     U = zeros((N+1, Nv))
     U[0, :] = U0



     for n in range(0, N):
        Dt = t[n + 1] - t[n]
        if scheme == 'Euler':
            U[n + 1, :] = Euler_Function(F, U[n, :], Dt)
        elif scheme == 'inv_Euler':
            U[n + 1, :] = Inverse_Euler_Function(F, U[n, :], Dt)
        elif scheme == 'RK4':
            U[n + 1, :] = RK4_function(F, U[n, :], Dt)
        elif scheme == 'Crank':
            U[n + 1, :] = Crank_function(F, U[n, :], Dt)
        elif scheme == 'Crank_Nicolson':
            U[n + 1, :] = Crank_Nicolson_function(F, U[n, :], Dt)
        else :
            print('Scheme not implemented')

     plt.plot(U[:, 0], U[:, 1])
     plt.show()