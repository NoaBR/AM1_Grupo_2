from numpy import array, zeros
import matplotlib.pyplot as plt
from Milestone_2.Functions.esquemas_temporales import Crank_Nicolson_function, Euler_Function,  Inverse_Euler_Function, RK4_function, Crank_function


def Cauchy(F:callable, U0:  array, scheme:str, t:array, plot: bool = True) -> array:

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

     if plot:
         plt.figure()
         plt.plot(U[:, 0], U[:, 1])
         plt.xlabel('u1')
         plt.ylabel('u2')
         plt.title(f'Phase portrait ({scheme})')
         plt.axis('equal')
         plt.grid(True)
         plt.show()

     return U

     
def partition(a: float, b: float, N: int) -> list:
    return [a + i*(b-a)/N for i in range(N+1)]


def oscilador(U: array) -> array:
    return array([U[1], -U[0]])