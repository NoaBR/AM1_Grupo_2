from numpy import array
from numpy.linalg import norm   

def Crank_function(F: callable, U0: array, Dt: float) -> array:
    Y = U0
    while norm(Y - U0 - Dt/2*(F(Y) + F(U0))) > 1e-10:
        R = Y - U0 - Dt/2*(F(Y) + F(U0))
        Y = Y - R
        print(norm(R))
    return  Y