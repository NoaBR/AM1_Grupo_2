from Functions.Cauchy import Cauchy
from numpy import array
from numpy.linalg import norm


def Kepler(U0:array, Scheme:str, t:array) -> None:

    #def F(U):
       #return array([U[1], -U[0]/norm(U)**3])

    def Kepler(U):

        r = U[:2]
        v = U[2:]   

        return array([*v, *(-r / norm(r)**3)])

    Cauchy(F=Kepler, U0=U0, scheme=Scheme, t=t)