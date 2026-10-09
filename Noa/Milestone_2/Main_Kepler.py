from numpy import array, zeros
from Functions.Kepler import Kepler

t = zeros(1000)
for i in range(0, 1000):
    t[i] = i * 0.01

U0 = array([1,0, 0, 1])


Scheme = 'Crank_Nicolson' #'Euler','RK4','Crank','Crank_Nicolson', 'inv_Euler' 
Kepler(U0, Scheme, t)