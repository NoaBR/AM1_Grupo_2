from numpy import array, zeros
from Functions.Cauchy import Cauchy
from Functions.esquemas_temporales import oscilador

t = zeros(1000)
for i in range(0, 1000):
    t[i] = i * 0.01




#Scheme = 'RK4','Euler','RK4','Crank', 'Crank_Nicolson','inv_Euler' 
Cauchy(F=oscilador, U0=array([1, 0]), scheme='Crank_Nicolson', t=t)