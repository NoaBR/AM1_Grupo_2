from numpy import array, zeros
from numpy.linalg import norm, solve
from math import sqrt

def Euler_PrincipalAxis(I: array, omega: array) -> array:
    w_dot = zeros(3)
    w_dot[0] = (I[1] - I[2]) / I[0] * omega[1] * omega[2]
    w_dot[1] = (I[2] - I[0]) / I[1] * omega[2] * omega[0]
    w_dot[2] = (I[0] - I[1]) / I[2] * omega[0] * omega[1]
    return w_dot

def InvariantProperties(I: array, omega: array) -> tuple:
    T = 0.5 * (I[0] * omega[0]**2 + I[1] * omega[1]**2 + I[2] * omega[2]**2)
    K_e_rot = 2 * T
    L = sqrt((I[0] * omega[0])**2 + (I[1] * omega[1])**2 + (I[2] * omega[2])**2)
    AngMom_Module = L**2
    return T, L, K_e_rot, AngMom_Module

def Jacobian_field(I: array, omega: array) -> array:
    J = zeros((3, 3))
    J[0, 0] = 0
    J[0, 1] = (I[1] - I[2]) / I[0] * omega[2]
    J[0, 2] = (I[1] - I[2]) / I[0] * omega[1]
    J[1, 0] = (I[2] - I[0]) / I[1] * omega[2]
    J[1, 1] = 0
    J[1, 2] = (I[2] - I[0]) / I[1] * omega[0]
    J[2, 0] = (I[0] - I[1]) / I[2] * omega[1]
    J[2, 1] = (I[0] - I[1]) / I[2] * omega[0]
    J[2, 2] = 0
    return J

def eigenvalues(I: array, omega: array) -> array:
    if omega[0] != 0:
         Omega = omega[0]
         # only imaginary eigenvalues
         lambda1 = Omega*sqrt((I[2] - I[1])*(I[2] - I[0])/(I[2]*I[1]))*1j
         lambda2 = -Omega*sqrt((I[2] - I[1])*(I[2] - I[0])/(I[2]*I[1]))*1j
    elif omega[1] != 0:
         Omega = omega[1]
         # only real eigenvalues
         lambda1 = Omega*sqrt((I[2] - I[1])*(I[2] - I[1])/(I[0]*I[2]))
         lambda2 = -Omega*sqrt((I[2] - I[1])*(I[2] - I[1])/(I[0]*I[2]))
    elif omega[2] != 0:
         Omega = omega[2]
            # only imaginary eigenvalues
         lambda1 = Omega*sqrt((I[2] - I[1])*(I[2] - I[0])/(I[0]*I[1]))*1j
         lambda2 = -Omega*sqrt((I[2] - I[1])*(I[2] - I[0])/(I[0]*I[1]))*1j
    return lambda1, lambda2