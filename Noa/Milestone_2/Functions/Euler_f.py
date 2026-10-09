from numpy import array, zeros
import matplotlib.pyplot as plt


def Euler_Function( Dt: float, F: callable, U0: array) -> array:


    return U0 + Dt * F(U0)
