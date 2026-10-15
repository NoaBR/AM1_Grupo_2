
from numpy import array
from numpy.linalg import norm
from Milestone_2.Functions.Integrators import Cauchy, oscilador
from Milestone_3.Functions.Richardson import error_richardson

def error_Cauchy(scheme: str, U0:array, t1:array, t2:array, order:int) -> array:
    """
    Computes the error of a given time integration scheme using the Cauchy convergence test.


    Returns:
    - error_estimate: array containing the estimated error for each state component
    """
    
    U_coarse = Cauchy(F=oscilador, U0=U0, scheme=scheme, t=t1, plot=False)
    U_fine = Cauchy(F=oscilador, U0=U0, scheme=scheme, t=t2, plot=False)
    
    # Estimate error using the difference between coarse and fine solutions
    error_cauchy = error_richardson(order, U_coarse, U_fine[::2])
    
    return error_cauchy