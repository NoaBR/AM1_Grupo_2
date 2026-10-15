
from numpy import array
from numpy.linalg import norm
from Milestone_2.Functions.Integrators import Cauchy
from Milestone_3.Functions.Richardson import error_richardson

def error_Cauchy(scheme: str, U0:array, Nt:int, Dt:float, F:callable, order:int) -> float:
    """
    Computes the error of a given time integration scheme using the Cauchy convergence test.

    Parameters:
    - scheme: str, the name of the time integration scheme ('Euler', 'RK4', etc.)
    - U0: array, initial condition
    - Nt: int, number of time steps
    - Dt: float, time step size
    - F: callable, function defining the system of ODEs

    Returns:
    - error_estimate: float, estimated error based on the Cauchy convergence test
    """
    
    # Compute solution with coarse time step
    U_coarse = Cauchy(scheme, U0, Nt//2, 2*Dt, F)
    
    # Compute solution with fine time step
    U_fine = Cauchy(scheme, U0, Nt, Dt, F)
    
    # Estimate error using the difference between coarse and fine solutions
    error_cauchy = error_richardson(order, U_coarse[-1], U_fine[-1])
    
    return error_cauchy