from numpy import array, linspace
from numpy.linalg import norm
import matplotlib.pyplot as plt
from Milestone_2.Functions.Integrators import Cauchy
from Milestone_2.Functions.Integrators import oscilador
from Milestone_3.Functions.Errors import error_Cauchy

Dt = 0.005
t_final = 50.0
n_steps = int(t_final / Dt)
t1 = linspace(0.0, t_final, n_steps + 1)
t2 = linspace(0.0, t_final, 2 * n_steps + 1)
scheme = 'Euler'
U0 = array([1.0, 0.0])

U = Cauchy(F=oscilador, U0=U0, scheme=scheme, t=t1)
error = error_Cauchy(scheme=scheme, U0=U0, t1=t1, t2=t2, order=1)
print(f'Estimated final-state error: {norm(error):.6g}')

plt.figure()
plt.plot(t1, error[:, 0], label='Estimated Error')
plt.plot(t1, error[:, 1], label='Reference Error')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.show()
