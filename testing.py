import numpy as np
import matplotlib.pyplot as plt
from circuit.DGA_ansatz import DGA_ansatz

from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA

from qiskit.circuit import Parameter, ParameterExpression
from qiskit.circuit.tools.pi_check import pi_check
from numbers import Real
from gates.custom_gates import _is_param, _fmt_angle_for_label

n_layers = 2
L = 3
N_f = 2
theta_values = np.arange(n_layers * (L - 1))*np.pi/10
# theta_values = np.random.rand(n_layers * (L - 1))
print(theta_values)

ansatz, theta = DGA_ansatz(L=L, N_f=N_f, n_layers=n_layers)
ansatz.assign_parameters({theta: list(theta_values)})
ansatz.draw(output='mpl')
plt.show()



x = _is_param(theta)
y = _is_param(theta_values)
print("x is param:", x)
print("y is param:", y)

label1 = _fmt_angle_for_label(theta[2])
label2 = _fmt_angle_for_label(theta_values[2])
print(label1)
print(label2)   