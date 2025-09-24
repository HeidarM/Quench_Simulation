import numpy as np
import matplotlib.pyplot as plt
from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit

from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA




n_layers = 2
L = 3
N_f = 2
theta_values = np.arange(n_layers * (L - 1))*np.pi/10
# theta_values = np.random.rand(n_layers * (L - 1))
print(theta_values)

ansatz, theta = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers)
ansatz_b = ansatz.assign_parameters({theta: list(theta_values)})
ansatz_b.draw(output='mpl')
plt.show()
