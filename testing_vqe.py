# test_dga_vqe.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit import transpile


from gates.DGA_ansatz import DGA_ansatz
from circuit.VQE_DGA import VQE_DGA


# def _pauli_label(n, ops):
#     """
#     Build a Pauli label string of length n with little-endian indexing:
#     qubit 0 is rightmost char. 'ops' is a dict {q: 'X'/'Y'/'Z'}.

#     example: _pauli_label(10, {0: 'X', 5: 'Y', 7: 'Z'}) -> 'IIZIYIIIIX'
#     """
#     s = ['I'] * n
#     for q, p in ops.items():
#         s[n - 1 - q] = p
#     return ''.join(s)

# def free_fermion_H0(L: int, J: float = 1.0) -> SparsePauliOp:
#     """
#     H0 = -J * sum_{i=0}^{L-2} sum_{spin in {up,down}} ( c†_{i,spin} c_{i+1,spin} + h.c.)

#     JW (nearest neighbors) -> -J/2 * (X_i X_{i+1} + Y_i Y_{i+1}), separately in each spin sector.

#     Block ordering: [0..L-1]=↑, [L..2L-1]=↓.
#     """
#     n = 2 * L
#     terms = []
#     coeffs = []
#     for base in (0, L):  # up block, down block
#         for i in range(base, base + L - 1):
#             # -J/2 * (X_i X_{i+1} + Y_i Y_{i+1})
#             terms.append(_pauli_label(n, {i: 'X', i + 1: 'X'})); coeffs.append(-J / 2)
#             terms.append(_pauli_label(n, {i: 'Y', i + 1: 'Y'})); coeffs.append(-J / 2)
#     return SparsePauliOp.from_list(list(zip(terms, coeffs)))

# # ---------------- run a quick VQE on H0 with DGA ----------------
if __name__ == "__main__":
    # Small test instance
    L = 8
    N_f = 6           # total fermions
    n_layers = 1

#     # Build symbolic ansatz
#     ansatz, theta = DGA_ansatz(L=L, N_f=N_f, n_layers=n_layers)

#     backend = AerSimulator()
#     ansatz = transpile(ansatz, backend=backend, optimization_level=2)

#     # Free-fermion Hamiltonian (U=0)
#     H0 = free_fermion_H0(L, J=1.0)

#     # Estimator + VQE
#     estimator = EstimatorV2(options={"default_precision": 1e-2})  # shots-like behavior
#     opt = SPSA(maxiter=100)
#     vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt, initial_point=np.zeros(len(theta)))

#     res = vqe.compute_minimum_eigenvalue(H0)

#     print("Optimized energy  <H0>:", float(np.real(res.eigenvalue)))
#     print("Number of params:", len(theta))
#     print("Optimal parameters:", res.optimal_point)

#     # If you want to see the circuit with numbers bound (purely for plotting):
#     bound = ansatz.assign_parameters({theta: res.optimal_point})
#     try:
#         from matplotlib import pyplot as plt
#         bound.draw(output="mpl")
#         plt.show()
#     except Exception:
#         print(bound.draw(output="text"))

    VQE_DGA(L=L, N_f=N_f, n_layers=n_layers)
