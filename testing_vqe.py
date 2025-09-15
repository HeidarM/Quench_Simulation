# test_dga_vqe.py
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SPSA
from qiskit_aer import AerSimulator
from qiskit import transpile


from circuit.DGA_ansatz import DGA_ansatz

from runners.run_quench import run_QuenchSpectroscopy

from backend.backend import BackendConfig
# from runners.run_quench import generate_Q_mat


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
    # # Small test instance
    # L = 8
    # N_f = 6           # total fermions
    # n_layers = 1

    # # Build symbolic ansatz
    # ansatz, theta = DGA_ansatz(L=L, N_f=N_f, n_layers=n_layers)

    # backend = AerSimulator()
    # ansatz = transpile(ansatz, backend=backend, optimization_level=2)

    # # Free-fermion Hamiltonian (U=0)
    # H0 = free_fermion_H0(L, J=1.0)

    # # Estimator + VQE
    # estimator = EstimatorV2(options={"default_precision": 1e-2})  # shots-like behavior
    # opt = SPSA(maxiter=100)
    # vqe = VQE(estimator=estimator, ansatz=ansatz, optimizer=opt, initial_point=np.zeros(len(theta)))

    # res = vqe.compute_minimum_eigenvalue(H0)

    # print("Optimized energy  <H0>:", float(np.real(res.eigenvalue)))
    # print("Number of params:", len(theta))
    # print("Optimal parameters:", res.optimal_point)

    # # If you want to see the circuit with numbers bound (purely for plotting):
    # bound = ansatz.assign_parameters({theta: res.optimal_point})
    # try:
    #     from matplotlib import pyplot as plt
    #     bound.draw(output="mpl")
    #     plt.show()
    # except Exception:
    #     print(bound.draw(output="text"))

    
    # from main import *
    
    # backend_config = BackendConfig(kind="aer",  transpile_ol=0, default_precision=1e-2, aer_method="matrix_product_state")
    # backend_config = BackendConfig(
    # kind="aer",
    # aer_method="matrix_product_state",
    # aer_options={
    #     "max_parallel_threads": 0,
    #     "max_parallel_experiments": 0,
    #     "runtime_parameter_bind_enable": True,
    #     "mps_lapack": True,
    #     "precision": "single",
    #     # For Sampler runs with measure-all-at-end:
    #     # "mps_sample_measure_algorithm": "mps_probabilities",
    #     # Optional:
    #     # "max_parallel_shots": 0,
    #     # "mps_swap_direction": "mps_swap_right",
    #     # "max_memory_mb": -1,
    #     # "seed_simulator": 1234,
    # },)

    # run_VQE_for_DGA(L=30, N_f=10, n_layers=2, backend_config=backend_config, max_iterations=100, verbose=True)


    # Q_mat = generate_Q_mat(L=6, N_f=4)
    backend_config = BackendConfig(kind="aer", transpile_ol=0, shots=None)  # None → quasi-probs
    job, results = run_QuenchSpectroscopy(L=10, N_f=4, n_layers=2, dt=0.1, J=1.0, U=0.2, N_Trotter=10, backend_config=backend_config)

    # print("Job:", job)
    # print("Job id:", job.job_id())
    # for result in results:
    #     print("Counts:", result.join_data().get_counts())
    #     print()



