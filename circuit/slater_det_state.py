# circuits/slater_det_state.py

import numpy as np
from qiskit import QuantumCircuit
from gates.givens_rotations import slaters_determinant_givens_rotation_list
from gates.custom_gates import G


# For constructing Slater's determinant state (for one spin sector only)
def generate_Q_mat(L, N_f):
    # Wave function for OBC
    j  = np.arange(L)                      # site indices 0 … L−1
    n  = np.arange(1, N_f + 1)[:, None]    # mode numbers 1 … N_f (column)
    k  = n * np.pi / (L + 1)               # quantised momenta (N_f×1)
    Q_mat  = np.sqrt(2.0 / (L + 1)) * np.sin(k * (j + 1))
    return Q_mat



def slaters_determinant_state(Q_mat: np.ndarray) -> QuantumCircuit:
    """
    Create a circuit that prepares a simple Slater determinant state from a Q matrix of shape (N_f, L)
    with L sites (2*Lqubits) and N_f fermions per spin sector.

    State is in block spin ordering: ↑....↑ ↓...↓ .
    """

    N_f, L = Q_mat.shape

    if L < 2: raise ValueError("Need at least 2 sites.")
    if N_f > L: raise ValueError("More fermions than states")

    qc = QuantumCircuit(2*L)
    qc.name = "SlaterDeterminantState"

    # Given rotations needed
    rot_list = slaters_determinant_givens_rotation_list(Q_mat)

    # print([φ for (j, k, θ, φ) in rot_list])

    # Put N_f fermions in ↑ block and N_f in ↓ block: 2*N_f fermions in total
    for q in range(N_f):
        qc.x(q)           # ↑ block
        qc.x(q+L)         # ↓ block

    # Create Slaters determinant circuit
    for (j, k, θ, φ) in rot_list:   
        qc.append(G(θ,φ), [j, k])         # ↑ block
        qc.append(G(θ,φ), [j+L, k+L])     # ↓ block

    return qc