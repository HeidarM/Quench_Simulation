# circuit/slater_det_circuit.py

import numpy as np
from itertools import combinations

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector


from gates.givens_rotations import slaters_determinant_givens_rotation_list
from gates.custom_gates import G


# For constructing Slater's determinant state (for one spin sector only)
def generate_Q_mat(L, N_f):
    """
    First N_f single-particle orbitals (rows) for a 1D tight-binding chain of length L
    with open boundary conditions (OBC).

    Orbitals (standing waves):
        ψ_n(j) = Sqrt[2/(L+1)] * sin(k_n * (j+1)),  k_n = nπ/(L+1),  n=1..N_f,  j=0..L-1
    Hamiltonian (spinless, OBC, hopping t):
        H = -t * Sum_{j=0}^{L-2} ( c†_j c_{j+1} + c†_{j+1} c_j )
    Energies of these modes:
        ε_n = -2 t cos(k_n)

    Rows of Q_mat are orthonormal and basis used to build a Slater determinant.
    """
    j  = np.arange(L)                      # site indices 0 … L−1
    n  = np.arange(1, N_f + 1)[:, None]    # mode numbers 1 … N_f (column)
    k  = n * np.pi / (L + 1)               # quantised momenta (N_f×1)
    Q_mat  = np.sqrt(2.0 / (L + 1)) * np.sin(k * (j + 1))
    return Q_mat



def slaters_determinant_circuit(Q_mat: np.ndarray) -> QuantumCircuit:
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
    # Givens rotation construction assumes this, and will delocalize the fermions afterwards
    for q in range(N_f):
        qc.x(q)           # ↑ block
        qc.x(q+L)         # ↓ block

    # Create Slaters determinant circuit
    for (j, k, θ, φ) in rot_list:   
        qc.append(G(θ,φ), [j, k])         # ↑ block
        qc.append(G(θ,φ), [j+L, k+L])     # ↓ block

    return qc




# Create the Statevector for free fermions in a Slater determinant state for testing purposes
def slater_statevector(L: int, N_f: int) -> Statevector:
    """
    Build the 2L-qubit Statevector for a Slater determinant with block-spin ordering
    [↑0..↑L-1 | ↓0..↓L-1], using OBC orbitals from generate_Q_mat(L, N_f).

    Logic:
      • Per spin sector, a Slater determinant |ψ⟩ is fully specified by the occupied
        one-particle orbitals Q (shape: N_fxL). For any configuration that occupies
        site-set I (|I| = N_f), its amplitude equals det(Q[:, I]).
      • Our full state has identical, independent ↑ and ↓ sectors prepared with the
        same Q, so the total amplitude factorizes: amp(I_up, I_dn) = det(Q[:, I_up]) * det(Q[:, I_dn]).
      • We map (I_up, I_dn) into a computational-basis index using block ordering:
        ↑ on qubits 0..L-1 and ↓ on qubits L..2L-1 (Qiskit little-endian).
    """
    if L < 2:
        raise ValueError("Need at least 2 sites.")
    if not (0 <= N_f <= L):
        raise ValueError("N_f must satisfy 0 <= N_f <= L")

    # One-particle orbitals per spin sector (rows = occupied orbitals)
    Q = generate_Q_mat(L, N_f).astype(complex)   # shape (N_f, L)

    # Enumerate all choices of N_f occupied sites among L for ONE spin sector.
    # For each such set I, the (per-spin) amplitude is det(Q[:, I]).
    occ_sets = list(combinations(range(L), N_f))

    # Precompute per-spin amplitudes: A(I) = det(Q[:, I])
    amp = {I: np.linalg.det(Q[:, I]) for I in occ_sets}

    # Map occupied-site sets to a basis index (Qiskit little-endian):
    # ↑ occupies qubits [0..L-1], ↓ occupies qubits [L..2L-1].
    def basis_index(I_up, I_dn):
        idx = 0
        for i in I_up:
            idx += 2**i
        for j in I_dn:
            idx += 2**(L + j)
        return idx

    dim = 2**(2 * L)
    psi = np.zeros(dim, dtype=complex)

    # Only configurations with exactly N_f↑ and N_f↓ are nonzero; amplitudes factorize.
    for Iu in occ_sets:
        Au = amp[Iu]
        for Id in occ_sets:
            psi[basis_index(Iu, Id)] = Au * amp[Id]

    # Normalize
    nrm = np.linalg.norm(psi)
    if nrm == 0:
        raise ValueError("Constructed zero vector; check Q orthonormality.")
    psi /= nrm

    return Statevector(psi)
