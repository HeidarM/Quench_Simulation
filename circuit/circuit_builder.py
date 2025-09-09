# circuit/circuit_builder.py

import numpy as np
from qiskit import QuantumCircuit

from gates.givens_rotations import slaters_determinant_givens_rotation_list
from gates.custom_gates import G, Q, BXY, FSWAP
from .trotter import trotter_step


# For re-ordering qubits with based on spins: (↑…↑)(↓…↓) into (↑,↓)…(↑,↓)
def _interleave_pairs(L: int):
    order  = list(range(2*L))           # current wire→qubit mapping
    pairs  = []                         # list of (wireA, wireB) swaps
    for s in range(L):                  # bubble ↓_s leftwards
        lbl_down   = L + s              # qubit label "s↓"
        idx_down   = order.index(lbl_down)
        target_idx = 2*s + 1
        while idx_down > target_idx:    # move one position to the left
            pairs.append((idx_down-1, idx_down))
            order[idx_down-1], order[idx_down] = \
                order[idx_down], order[idx_down-1]
            idx_down -= 1
    return pairs                         # length = L(L-1)/2

def reorder_qubits(qc: QuantumCircuit, L: int):
    """
    Convert (↑…↑)(↓…↓) into (↑,↓)…(↑,↓) using the minimal number of FSWAPs.
    """
    for a, b in _interleave_pairs(L):
        qc.append(FSWAP(), [a, b])



def QuenchSpectroscopyCircuit(Q_mat: np.ndarray, dt: float, J: float,
                              U: float, N_Trotter: int) -> QuantumCircuit:
    """
    Build the full Fig-4 circuit for an arbitrary number of sites L≥2.

    Evolves the system for t = N_Trotter * dt.

    """
    N_f, L = Q_mat.shape

    if L < 2: raise ValueError("Need at least 2 sites.")
    if N_f > 2*L: raise ValueError("More fermions than states")

    qc = QuantumCircuit(2*L)


    # ---- Create Slater's determinant state ----
    # Given rotations needed
    rot_list = slaters_determinant_givens_rotation_list(Q_mat)

    # print([φ for (j, k, θ, φ) in rot_list])

    # Put N_f/2 fermions in ↑ block and N_f/2 in ↓ block
    for q in range(N_f):
        qc.x(q)           # ↑ block
        qc.x(q+L)         # ↓ block

    # Create Slaters determinant circuit
    for (j, k, θ, φ) in rot_list:   
        qc.append(G(θ,φ), [j, k])         # ↑ block
        qc.append(G(θ,φ), [j+L, k+L])     # ↓ block

    # qc.barrier()

    # Re-order from (↑...↑ ↓...↓) to (↑,↓)(↑,↓)...(↑,↓) 
    reorder_qubits(qc, L)

    # qc.barrier()

    # Local quench  Q(π/4) on central site
    centre = L//2
    qc.append(Q(np.pi/4), [2*centre, 2*centre+1])

    # qc.barrier()

    # Time evolution  (N_Trotter primitive slices) 
    for _ in range(N_Trotter):
        trotter_step(qc, L, dt, J, U)
        # qc.barrier()
    

    for site in range(L):
        up, down = 2*site, 2*site + 1
        qc.append(BXY(), [up, down])


    return qc


def QuenchSpectroscopyCircuits(Q_mat: np.ndarray, dt: float, J: float,
                              U: float, Max_N_Trotter: int, verbose=False) -> list[QuantumCircuit]:
    """
    Build an array of circuits for up to Max_N_Trotter time steps.

    """
    # Circuits of each time step
    circuits = []

    if verbose:
        print("Creating circuits...\n")
    for time_step in range(Max_N_Trotter):
        if verbose:
            print("Time step: ", time_step)
        qc = QuenchSpectroscopyCircuit(Q_mat, dt, J, U, time_step)
        qc = qc.decompose().decompose() # To make it work with aer
        qc.measure_all()
        circuits.append(qc)
    
    return circuits