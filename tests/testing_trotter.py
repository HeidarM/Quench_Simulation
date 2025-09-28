# tests/testing_trotter.py
# run using python -m tests.testing_trotter

# Testing the trotter step circuit

import numpy as np
from math import prod
from scipy.linalg import expm
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from circuit.trotter import trotter_step


# ---------- Jordan–Wigner operators ----------

# Given a list of matrices, [A0, A1, ..., A_{n-1}], this will return the tensor/kronecker product A0x...xA_{n-1}
def kron_all(operator_list):
    tensor_product = operator_list[0]
    for A in operator_list[1:]:
        tensor_product = np.kron(tensor_product, A)
    return tensor_product

# Pauli matrices
I = np.array([[1, 0], [0, 1]], dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
P = np.array([[0, 1], [0, 0]], dtype=complex)  # sigma^+ = |1><0|
M = np.array([[0, 0], [1, 0]], dtype=complex)  # sigma^- = |0><1|

# Jordan-Wigner representation of creation/annihilation operators
def jw_creation(j, n_qubits):
    # c^\dagger_j = (prod_{k<j} Z_j) * sigma^+_j
    ops = [I] * n_qubits    # Fill with identity operator

    # Then replace what's needed
    for k in range(j):
        ops[k] = Z
    ops[j] = P

    # Return tensor product
    return kron_all(ops)

def jw_annihilation(j, n_qubits):
    # c_j = (prod_{k<j} Z_k) * sigma^-_j
    ops = [I] * n_qubits
    for k in range(j):
        ops[k] = Z
    ops[j] = M
    return kron_all(ops)

def jw_number(j, n_qubits):
    # n = c^\dagger_j c_j = (I - Z_j)/2
    ops = [I] * n_qubits
    ops[j] = (I - Z) / 2
    return kron_all(ops)


# ---------- Hubbard Hamiltonian in (↑,↓)...(↑,↓) order ----------

def mode_index(site, spin):
    # spin: 0 for up, 1 for down; interleaved order (↑,↓)_1 (↑,↓)_2 ... (↑,↓)_{L-1}
    return 2 * site + spin

def hubbard_hamiltonian(L, J, U):
    #  -J sum_{i, s \in (↑,↓)} (c^dagger_{i,s} c_{i+1,s} + h.c.)
    #    + U sum_i n_{i,up} n_{i,down}

    n_qubits = 2 * L
    dim = 2 ** n_qubits
    H = np.zeros((dim, dim), dtype=complex)

    # Hopping terms
    for i in range(L - 1): # site index
        for s in (0, 1):   # spin: up, down
            a = mode_index(i, s)
            b = mode_index(i + 1, s)
            cd_a = jw_creation(a, n_qubits)
            c_a = jw_annihilation(a, n_qubits)
            cd_b = jw_creation(b, n_qubits)
            c_b = jw_annihilation(b, n_qubits)
            H += -J * (cd_a @ c_b + cd_b @ c_a) 

    # Hubbard interaction
    for i in range(L):
        up = mode_index(i, 0)
        dn = mode_index(i, 1)
        H += U * (jw_number(up, n_qubits) @ jw_number(dn, n_qubits))

    return H


# ---------- Circuit unitary for a single Trotter slice ----------

def trotter_slice_unitary(L, dt, J, U):
    qc = QuantumCircuit(2 * L)
    trotter_step(qc, L, dt, J, U)
    return Operator(qc).data


# ---------- Metrics ----------

def spectral_norm(M):
    # 2-norm (largest singular value)
    return np.linalg.norm(M, 2)

def unitary_distance(U, V):
    # ||U - V||_2 (spectral norm)
    return spectral_norm(U - V)

def process_fidelity(U, V):
    # F_pro = |Tr(U^\dagger V)|^2 / d^2  ; d = 2^n
    d = U.shape[0]
    return (np.abs(np.trace(U.conj().T @ V)) / d) ** 2


# ---------- Main check ----------

def check_trotter_steps(L=3, J=1.0, U=0.7, T=1.0, N_trotter=10, verbose=True):
    # Compare U_exact(T) = exp(-i H T) to the N_trotter-slice product
    # U_trot(T) = [trotter_step(dt)]^N_trotter with dt = T / N_trotter.

    n_qubits = 2 * L
    d = 2 ** n_qubits
    dt = T / N_trotter

    # Exact full-time unitary
    H = hubbard_hamiltonian(L, J, U)
    U_exact = expm(-1j * H * T)

    # Build N Trotter slices of size dt
    qc = QuantumCircuit(2 * L)
    for _ in range(N_trotter):
        trotter_step(qc, L, dt, J, U)
    U_trot = Operator(qc).data

    # Metrics
    dist = unitary_distance(U_trot, U_exact)
    Fp = process_fidelity(U_exact, U_trot)

    if verbose:
        print(f"[L={L}, J={J}, U={U}, T={T}, N={N_trotter}, dt={dt}]")
        print(f"  dim = {d}, ||U_trot - U_exact||_2 = {dist:.3e}")
        print(f"  process fidelity F_pro = {Fp:.8f}")

    return dist, Fp


def sweep_convergence(L=3, J=1.0, U=0.7, T=1.0, N_list=(1,2,4,8,16,32)):
    rows = []
    for N in N_list:
        dist, Fp = check_trotter_steps(L=L, J=J, U=U, T=T, N_trotter=N, verbose=False)
        rows.append((N, T/N, dist, Fp))
    print(f"\nConvergence at fixed T={T}:")
    print("   N   |     dt      |   ||ΔU||_2    |   F_pro")
    print("--------+-------------+---------------+-----------")
    for N, dt, dist, Fp in rows:
        print(f"{N:6d} | {dt:11.6g} | {dist:13.6e} | {Fp:9.7f}")
    return rows



if __name__ == "__main__":
    # A couple of small-L smoke tests
    params = [
        dict(L=3, J=1.0, U=3.0, T=0.05, N_trotter=1),
        dict(L=3, J=1.0, U=3.0, T=0.1, N_trotter=1),
        dict(L=3, J=1.0, U=3.0, T=0.5, N_trotter=1),
        dict(L=3, J=1.0, U=3.0, T=1.0, N_trotter=1)
    ]

    print("\n-------- Testing Single Step --------")
    for p in params:
        check_trotter_steps(**p)


    print("\n-------- Testing Total Time --------")
    check_trotter_steps(L=3, J=1.0, U=3.0, T=5.0, N_trotter=1000)

    sweep_convergence(L=3, J=0.8, U=1.2, T=2.0, N_list=[1,2,4,8,16,32,64])
