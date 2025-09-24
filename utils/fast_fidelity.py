# utils/fast_fidelity.py

# Computes fidelity of DGA state relative to free fermion Slate using a simple determinant
# Significantly faster than using the circuits 

import numpy as np

from circuit.DGA_ansatz_circuit import _evenly_spaced_sites


def _build_B_from_brick(L, N_f, thetas, n_layers, S: np.ndarray | None = None):
    """
    From the DGA brick (even then odd per layer), build B = U[:, S]
    for occupied states S. S = {0,1,...,N_f-1} if none is given.
    """

    if S is None:
        S = _evenly_spaced_sites(L, N_f)
    else:
        S = np.asarray(S, dtype=int)

    B = np.zeros((L, N_f), dtype=float)
    B[S, np.arange(N_f)] = 1.0

    thetas = np.asarray(thetas, dtype=float)
    cs = np.cos(thetas)
    ss = np.sin(thetas)

    for layer in range(n_layers):
        base = layer * (L - 1)

        # even bonds
        for q in range(0, L - 1, 2):
            c, s = cs[base + q], ss[base + q]

            # Givens rotation
            G = np.array([[c, -s],
                          [s,  c]], dtype=float)
            sub = B[q:q+2, :]          # view (2 × N_f)
            sub[:] = G @ sub           # act with G

        # odd bonds
        for q in range(1, L - 1, 2):
            c, s = cs[base + q], ss[base + q]
            G = np.array([[c, -s],
                          [s,  c]], dtype=float)
            sub = B[q:q+2, :]
            sub[:] = G @ sub 
    return B

def dga_overlap_and_fidelity(Q, thetas, n_layers):
        """
        Fast overlap & fidelity between Slater state (from wave function matrix Q) and the DGA state (with n_layers and thetas parameters)

        creation operators for each state:
        a_iˆdagger = Sum_x A_{i,x} c_xˆdagger            (Slater)
        b_iˆdagger = Sum_x B_{i,x} c_xˆdagger            (DGA)

        A = QˆT                                          (Slater)
        B = U[:, S]                                      (DGA)
                    

        ⟨Slater|DGA⟩ = det(AˆT * B)ˆ2                  (squared since it's same for both spin sectors)
        fidelity = |det(AˆT * B)|ˆ4

        Assuming everything is real for performance - as more general case is not needed
        """
        Q = np.asarray(Q, dtype=float)           # (N_f x L)
        N_f, L = Q.shape
        thetas = np.asarray(thetas, dtype=float)

        # build B from bricks without forming U
        B = _build_B_from_brick(L, N_f, thetas, n_layers)   # (L x N_f)

        # per-spin overlap matrix S = Q* @ B = Q @ B  (real)
        S = Q @ B                                   # (N_f × N_f)
        sign, logabsdet = np.linalg.slogdet(S)      # for stability (avoid overflow if det S is very small/large), better than det
        
        overlap_per_spin = sign * np.exp(logabsdet) # det(AˆT @ B)
        overlap_total = overlap_per_spin**2         # spin-up/down identical
        fidelity_total = float(np.exp(4.0 * logabsdet.real))


        return overlap_total, fidelity_total
