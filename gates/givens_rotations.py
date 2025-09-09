# gates/givens_rotations.py
import numpy as np
from typing import List, Tuple

def _givens_matrix(a: complex, b: complex, direction="row") -> np.ndarray:
    """
    Returns a complex Givens rotation matrix G(θ, φ) such that:

        G @ [a, b] = [c, 0]  (direction = row)  or     [a, b] @ G^† = [c, 0] (direction = col)

    where:
        c = r * exp(i * arg(a))
        
        and

        G = [[ cos(θ), -exp(iφ)·sin(θ)],
             [ sin(θ),  exp(iφ)·cos(θ)]].

    The components are given by
        cos(θ) = |a| / r
        sin(θ) = -|b| / r
        r = sqrt(|a|² + |b|²)

        exp(iφ) = (a / |a|) * (conj(b) / |b|)
    """
    if np.isclose(a, 0):
        # exp_phi chosen so it's consistent with the other cases c = r * exp(i * arg(a)) = r
        cos, sin, exp_phi = 0.0, 1.0, -b.conjugate() / abs(b)
    elif np.isclose(b, 0):
        cos, sin, exp_phi = 1.0, 0.0, 1.0
    else:
        abs_a = abs(a)
        abs_b = abs(b)
        r = np.sqrt(abs_a * abs_a + abs_b * abs_b)

        cos = abs_a / r
        sin = -abs_b / r

        if direction == "row":
            exp_phi = (a * b.conjugate()) / (abs_a * abs_b)

        if direction == "col":
            exp_phi = (b * a.conjugate()) / (abs_a * abs_b)

    return np.array([[cos, -exp_phi * sin],
                     [sin,  exp_phi * cos]])


def _apply_rows(mat: np.ndarray, i: int, j: int, G: np.ndarray) -> None:
    mat[[i, j], :] = G @ mat[[i, j], :]


def _apply_cols(mat: np.ndarray, i: int, j: int, G: np.ndarray) -> None:
    mat[:, [i, j]] = mat[:, [i, j]] @ G.conj().T



def slaters_determinant_givens_rotation_list(Q: np.ndarray) -> List[Tuple[int, int, float, float]]:
    """
    Return a list of Givens-rotation instructions
        (j, k, θ, φ)      ←→      G_jk(θ, φ)
    which, when applied to qubits **j** and **k** (Jordan-Wigner order),
    prepare the Slater determinant represented by the N_fxN matrix ``Q``.

    Following arXiv:1711.05395:

    •  N       total number of spin-orbitals (``n``)
       N_f     number of occupied orbitals  (``m``)
       Q       coefficient matrix, Eq. (7)

    •  Step 1: Simplify Q
       Freedom  **Q → V Q**  (Eq. (10)):
       Any N_fxN_f unitary **V** acting on the *rows* leaves the Slater
       determinant unchanged except for the global phase det V.  We exploit
       that freedom (row-Givens sweep) to zero an upper-right block of Q to find a simplified but equivalent Q.

    •  Step 2: Find Givens rotations for quantum circuit
       Starting from simplified Q, we use *column* rotations **Q → Q U** to effectively "diagonalize" it.
       By reversing this process, we can recreate Q and these given rotations thus correspond to physical gates.
       The list of Givens rotations are returned.

    The routine therefore has two sweeps:

      1. **Row sweep  (simplification, no gate)**
         Make Q -> VQ have zeros in the upper-right “corner”
         - Leaves projector P_S = Q Q† invariant → same many-body state,  
           up to an overall phase det V.  

      2. **Column sweep  (physical gates)**  
         Diagonalise VQ column-wise:  VQ U_R = diag.  
         Every column Givens rotation adds one tuple (j,k,θ,φ) to *rotations*.

      3. The list is reversed so that the first element corresponds to the
         first gate executed on hardware (matching Eq. (15)).

    The total count |rotations| equals (N - N_f)·N_f  as in Eq. (18).
    """

    m, n = Q.shape                    # m = N_f (occupied), n = N (orbitals)
    A = Q.astype(complex, copy=True)  # working copy of Q
    rotations: List[Tuple[int, int, float, float]] = []  # (j, k, θ, φ)

    # Fully filled configuration
    if m == n:
        return rotations

    # ───────────────────────────────────────────────────────────────
    #  ROW-GIVENS SWEEP  (implements freedom  V  in  Q → VQ)
    #  Acts on rows ⇒ preserves Slater determinant up to phase det V.
    #  Gates are NOT stored because they never have to run on qubits.
    # ───────────────────────────────────────────────────────────────
    for j in reversed(range(n - m + 1, n)):           # columns to clear
        for i in range(m - n + j):                    # rows inside corner
            if not np.isclose(A[i, j], 0.0):
                G = _givens_matrix(A[i + 1, j], A[i, j])
                _apply_rows(A, i + 1, i, G)

    # ───────────────────────────────────────────────────────────────
    #  COLUMN-GIVENS SWEEP  (produces physical two-qubit gates)
    #  Each rotation mixes orbitals j,k ⇒ must be enacted on qubits.
    #  We record every (j,k,θ,φ) needed to triangularise VQ.
    # ───────────────────────────────────────────────────────────────
    for i in range(m):
        for j in range(n - m + i, i, -1):             # walk left across row i
            if not np.isclose(A[i, j], 0.0):
                G = _givens_matrix(A[i, j - 1], A[i, j], direction="col")
                θ = float(np.arcsin(np.real(G[1, 0])))
                φ = float(np.angle(G[1, 1]))
                rotations.append((j, j - 1, θ, φ))    # physical gate
                _apply_cols(A, j - 1, j, G)           # A ← A @ G†

    # Reverse so earliest gate appears first.
    rotations.reverse()

    return rotations