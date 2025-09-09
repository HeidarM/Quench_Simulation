# circuit/trotter.py
from qiskit import QuantumCircuit
from gates.custom_gates import H, O, FSWAP

# CHECK THIS
def trotter_step(qc, L: int, dt: float, J: float, U: float):
    """
    ONE primitive (first-order) Suzuki–Trotter slice:
        even hoppers → swap layer → odd hoppers → on-site interaction.

         e^{-i dt J ∑_{⟨ij⟩}(X_i X_j + Y_i Y_j)}

    to a circuit that is laid out as
        (↑,↓)(↑,↓)…(↑,↓).

    Pattern per bond s–(s+1):
        FSWAP        (↓_s  ↔  ↑_{s+1})
        H(dt·J) on   ↑_s  ↑_{s+1}
        H(dt·J) on   ↓_s  ↓_{s+1}
        FSWAP        (undo)
    """

    hop = H(dt * J)                           # reuse the gate instance

    # ---------- nearest-neighbour bonds, one by one -----------------
    for s in range(L - 1):
        i, j = 2*s + 1, 2*s + 2               # qubits ↓_s  and  ↑_{s+1}

        qc.append(FSWAP(), [i, j])            # bring odd bond together
        qc.append(hop, [2*s,     2*s + 1])    # ↑_s  –  ↑_{s+1}
        qc.append(hop, [2*s + 2, 2*s + 3])    # ↓_s  –  ↓_{s+1}
        qc.append(FSWAP(), [i, j])            # restore JW ordering

    # ---------- on-site Hubbard U  ----------------------------------
    for site in range(L):
        up, down = 2*site, 2*site + 1
        qc.append(O(dt * U), [up, down]) # one gate per site