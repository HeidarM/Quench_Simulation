# circuits/DGA_ansatz.py

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector

from gates.custom_gates import G

# Dense Givens Ansatz (DGA) parametrized by angles theta
def DGA_ansatz(L: int, N_f: int, n_layers: int):
    """
    Dense Givens Ansatz (DGA) for a spinful chain in block spin ordering:
      qubits [0..L-1] = spin ↑ sector, qubits [L..2L-1] = spin ↓ sector.

    N_f : number of fermions per spin sector (total 2*N_f fermions)

    theta : Givens rotation angles - same for both spin sectors
    """

    qc = QuantumCircuit(2 * L, name="DGA") # 2*L qubits for spinful fermions

    # Put N_f fermions in ↑ block and N_f in ↓ block: 2*N_f fermions in total
    for q in range(N_f):
        qc.x(q)           # ↑ block
        qc.x(q+L)         # ↓ block

    qc.barrier()


    # --- Parameters ---
    theta = ParameterVector("θ", n_layers * (L - 1))

    # Reshape to get parameters layer-wise
    def theta_layerwise(layer: int, bond: int):
        return theta[layer * (L - 1) + bond]

    
    φ = 0.0
    # --- Dense brick per layer, per spin sector (shared θ's) ---
    for layer in range(n_layers):
        # Even
        for q in range(0, L - 1, 2):
            θ = theta_layerwise(layer, q)
            qc.append(G(θ, φ), [q, q + 1])          # ↑
            qc.append(G(θ, φ), [L + q, L + q + 1])  # ↓
        # Odd
        for q in range(1, L - 1, 2):
            θ = theta_layerwise(layer, q)
            qc.append(G(θ, φ), [q, q + 1])          # ↑
            qc.append(G(θ, φ), [L + q, L + q + 1])  # ↓

    return qc, theta
