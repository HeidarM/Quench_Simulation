# fidelity_test_slater.py
# run using python -m tests.fidelity_test_slater

from qiskit.quantum_info import Statevector
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat, slater_statevector


if __name__ == "__main__":

    nf = 2/3


    fidelities = []

    for L in range(2,10):
        print(f"\n\n=== L = {L} ===")
        N_f = round(nf/2*L)

        Q_mat = generate_Q_mat(L, N_f)
        slater_circuit = slaters_determinant_circuit(Q_mat)
        
        psi_from_circuit = Statevector.from_instruction(slater_circuit)
        psi = slater_statevector(L, N_f)

        overlap = psi.inner(psi_from_circuit)
        fidelity = abs(overlap)**2

        fidelities.append((L, fidelity))

        print("Overlap:", overlap)
        print("Fidelity:", fidelity)

    # Summary
    print("\n\n=== Summary of fidelities ===")
    for L, fidelity in fidelities:
        print(f"L={L}: Fidelity={fidelity}")