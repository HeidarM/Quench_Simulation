import numpy as np
from qiskit.quantum_info import Statevector

from backend.backend import BackendConfig
from runners.run_VQE_DGA import run_VQE_for_DGA
from circuit.slater_det_state import slaters_determinant_state, generate_Q_mat


def create_DGA_state(L, N_f, n_layers):

    backend_config_DGA = BackendConfig(kind="aer",  transpile_ol=0, default_precision=1e-2, aer_method="matrix_product_state", aer_options={"runtime_parameter_bind_enable": True})
    result, ansatz, _ = run_VQE_for_DGA(L, N_f, n_layers, backend_config_DGA, max_iterations=1000, verbose=True)
        
    # Bind the optimal parameters to get a concrete state-prep circuit
    theta_star = np.asarray(result.optimal_point, dtype=float)
    # Safer mapping (in case parameter order changes):
    param_dict = dict(zip(list(ansatz.parameters), theta_star))

    dga_state = ansatz.assign_parameters(param_dict, inplace=False)
    dga_state.name = f"DGA(L={L},layers={n_layers})"

    return dga_state


if __name__ == "__main__":

    L = 10
    N_f = 4
    # n_layers = 2

    fidelities = []

    for n_layers in range(5,6):
        print(f"\n\n=== n_layers = {n_layers} ===")
        dga_state = create_DGA_state(L, N_f, n_layers)

        Q_mat = generate_Q_mat(L, N_f)
        slater_state = slaters_determinant_state(Q_mat)

        psi_dga = Statevector.from_instruction(dga_state)
        psi_slater = Statevector.from_instruction(slater_state)

        # print("\n|psi_dga>:", psi_dga.data)
        # print("\n|psi_slater>:", psi_slater.data)

        # norm_dga = psi_dga.inner(psi_dga)
        # norm_slater = psi_slater.inner(psi_slater)
        # print("\nNorm |psi_dga>:", abs(norm_dga))
        # print("Norm |psi_slater>:", abs(norm_slater))

        overlap = psi_dga.inner(psi_slater)
        fidelity = abs(overlap)**2

        fidelities.append((n_layers, fidelity))

        print("Overlap:", overlap)
        print("Fidelity:", fidelity)

    # Summary
    print("\n\n=== Summary of fidelities ===")
    for n_layers, fidelity in fidelities:
        print(f"n_layers={n_layers}: Fidelity={fidelity}")