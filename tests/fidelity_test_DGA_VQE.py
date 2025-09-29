# tests/fidelity_test_DGA_VQE.py
# run using python -m tests.fidelity_test_DGA_VQE

import numpy as np
from qiskit.quantum_info import Statevector


from backend.backend import BackendConfig
from runners.run_VQE_DGA import run_VQE_for_DGA
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat, slater_statevector

from utils.optimize_fidelity import optimize_thetas_multistart
from utils.fast_fidelity import dga_overlap_and_fidelity



def create_DGA_circuit(L, N_f, n_layers, init):
    
    # Using aer backend with matrix product states
    backend_config_DGA = BackendConfig(kind="aer",  transpile_ol=0, default_precision=1e-2, aer_method="matrix_product_state", aer_options={"runtime_parameter_bind_enable": True})
    
    # Run VQE
    result, ansatz, _ = run_VQE_for_DGA(L, N_f, n_layers, backend_config_DGA,
                                        max_iterations=10000, fidelity_goal=0.95, history_window=10,
                                        init=init,
                                        verbose=True, live_plot=True, print_best_energy=False, print_best_parameter=False, print_best_fidelity=True
                                        )
        
    # Bind the optimal parameters to get a concrete state-prep circuit
    theta_star = np.asarray(result.optimal_point, dtype=float)
    # Safer mapping (in case parameter order changes):
    param_dict = dict(zip(list(ansatz.parameters), theta_star))

    dga_state = ansatz.assign_parameters(param_dict, inplace=False)
    dga_state.name = f"DGA(L={L},layers={n_layers})"

    return dga_state



if __name__ == "__main__":

    L = 6
    nf = 2/3
    N_f = round(nf/2*L) # Number of fermions per spin sector
    n_layers = 3
    fast_optimizer = True # True will use a much faster optimizer instead of VQE, and use it as initial

    init = None

    # Faster way to find optimal parameters, than using VQE.
    if fast_optimizer:
        Q = generate_Q_mat(L, N_f).astype(float)
        init, _, _ = optimize_thetas_multistart(Q, n_layers, n_starts=32, maxiter=100)

    # Q_mat = generate_Q_mat(L, N_f)
    # slater_circuit = slaters_determinant_circuit(Q_mat)
    # psi_slater = Statevector.from_instruction(slater_circuit)

    psi_slater = slater_statevector(L, N_f) # Much faster


    print(f"\n\n=== n_layers = {n_layers} ===")
    dga_circuit = create_DGA_circuit(L, N_f, n_layers, init)
    psi_dga = Statevector.from_instruction(dga_circuit)
    overlap = psi_dga.inner(psi_slater)
    fidelity = abs(overlap)**2


    print("Overlap:", overlap)
    print("Fidelity:", fidelity)