# runners/run_VQE_DGA.py

import numpy as np

from qiskit.quantum_info import SparsePauliOp
from qiskit_algorithms.optimizers import SPSA, COBYLA, L_BFGS_B, SLSQP, ADAM, QNSPSA, NFT

from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit
from backend.backend import BackendManager, BackendConfig
from utils.VQE_convergence import SPSAConvergence, VQETrackProgressCallback, KeepBestCallback, cosine_anneal_with_restarts, powerlaw_with_restarts
from circuit.slater_det_circuit import generate_Q_mat




# -------- Free fermion Hamiltonian --------
def _pauli_label(n, ops):
    """
    Build a Pauli label string of length n with little-endian indexing:
    qubit 0 is rightmost char. 'ops' is a dict {q: 'X'/'Y'/'Z'}.

    example: _pauli_label(10, {0: 'X', 5: 'Y', 7: 'Z'}) -> 'IIZIYIIIIX'
    """
    s = ['I'] * n
    for q, p in ops.items():
        s[n - 1 - q] = p
    return ''.join(s)

def free_fermion_H0(L: int, J: float = 1.0) -> SparsePauliOp:
    """
    H0 = -J * sum_{i=0}^{L-2} sum_{spin in {up,down}} ( c†_{i,spin} c_{i+1,spin} + h.c.)

    JW (nearest neighbors) -> -J/2 * (X_i X_{i+1} + Y_i Y_{i+1}), separately in each spin sector.

    Block ordering: [0..L-1]=↑, [L..2L-1]=↓.
    """
    n = 2 * L
    terms = []
    coeffs = []
    for base in (0, L):  # up block, down block
        for i in range(base, base + L - 1):
            # -J/2 * (X_i X_{i+1} + Y_i Y_{i+1})
            terms.append(_pauli_label(n, {i: 'X', i + 1: 'X'})); coeffs.append(-J / 2)
            terms.append(_pauli_label(n, {i: 'Y', i + 1: 'Y'})); coeffs.append(-J / 2)
    return SparsePauliOp.from_list(list(zip(terms, coeffs)))


# ---------------- VQE for DGA states  ----------------
def run_VQE_for_DGA(L: int, N_f: int, n_layers: int, backend_config: BackendConfig,
                    max_iterations:int = 400, fidelity_goal: float = 0.9, history_window: int = 30, init: list | None = None,
                    verbose: bool = False, live_plot: bool = False,
                    print_best_energy: bool = False, print_best_parameter: bool = False, print_best_fidelity: bool = False
                    ):
    if verbose:
        print(f"Running VQE for DGA with L={L}, N_f={N_f}, n_layers={n_layers} on {backend_config.kind} backend")
    
    backend_manager = BackendManager(backend_config)
    ansatz, theta = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers, maximal_spread=True)
    H0 = free_fermion_H0(L, J=1.0)

    Q = generate_Q_mat(L, N_f)
    
    termination_checker = SPSAConvergence(n_layers=n_layers, Q = Q, fidelity_goal=fidelity_goal)

    # Causes one extra evaluations per iteration, for SPSA the termination checker will do this task
    keepBest = None #KeepBestCallback()

    eta, eps = cosine_anneal_with_restarts(
                                            max_iter=max_iterations,
                                            lr_max=0.02,  lr_min=0.0002,   
                                            eps_max=0.03, eps_min=0.001,  
                                            T0=150, T_mult=1.3)


    # eta, eps = powerlaw_with_restarts(
    #                                     max_iter=max_iterations,
    #                                     A=0, a=0.045, alpha=0.360,
    #                                     c=0.01, gamma=0.05101,
    #                                     T0=150, T_mult=1.3)


    optimizer = SPSA(
        maxiter=max_iterations,
        learning_rate=eta,
        perturbation=eps,
        blocking=True,               # helpful: reject steps that increase too much
        allowed_increase=1e-2,       # tweak based on noise scale
        callback=keepBest,             
        termination_checker=termination_checker
    )

    
    if init is None:
        init = np.random.rand(len(theta))


    # To track progress and live plot energy and fidelity
    track_progress_callback = VQETrackProgressCallback(history_window=history_window,
                                                       live_plot = live_plot,
                                                       print_best_energy = print_best_energy,
                                                       print_best_parameter = print_best_parameter,
                                                       print_best_fidelity = print_best_fidelity,
                                                       Q=Q,
                                                       n_layers=n_layers)


    # run VQE
    result = backend_manager.run_vqe(ansatz=ansatz,
                                     hamiltonian=H0,
                                     optimizer=optimizer,
                                     initial_point=init,
                                     callback=track_progress_callback
                                     )
    
    
    # Overwrite with the best-so-far point if it's better than the optimizer's return
    current_energy = float(np.real(result.eigenvalue))
    if track_progress_callback.min_energy < current_energy:
        result.optimal_point = track_progress_callback.min_energy_parameter
        result.eigenvalue = track_progress_callback.min_energy

    

    if verbose:
        # Get how many iterations/evaluations were actually used for convergence
        used_iters = getattr(getattr(result, "optimizer_result", None), "nit", None)
        used_evals = (
        getattr(result, "optimizer_evals", None) or
        getattr(result, "cost_function_evals", None) or
        getattr(getattr(result, "optimizer_result", None), "nfev", None))

        print()
        print("Number of params:", len(theta))
        print("Optimized energy  <H0>:", float(np.real(result.eigenvalue)))
        if used_iters is not None:
            print("Optimizer iterations used:", used_iters)
        if used_evals is not None:
            print("Function evaluations:", used_evals)
        print("Optimal parameters:", result.optimal_point)

    return result, ansatz, theta