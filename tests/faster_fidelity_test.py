# faster_fidelity_test.py
# run using python -m tests.faster_fidelity_test

import numpy as np
from time import perf_counter

from qiskit.quantum_info import Statevector

from circuit.slater_det_circuit import generate_Q_mat, slater_statevector
from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit

from utils.fast_fidelity import dga_overlap_and_fidelity



def create_DGA_circuit(L, N_f, n_layers, thetas):

    ansatz, theta_param = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers)

    return ansatz.assign_parameters(dict(zip(theta_param, thetas)))



if __name__ == "__main__":
 

    L_values = [2, 3, 4, 5, 6, 7, 8, 9]
    # L_values = [10, 11, 12]
    n_layers = 1
    nf = 2/3
    K = 10

    results = {}

    rng = np.random.default_rng(0)

    for L in L_values:
        N_f = round(nf/2 * L)
        Q_mat = generate_Q_mat(L, N_f)
        psi_slater = slater_statevector(L, N_f)

        slow = []
        fast = []
        t_slow = []
        t_fast = []

        for _ in range(K):
            thetas =rng.random(n_layers * (L - 1))

            # Slow method
            t0 = perf_counter()
            dga_circuit = create_DGA_circuit(L, N_f, n_layers, thetas)
            psi_dga = Statevector.from_instruction(dga_circuit)
            overlap_slater_DGA = psi_dga.inner(psi_slater)
            fidelity_slater_DGA = abs(overlap_slater_DGA) ** 2
            t1 = perf_counter()
            slow.append(float(fidelity_slater_DGA))
            t_slow.append(t1 - t0)

            # Fast method
            t2 = perf_counter()
            _, fidelity_fast = dga_overlap_and_fidelity(Q_mat, thetas, n_layers)
            t3 = perf_counter()
            fast.append(float(fidelity_fast))
            t_fast.append(t3 - t2)

        slow = np.array(slow)
        fast = np.array(fast)
        diff = slow - fast
        t_slow = np.array(t_slow)
        t_fast = np.array(t_fast)

        results[L] = {
            "slow": slow, "fast": fast, "diff": diff,
            "t_slow": t_slow, "t_fast": t_fast,
        }

        print(f"\nL={L} (N_f={N_f})")
        print("max|diff|:", np.max(np.abs(diff)))
        print("mean diff:", np.mean(diff))

        print(f"slow time: mean={t_slow.mean():.6e}s, min={t_slow.min():.6e}s, max={t_slow.max():.6e}s")
        print(f"fast time: mean={t_fast.mean():.6e}s, min={t_fast.min():.6e}s, max={t_fast.max():.6e}s")

        print(f"speedup (mean): {t_slow.mean()/t_fast.mean():.2f}x")
