# optimize_fidelity_test.py
# run using python -m tests.optimize_fidelity_test

import numpy as np

from utils.optimize_fidelity import optimize_thetas_multistart
from circuit.slater_det_circuit import generate_Q_mat
from utils.fast_fidelity import dga_overlap_and_fidelity

from circuit.DGA_ansatz_circuit import _evenly_spaced_sites




if __name__ == "__main__":
  
    # L = 10
    nf = 2/3
    # N_f = round(nf/2 * L)

      

    for L in [8, 9, 10, 11, 12]:
        N_f = round(nf/2 * L)
        Q = generate_Q_mat(L, N_f).astype(float)

        print("\n\nFidelity optimization for L = {} and n_f = {}".format(L, N_f))

        for n_layers in [1, 2, 3, 4]:
            # optimize
            best_theta, best_fidelity, res = optimize_thetas_multistart(Q, n_layers, seed=1, n_starts=64, maxiter=1000)
            
            # overlap, fidelity = dga_overlap_and_fidelity(Q, thetas_opt, n_layers)

            print(f"layers={n_layers:2d}  iters={res.nit:3d}  F={best_fidelity}")
            # print("thetas:", best_theta)