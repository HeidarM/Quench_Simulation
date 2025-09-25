# tests/optimize_fidelity_test.py
# run using python -m tests.optimize_fidelity_test

import numpy as np
import matplotlib.pyplot as plt


from utils.optimize_fidelity import optimize_thetas_multistart
from circuit.slater_det_circuit import generate_Q_mat
from utils.fast_fidelity import dga_overlap_and_fidelity


def plot_fidelity_vs_L(
    L_list=(4, 5, 6, 7, 8),
    layers_list=(1, 2, 3, 4),
    nf=2/3,
    seed=0,
    n_starts=32,
    maxiter=500
):

    results = {}
    for n_layers in layers_list:
        Ls, Fs = [], []
        for L in L_list:
            N_f = round(nf/2 * L)
            Q = generate_Q_mat(L, N_f).astype(float)
            best_theta, best_fidelity, res  = optimize_thetas_multistart(
                Q, n_layers, seed=seed, n_starts=n_starts, maxiter=maxiter
            )
            Ls.append(L)
            Fs.append(best_fidelity)

            print(f"(L, n_layers) = ({L}, {n_layers:2d})  iters={res.nit:3d}  F={best_fidelity}")
            print("thetas:", best_theta)
            print()
        results[n_layers] = (np.array(Ls), np.array(Fs))

    # ---- plot ----
    fig, ax = plt.subplots(figsize=(4, 3.2), dpi=120)
    for n_layers in layers_list:
        Ls, Fs = results[n_layers]
        ax.plot(Ls, Fs, marker='o', label=rf"$n_{{\mathrm{{layers}}}} = {n_layers}$")

    ax.set_xlabel(r"$L$")
    ax.set_ylabel("Fidelity")
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True)
    ax.grid(False)

    fig.tight_layout()
    return fig, ax, results



if __name__ == "__main__":
  
    nf = 2/3
    # L_list = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
    L_list = [3, 4, 5, 6, 7, 8]
    layers_list = [1, 2, 3, 4]

    
    fig, ax, results = plot_fidelity_vs_L(L_list=L_list, layers_list=layers_list, nf=nf,
                                          seed=0, n_starts=32, maxiter=500)

    plt.show()

    # L = 15
    # nf = 2/3
    # N_f = round(nf/2 * L)
    # n_layers = 3
    # Q = generate_Q_mat(L=8, N_f=3).astype(float)

    # best_theta, best_fidelity, res  = optimize_thetas_multistart(Q = Q, n_layers=n_layers, n_starts=64, maxiter=500)

    # print(f"(L, n_layers) = ({L}, {n_layers:2d})  iters={res.nit:3d}  F={best_fidelity}")
    # print("thetas:", best_theta)