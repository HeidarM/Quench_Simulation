# utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt

from utils.quench_spectral_function import quench_spectral_function


def plot_Sx_t_and_Qwk(Sx_t, dt, omega_min=None, omega_max=None):

    Sx_t = np.array(Sx_t)  # Shape (N_Trotter, L)

    N_Trotter, L = Sx_t.shape


    # Compute spectral function
    omega, k_vals, Q = quench_spectral_function(Sx_t, dt, pad_time=True, reflect_time=True)

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    

    # Plot ⟨Sx_i(t)⟩ heatmap
    im0 = axs[0].imshow(Sx_t, aspect='equal', origin='lower', cmap='RdBu_r',
                        vmin=-0.5, vmax=0.5)
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label(r"$\langle S^x_i(t) \rangle$")
    axs[0].set_xlabel("Site i")
    axs[0].set_ylabel("Time t")
    axs[0].set_title("Spin-x expectation values over time")

    # Plot spectral function Q(ω, k)
    im1 = axs[1].imshow(Q, origin='lower', aspect='equal',
                        extent=[k_vals[0], k_vals[-1], omega[0], omega[-1]],
                        cmap='Blues'#, interpolation='bilinear'
                        )
    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label("Spectral intensity")
    axs[1].set_xlabel("Lattice momentum $k$")
    axs[1].set_ylabel("Frequency $\\omega$")
    axs[1].set_title("Quench Spectral Function $Q(\\omega, k)$")

    if (omega_min is not None) or (omega_max is not None):
        # fall back to current limits if one bound is None
        cur_min, cur_max = axs[1].get_ylim()
        axs[1].set_ylim(
            omega_min if omega_min is not None else cur_min,
            omega_max if omega_max is not None else cur_max
        )

    plt.tight_layout()
    plt.show()