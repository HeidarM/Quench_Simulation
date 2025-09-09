# utils/plotting.py

import numpy as np
import matplotlib.pyplot as plt

from utils.quench_spectroscopy import quench_spectral_function


def plot_Sx_t_and_Qwk(Sx_t, dt):

    Sx_t = np.array(Sx_t)  # Shape (N_Trotter, L)

    N_Trotter, L = Sx_t.shape


    # Compute spectral function
    omega, k_vals, Q = quench_spectral_function(Sx_t, dt)

    # Create side-by-side plots
    fig, axs = plt.subplots(1, 2, figsize=(14, 4))

    # Plot ⟨Sx_i(t)⟩ heatmap
    im0 = axs[0].imshow(Sx_t, aspect='auto', origin='lower', cmap='RdBu_r',
                        vmin=-0.5, vmax=0.5, extent=[0, L, 0, dt * N_Trotter])
    cbar0 = fig.colorbar(im0, ax=axs[0])
    cbar0.set_label(r"$\langle S^x_i(t) \rangle$")
    axs[0].set_xlabel("Site i")
    axs[0].set_ylabel("Time t")
    axs[0].set_title("Spin-x expectation values over time")

    # Plot spectral function Q(ω, k)
    im1 = axs[1].imshow(Q, origin='lower', aspect='auto',
                        extent=[k_vals[0], k_vals[-1], omega[0], omega[-1]],
                        cmap='magma')
    cbar1 = fig.colorbar(im1, ax=axs[1])
    cbar1.set_label("Spectral intensity")
    axs[1].set_xlabel("Lattice momentum $k$")
    axs[1].set_ylabel("Frequency $\\omega$")
    axs[1].set_title("Quench Spectral Function $Q(\\omega, k)$")

    plt.tight_layout()
    plt.show()