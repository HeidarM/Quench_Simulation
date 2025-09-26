# Classical simulation of 1D Fermi-Hubbard model quench spectroscopy
# using Time-Dependent Variational Principle (TDVP) with Matrix Product States (MPS)
# Reproducing 2501.04649

# Run from root folder: python -m classical_simulation.1d_Fermi_Hubbard_quench_spectroscopy


import numpy as np
from tenpy.algorithms import tdvp
from tenpy.networks.mps import MPS
from tenpy.models.hubbard import FermiHubbardModel
from tenpy.algorithms import dmrg

import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import scipy.fft as fft

from utils.quench_spectral_function import quench_spectral_function



def ground_state_dmrg(model, product_state, chi_max=200):
    """
        Find ground state of model using DMRG.

        Returns MPS and energy of ground state.
    """
    # build an MPS from the product state
    psi0 = MPS.from_product_state(model.lat.mps_sites(),
                                  product_state,
                                  bc=model.lat.bc_MPS,
                                  form='B')

    # run a standard two-site DMRG sweep
    dmrg_params = {
        'trunc_params': {'chi_max': chi_max, 'svd_min': 1e-10},
        'mixer': True,     # helps convergence for small systems
        'max_E_err': 1e-10,
        'combine': True
    }
    eng = dmrg.TwoSiteDMRGEngine(psi0, model, dmrg_params)
    E0, psi0 = eng.run()

    return psi0, E0


def apply_spin_quench(psi, j_site, theta):
    """
    Apply a local spin-rotation quench  Q_j = exp(i θ S^x_j)

    Works with SpinHalfFermionSite when     cons_Sz = None.
    """
    site = psi.sites[j_site]

    # Sx = ½ (Sp + Sm)
    Sx = 0.5 * (site.Sp + site.Sm)

    # Q = exp(i θ Sx) = cos(θ/2) I + 2i sin(θ/2) Sx
    Q = np.cos(theta/2.0) * site.Id + 2.0j * np.sin(theta/2.0) * Sx

    # Apply the unitary; TenPy supplies Q† automatically
    psi.apply_local_op(j_site, Q, unitary=True)




def TDVP_FermiHubbard():

    # Model parameters
    L = 51 #51
    U = 3.0
    Ne = 34 #34  # total number of electrons
    N_up = Ne // 2
    N_down = Ne - N_up
    chi = 200
    delta_t = 0.002 # 0.002

    # Quench parameters
    quench_site = L // 2
    quench_theta = np.pi / 4

    model_params = {
        'L': L,
        't': 1.0,
        'U': U,
        'mu': 0.0,
        'V': 0.0,
        'bc_MPS': 'finite',
        'cons_N': 'N',
        'cons_Sz': None      #  DO NOT conserve Sᶻ  <-- important
    }

    model = FermiHubbardModel(model_params)

    # Initialize all sites as empty
    product_state = ["empty"] * L

    # Randomly place spin-up electrons
    up_indices = random.sample(range(L), N_up)
    for i in up_indices:
        product_state[i] = "up"

    # Randomly place spin-down electrons on unoccupied or singly-occupied sites
    available_for_down = [i for i in range(L) if product_state[i] != "down"]
    down_indices = random.sample(available_for_down, N_down)
    for i in down_indices:
        if product_state[i] == "up":
            product_state[i] = "full"
        elif product_state[i] == "empty":
            product_state[i] = "down"

    # Find the ground state
    psi, E0 = ground_state_dmrg(model, product_state)

    # Apply quench at t = 0
    apply_spin_quench(psi, j_site=quench_site, theta=quench_theta)

    tdvp_params = {
        'start_time': 0,
        'dt': delta_t,
        'N_steps': 1,
        'trunc_params': {
            'chi_max': chi,
            'svd_min': 1.e-10,
            'trunc_cut': None
        }
    }

    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, model, tdvp_params)
    
    # Containers for observables
    times = []
    S_mid = []
    Es = []
    Sx_t = []

    two_site_steps = int(10.0 / delta_t)
    one_site_steps = int(10.0 / delta_t)

    def measure():
        t_now = tdvp_engine.evolved_time
        times.append(t_now)

        # Entropy / energy
        S_mid.append(psi.entanglement_entropy(bonds=[L // 2])[0])
        Es.append(model.H_MPO.expectation_value(psi))

        # ----------- site-resolved ⟨Sx⟩ -----------
        Sx_now = [psi.expectation_value('Sx', i).item() for i in range(L)]

        Sx_t.append(Sx_now)


    measure()
    for i in range(two_site_steps):
        tdvp_engine.run()
        measure()

    tdvp_engine = tdvp.SingleSiteTDVPEngine.switch_engine(tdvp_engine)
    for i in range(one_site_steps):
        tdvp_engine.run()
        measure()

    return times, Sx_t, S_mid, Es, delta_t


def plot_TDVP(times, Sx_t, S_mid, Es, two_site_steps=30, dt=0.05):

    fig = plt.figure(figsize=(11, 9))     # extra width for two heat-maps
    gs  = gridspec.GridSpec(
        3, 1,
        height_ratios=[1, 1, 2],
        hspace=0.32
    )

    # ---------- entropy ----------------------------------------------
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(times, S_mid, lw=1.4)
    ax0.set_ylabel(r'entropy $S$')
    ax0.set_xlim(times[0], times[-1])

    # ---------- energy -----------------------------------------------
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax1.plot(times, np.array(Es) - Es[0], lw=1.4)
    ax1.set_ylabel(r'$E(t)-E(0)$')
    ax1.set_xlabel(r'time $t$')

    # red guide line for engine switch (top panels)
    t_switch = two_site_steps * dt
    for ax in (ax0, ax1):
        ax.axvline(t_switch, color='red', lw=1, ls='--')

    gs_bottom = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=gs[2], wspace=0.25
    )

    ax2 = fig.add_subplot(gs_bottom[0])
    Sx_arr = np.squeeze(np.array(Sx_t))        # (N_t, L)
    im2 = ax2.imshow(
        Sx_arr,
        origin='lower',
        aspect='auto',
        extent=[0, Sx_arr.shape[1]-1, times[0], times[-1]]
    )
    ax2.set_xlabel(r'site $i$')
    ax2.set_ylabel(r'time $t$')
    ax2.set_title(r'$\langle S^{x}_{i}(t) \rangle$')
    ax2.set_xlim(-0.5, Sx_arr.shape[1]-0.5)
    ax2.set_box_aspect(1)                      # square subplot box
    cbar2 = fig.colorbar(im2, ax=ax2, pad=0.02)
    cbar2.set_label(r'$\langle S^{x} \rangle$')
    ax2.axhline(t_switch, color='red', lw=1, ls='--')

    # ---------- (iii-b) |Q(k,ω)| heat-map ---------------------------------
    ω, k, Q = quench_spectral_function(Sx_t, dt)

    ax3 = fig.add_subplot(gs_bottom[1])
    im3 = ax3.imshow(
        Q,
        origin='lower',
        aspect='auto',
        extent=[k[0], k[-1], ω[0], ω[-1]]
    )
    ax3.set_ylim(0, 2*np.pi) 
    ax3.set_xlabel(r'momentum $k$')
    ax3.set_ylabel(r'frequency $\omega$')
    ax3.set_title(r'normalised $|{\cal Q}(k,\omega)|$')
    ax3.set_box_aspect(1)
    cbar3 = fig.colorbar(im3, ax=ax3, pad=0.02)
    cbar3.set_label(r'$|{\cal Q}|/|{\cal Q}|_{\max}$')

    plt.tight_layout()
    plt.show()




if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    times, Sx_t, S_mid, Es, dt = TDVP_FermiHubbard()
    two_site_steps = int(10.0 / dt)
    plot_TDVP(times, Sx_t, S_mid, Es,
                  two_site_steps=two_site_steps, dt=dt)
