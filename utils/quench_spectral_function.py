# utils/quench_spectroscopy.py

import numpy as np
import scipy.fft as fft


def quench_spectral_function(Sx_t, dt, pad_time=True, reflect_time=True):
    """
    2-D FFT of ⟨Sx_i(t)⟩ → QSF(ω,k), normalised to max=1.

    Parameters
    ----------
    Sx_t : list[list[float]]   # shape (N_t, L)
    dt   : float               # time step used in TDVP/time evolution
    pad_time : bool
        If True, append zeros in time (T → T' ; as described in appendix of paper).
    reflect_time : bool
        If True, make an even extension around t=0 after padding:
        f''(-t) = f''(t), f''(t>0) = f'(t). This doubles the effective window
        again (Δω → Δω/2).

    Returns
    -------
    omega : 1-D np.ndarray  (frequency axis, centred with fftshift)
    k     : 1-D np.ndarray  (momentum axis, centred)
    Q     : 2-D np.ndarray  |FFT2| / max(|FFT2|)
    """

    Sx = np.array(Sx_t, copy=False)   # (N_t, L)
    Nt, L = Sx.shape
    T = Nt * dt
    J = 1
    omega_max = 6.0 * J

    # --- choose target padding length N' as in appendix ---
    if pad_time:
        N_prime = int(np.ceil((2*np.pi * Nt * Nt) / (omega_max * T)))
        N_prime = max(2*Nt, N_prime)           # at least double

        if N_prime > Nt:
            Sx = np.pad(Sx, ((0, N_prime - Nt), (0, 0)))

    # --- even reflection around t=0 ---
    if reflect_time:
        Sx = np.concatenate([Sx[::-1, :], Sx], axis=0)

    F = fft.fftshift(fft.fft2(Sx, norm='forward'))
    Q = np.abs(F)
    Q /= Q.max() if Q.size and Q.max() != 0 else 1.0

    omega = 2*np.pi*fft.fftshift(fft.fftfreq(Sx.shape[0], d=dt))
    k     = 2*np.pi*fft.fftshift(fft.fftfreq(L,          d=1))
    return omega, k, Q
