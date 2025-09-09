# utils/quench_spectroscopy.py

import numpy as np
import scipy.fft as fft

def quench_spectral_function(Sx_t, dt, pad_time=True):
    """
    2-D FFT of ⟨Sx_i(t)⟩ → QSF(ω,k), normalised to max=1.

    Parameters
    ----------
    Sx_t : list[list[float]]   # shape (N_t, L)
    dt   : float               # time step used in TDVP
    pad_time : bool
        If True, zero-pad the signal in time (doubles resolution, as in
        Appendix A of the paper).

    Returns
    -------
    omega : 1-D np.ndarray  (frequency axis, centred with fftshift)
    k     : 1-D np.ndarray  (lattice momentum axis, centred)
    Q     : 2-D np.ndarray  |FFT2| / max(|FFT2|)
            shape (N_ω, N_k) with origin='lower'
    """


    Sx = np.array(Sx_t)              # (N_t, L)
    Nt, L = Sx.shape

    # optional zero-padding in the time direction
    if pad_time:
        Sx = np.pad(Sx, ((0, Nt), (0, 0)))     # doubles length → 2*Nt

    # 2-D Fourier transform, shifted s.t. k,ω ∈ (−π,π]×(−π/dt,π/dt]
    F   = fft.fftshift(fft.fft2(Sx, norm='forward'))
    Q   = np.abs(F)
    Q  /= Q.max()                   # normalise to [0,1]

    omega = 2*np.pi*fft.fftshift(fft.fftfreq(Sx.shape[0], d=dt))
    k     = 2*np.pi*fft.fftshift(fft.fftfreq(L,          d=1))  # lattice a=1

    return omega, k, Q