# run_simulation.py

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer.primitives import SamplerV2
from qiskit.result import QuasiDistribution
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector

from circuit.circuit_builder import QuenchSpectroscopyCircuits
from qiskit_ibm_runtime import QiskitRuntimeService

from post_process import compute_Sx_i
from utils.plotting import plot_Sx_t_and_Qwk





def test_run(Q_mat: np.ndarray, dt: float, J: float, U: float, N_Trotter: int):

    N_shots = None #10024

    N_f, L = Q_mat.shape

    # Circuits of each time step
    circuits = QuenchSpectroscopyCircuits(Q_mat, dt, J, U, N_Trotter, verbose=True)

    # Run all circuits in a batch
    print("\nRunning circuits...\n")
    job = SamplerV2().run(circuits, shots=N_shots)
    results = job.result()

    # Extract counts and shot info for each time step
    counts_list = []
    shots_list = []

    for result in results:
        counts = result.join_data().get_counts()
        shots = sum(counts.values())
        counts_list.append(counts)
        shots_list.append(shots)


    print("Computing ⟨Sx_i(t)⟩ as function of time...\n")

    # Build Sx_t (⟨Sx_i(t)⟩ matrix)
    Sx_t = []
    for t in range(N_Trotter):
        counts = counts_list[t]
        shots  = shots_list[t]
        Sx_vals = [compute_Sx_i(counts, i, L, shots) for i in range(L)]
        Sx_t.append(Sx_vals)

    plot_Sx_t_and_Qwk(Sx_t, dt)

