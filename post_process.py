#post_process.py

from collections import Counter
from matplotlib import pyplot as plt

import numpy as np

from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.result import QuasiDistribution

from utils.plotting import plot_Sx_t_and_Qwk




def compute_Sx_i(measurement_counts: dict, i: int, L: int, N_shots: int):
    """
    Compute <S^x_i> from raw measurement bitstrings.

    Parameters:
    - measurement_counts: dict mapping bitstring -> count
    - i: site index (0-based)
    - L: total number of sites
    - N_shots: total number of shots

    Returns:
    - Expectation value <S^x_i> =0*P00 + P01 - P10 + 0*P11 = P01 - P10
            (measurement in computational basis after applying BXY)
    """
    Sx_total = 0.0


    for bitstring, count in measurement_counts.items():
        # Qiskit uses little-endian: qubit 0 is rightmost bit
        bitstring = bitstring[::-1]

        up = bitstring[2*i]
        down = bitstring[2*i + 1]

        if up == '0' and down == '1':
            Sx_val = +1.0
        elif up == '1' and down == '0':
            Sx_val = -1.0
        else:
            Sx_val = 0.0

        Sx_total += count * Sx_val

    return Sx_total / N_shots

# def post_process(job_id, L):
#     service = QiskitRuntimeService()
#     job = service.job(job_id)
#     print("\nJob status: \n", job.status())

#     result = job.result()[0]

#     counts = result.join_data().get_counts()
#     # print("Shot counts:", counts)                        


#     shots  = sum(counts.values())
#     quasi  = QuasiDistribution({int(b, 2): v / shots for b, v in counts.items()})
#     # print("Quasi-probs :", quasi.binary_probabilities())


#     print(quasi.values())
#     print(1.0/shots) 
#     print(sum(quasi.values()))

#     # plot_histogram(counts, title="Measurement result")    # bar chart
#     # plt.show()

#     Sx_vals = [compute_Sx_i(counts, i, L, shots) for i in range(L)]

#     print(Sx_vals)

#     plt.plot(range(L), Sx_vals, 'o-')
#     plt.xlabel("site i")
#     plt.ylabel("<Sx_i>")
#     plt.title("Local ⟨Sx⟩ after quench")
#     plt.grid(True)
#     plt.ylim(-0.6, 0.6)
#     plt.show()







def post_process(job_id, L, N_Trotter, dt):
    service = QiskitRuntimeService()
    job = service.job(job_id)
    print("\nJob status: \n", job.status())

    results = job.result()

    counts_list = [result.join_data().get_counts() for result in results]        
    shots_list  = [sum(counts.values()) for counts in counts_list]


    print("Computing ⟨Sx_i(t)⟩ as function of time...\n")

    # Build Sx_t (⟨Sx_i(t)⟩ matrix)
    Sx_t = []
    for t in range(N_Trotter):
        counts = counts_list[t]
        shots  = shots_list[t]
        Sx_vals = [compute_Sx_i(counts, i, L, shots) for i in range(L)]
        Sx_t.append(Sx_vals)


    plot_Sx_t_and_Qwk(Sx_t, dt)







