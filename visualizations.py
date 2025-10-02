
import matplotlib.pyplot as plt
from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat

from circuit.circuit_builder import QuenchSpectroscopyCircuit



if __name__ == "__main__":
    L = 5
    N_f = 2
    n_layers = 1
    N_Trotter = 1

    dga_circ, _ = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers)

    dga_circ.draw(output="mpl")
    plt.show()

    slater_circ = slaters_determinant_circuit(generate_Q_mat(L, N_f))

    slater_circ.draw(output="mpl")
    plt.show()

    quench_circ_dga = QuenchSpectroscopyCircuit(slater_circ, dt=0.1, J=1.0, U=0.2, N_Trotter=N_Trotter)

    quench_circ_dga.draw(output="mpl")
    plt.show()