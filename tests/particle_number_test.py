# tests/particle_number_test.py
# run using python -m tests.particle_number_test

# Quick sanity check that number of fermions is as expected

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp

from backend.backend import BackendConfig, BackendManager
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat
from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit



def number_operator(M: int) -> SparsePauliOp:
    # N_tot = sum_``i (I - Z_i)/2 = (M/2) I - (1/2) * sum_i Z_i
    terms = [('I' * M, M / 2.0)]
    for i in range(M):
        s = ['I'] * M               # Identity operator on every site
        s[i] = 'Z'                  # Z on site i
        terms.append((''.join(s), -0.5))
    return SparsePauliOp.from_list(terms)

def expval_total_particles_estimator(qc: QuantumCircuit) -> float:
    bm = BackendManager(BackendConfig(
        kind="aer",
        aer_method="matrix_product_state",
        aer_options={"runtime_parameter_bind_enable": True},
    ))

    qc_transpiled = bm.transpile(qc)

    op = number_operator(qc.num_qubits)
    job = bm.estimator().run([(qc_transpiled, op)])
    res = job.result()

    return float(res[0].data.evs)

if __name__ == "__main__":
    L = 10
    nf = 2/3
    N_f = round(nf/2*L)  # fermions per spin sector
    n_layers = 2


    print("There should be {} fermions per spin sector".format(N_f))

    Q_mat = generate_Q_mat(L, N_f)
    slater_circuit = slaters_determinant_circuit(Q_mat).decompose().decompose()

    val_slater = expval_total_particles_estimator(slater_circuit)
    print("\nFor slater state:")
    print(f"<N_tot> ≈ {val_slater:.8f}  (expected {2*N_f})")


    DGA_circuit, theta = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers)
    values = list(np.random.rand(n_layers * (L - 1)))      
    param_dict = dict(zip(list(theta), values))            
    DGA_bound = DGA_circuit.assign_parameters(param_dict, inplace=False)


    val_dga = expval_total_particles_estimator(DGA_bound)
    print("\nFor DGA state:")
    print(f"<N_tot> ≈ {val_dga:.8f} (expected {2*N_f})")

    