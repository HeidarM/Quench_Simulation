
# runners/run_quench.py

import numpy as np

from backend.backend import BackendConfig, BackendManager
from circuit.circuit_builder import QuenchSpectroscopyCircuits
from runners.run_VQE_DGA import run_VQE_for_DGA
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat

from utils.utils import append_to_job_log


# For keeping a log over all jobs run and their parameters
LOG_FILE = "job_log.jsonl"


def run_QuenchSpectroscopy(L: int, N_f: int,
                           n_layers: int, # If > 0 use DGA ansatz; else use Slater det.
                           dt: float, J: float, U: float, N_Trotter: int, backend_config: BackendConfig):
   
    # TODO: Put this into config or input args
    N_shots = 1024 

    # Use DGA ansatz
    if n_layers > 0:
        print(f"Using DGA ansatz with {n_layers} layers")
        backend_config_DGA = BackendConfig(kind="aer",  transpile_ol=0, default_precision=1e-2, aer_method="matrix_product_state", aer_options={"runtime_parameter_bind_enable": True})
        result, ansatz, _ = run_VQE_for_DGA(L, N_f, n_layers, backend_config_DGA, max_iterations=1000, verbose=True)
        
        # Bind the optimal parameters to get a concrete state-prep circuit
        theta_star = np.asarray(result.optimal_point, dtype=float)
        # Safer mapping (in case parameter order changes):
        param_dict = dict(zip(list(ansatz.parameters), theta_star))

        dga_state = ansatz.assign_parameters(param_dict, inplace=False)
        dga_state.name = f"DGA(L={L},layers={n_layers})"

        initial_state = dga_state
    # Use Slater determinant state
    else:
        print("Using Slater determinant state")
        Q_mat = generate_Q_mat(L, N_f)
        initial_state = slaters_determinant_circuit(Q_mat)

    circs = QuenchSpectroscopyCircuits(initial_state=initial_state,
                                           dt=dt, J=J, U=U, Max_N_Trotter=N_Trotter, verbose=False)
    backend_manager = BackendManager(backend_config)
    print("\nUsing backend: ", backend_manager.backend.name)
        
    # Submit job
    job = backend_manager.run_sampler(circs, shots=backend_config.shots)
    print("\nJob submitted. Job id = ", job.job_id())

     # Aer returns results immediately; IBM returns a Runtime job
    try:
        results = job.result()  # works for Aer; on IBM this blocks until done
    except Exception:
        results = None


    # Log the job information if run on IBM Quantum Hardware
    if backend_config.kind == "ibm":
        job_entry = {
        "job_id": job.job_id(),
        "backend": backend_manager.backend.name,
        "L": L,
        "N_f": N_f,
        "dt": dt,
        "J": J,
        "U": U,
        "N_Trotter": N_Trotter,
        "shots": N_shots,
        }

        job_num = append_to_job_log(job_entry, LOG_FILE)
        print(f"Logged job #{job_num}")

    return job, results