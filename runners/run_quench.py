# runners/run_quench.py

from qiskit.transpiler.passes import RemoveBarriers
from qiskit.transpiler import PassManager

from backend.backend import BackendConfig, BackendManager
from utils.utils import append_to_job_log, print_circuit_summary
from utils.optimize_fidelity import optimize_thetas_multistart
from circuit.circuit_builder import QuenchSpectroscopyCircuits
from circuit.slater_det_circuit import slaters_determinant_circuit, generate_Q_mat
from circuit.DGA_ansatz_circuit import DGA_ansatz_circuit
# from runners.run_VQE_DGA import run_VQE_for_DGA





# For keeping a log over all jobs run and their parameters
LOG_FILE = "job_log.jsonl"
# LOG_FILE = "job_log_test.jsonl"

def run_QuenchSpectroscopy(L: int, N_f: int,
                           n_layers: int, # If > 0 use DGA ansatz; else use Slater det.
                           dt: float, J: float, U: float, N_Trotter: int, backend_config: BackendConfig):
   
    # TODO: Put this into config or input args
    N_shots = 1024 

    # Use DGA ansatz
    if n_layers > 0:
        print(f"Using DGA ansatz with {n_layers} layers")
        backend_config_DGA = BackendConfig(kind="aer",  transpile_ol=0, default_precision=1e-2, aer_method="matrix_product_state", aer_options={"runtime_parameter_bind_enable": True})
        # result, ansatz, _ = run_VQE_for_DGA(L, N_f, n_layers, backend_config_DGA, max_iterations=1000, verbose=True)

        ansatz, theta = DGA_ansatz_circuit(L=L, N_f=N_f, n_layers=n_layers, maximal_spread=True)

        Q = generate_Q_mat(L, N_f).astype(float)
        best_theta, best_fidelity, _ = optimize_thetas_multistart(Q, n_layers, seed=1, n_starts=64, maxiter=1000)
        
        print("Fidelity of DGA state: ", best_fidelity)

        # Safer mapping (in case parameter order changes):
        param_dict = dict(zip(list(ansatz.parameters), best_theta))

        # Initial state = DGA
        initial_state = ansatz.assign_parameters(param_dict, inplace=False)
        initial_state.name = f"DGA(L={L},layers={n_layers})"

    # Use Slater determinant state
    else:
        print("Using Slater determinant state")
        Q_mat = generate_Q_mat(L, N_f)

        # Initial state = Slaters
        initial_state = slaters_determinant_circuit(Q_mat)

    print_circuit_summary(initial_state, "Initial state")


    circuits = QuenchSpectroscopyCircuits(initial_state=initial_state,
                                           dt=dt, J=J, U=U, Max_N_Trotter=N_Trotter, verbose=False)
    backend_manager = BackendManager(backend_config)
    print("\nUsing backend: ", backend_manager.backend.name)

    print("Transpiling circuits for backend with optimization level {}...".format(backend_config.transpile_ol))

    # Remove barriers to get better optimization
    pm = PassManager([RemoveBarriers()])
    circuits_no_barrier = pm.run(circuits)

    # Transpile circuits to backend
    circuits_t = backend_manager.transpile(circuits_no_barrier)

    # Submit job
    job = backend_manager.run_sampler(circuits_t, shots=backend_config.shots)
    print("\nJob submitted. Job id = ", job.job_id())

     # Aer returns results immediately; IBM returns a Runtime job
    try:
        results = job.result()  # works for Aer; on IBM this blocks until done
    except Exception:
        results = None


    # Log the job information if run on IBM Quantum Hardware
    if backend_config.kind == "ibm":# or backend_config.kind == "aer":
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
                        "initial_state": "slater" if n_layers == 0 else "DGA",
                        "n_layers": n_layers,
                        "initial_state fidelity": best_fidelity if n_layers > 0 else 1,
                    }

        job_num = append_to_job_log(job_entry, LOG_FILE)
        print(f"Logged job #{job_num}")

    return job, results
