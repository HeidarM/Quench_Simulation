
# runners/run_quench.py


from backend.backend import BackendConfig, BackendManager
from circuit.circuit_builder import QuenchSpectroscopyCircuits

from utils.utils import append_to_job_log

# For keeping a log over all jobs run and their parameters
LOG_FILE = "job_log.jsonl"


def run_QuenchSpectroscopy(Q_mat, dt, J, U, N_Trotter, backend_config: BackendConfig):
   
    # TODO: Put this into config or input args
    N_shots = 1024 

    N_f, L = Q_mat.shape
    circs = QuenchSpectroscopyCircuits(Q_mat, dt, J, U, N_Trotter, verbose=True)

    backend_manager = BackendManager(backend_config)
    print("\nUsing backend: ", backend_manager.backend.name)
        
    # Submit job
    job = backend_manager.run_sampler(circs, shots=backend_config.shots)
    print("\nJob id = ", job.job_id())

     # Aer returns results immediately; IBM returns a Runtime job
    try:
        results = job.result()  # works for Aer; on IBM this blocks until done
    except Exception:
        results = None


    # Log job information if run on IBM Quantum Hardware
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