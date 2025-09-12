
# runners/run_quench.py

import os
import json
from datetime import datetime
import numpy as np

from backend.backend import BackendConfig, BackendManager
from circuit.circuit_builder import QuenchSpectroscopyCircuits

from post_process import compute_Sx_i
from utils.plotting import plot_Sx_t_and_Qwk

# For keeping a log over all jobs run and their parameters
LOG_FILE = "job_log.jsonl"

def append_to_job_log(entry):
    job_num = 1
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            job_num = sum(1 for _ in f) + 1
    entry["job_num"] = job_num
    entry["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return job_num


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

        job_num = append_to_job_log(job_entry)
        print(f"Logged job #{job_num}")

    elif backend_config.kind == "aer":
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

    return job, results