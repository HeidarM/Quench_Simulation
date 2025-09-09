# run_on_quantum_computer.py
import os
import json
from datetime import datetime
import numpy as np

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


from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as QHWSampler
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from circuit.circuit_builder import QuenchSpectroscopyCircuit, QuenchSpectroscopyCircuits




# def Quantum_HW_run(Q_mat: np.ndarray, dt: float, J: float, U: float, N_Trotter: int):
   
#     N_shots = None #10024

#     N_f, L = Q_mat.shape

#     # 1) Create quantum circuit and measure
#     qc = QuenchSpectroscopyCircuit(Q_mat, dt, J, U, N_Trotter)

#     # To make it work with aer
#     qc = qc.decompose().decompose()

#     qc.measure_all()

#     # # Set up job for quantum hardware
#     # service = QiskitRuntimeService()
#     # backend = service.least_busy(simulator=False, operational=True)
#     # print("\nBackend: ", backend.name)

#     # pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    
#     # # Transpile for hw on local device
#     # qc_transpiled = pass_manager.run(qc)

#     # # Submit job
#     # sampler = QHWSampler(mode=backend)
#     # job     = sampler.run([qc_transpiled], shots=N_shots)
    
#     # job_id = job.job_id()
#     # print("\nJob id = ", job_id)


    

def Quantum_HW_run(Q_mat: np.ndarray, dt: float, J: float, U: float, N_Trotter: int):
   
    N_shots = 1024

    N_f, L = Q_mat.shape

    # 1) Create quantum circuit and measure
    circuits = QuenchSpectroscopyCircuits(Q_mat, dt, J, U, N_Trotter)

    # Set up job for quantum hardware
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    print("\nBackend: ", backend.name)

    pass_manager = generate_preset_pass_manager(optimization_level=1, backend=backend)
    
    # Transpile for hw on local device
    qc_transpiled = pass_manager.run(circuits)

    # # Submit job
    sampler = QHWSampler(mode=backend)
    job     = sampler.run(qc_transpiled, shots=N_shots)
    
    job_id = job.job_id()
    print("\nJob id = ", job_id)

    # Log it
    job_entry = {
        "job_id": job_id,
        "backend": backend.name,
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


    
