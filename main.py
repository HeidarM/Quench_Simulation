# main.py

import sys
import yaml
import argparse

from utils.post_process import post_process_ibm, compute_spectrum_and_plot
from utils.utils import get_job_by_number, print_job_summary, print_jobs_jsonl

from backend.backend import BackendConfig
from runners.run_quench import run_QuenchSpectroscopy, LOG_FILE_IBM, LOG_FILE_SIMULATION


def config_parser(config):
    
    # If job number given, load data from job log instead of config file
    if "job_num" in config and config["task"].startswith("post_process"):

        data_source = config["data_source"].lower()
        LOG_FILE = LOG_FILE_SIMULATION if data_source == "simulation" else LOG_FILE_IBM

        job_data = get_job_by_number(config["job_num"], LOG_FILE)
        print_job_summary(config["job_num"], LOG_FILE)


        config.setdefault("L", job_data["L"])
        config.setdefault("J", job_data["J"])
        config.setdefault("U", job_data["U"])
        config.setdefault("N_Trotter", job_data["N_Trotter"])
        config.setdefault("T", job_data["T"])
        config.setdefault("job_id", job_data["job_id"])

        if data_source == "simulation":
            counts_list = job_data["counts"]
            shots_list = job_data["shots_list"]
        

    L = config["L"]
    T, J, U = config["T"], config["J"], config["U"]
    N_Trotter = config["N_Trotter"]
    N_f = int(L * config.get("fill_fraction", 2.0 / 3)) # default: nf = 2/3
    
    dt = T / N_Trotter

    # Read initial state and n_layers (if field is empty, default: slater)
    initial_state = config.get("initial_state", "slater")
    if isinstance(initial_state, str) and initial_state.lower() == "dga":
        n_layers = config["n_layers"]
    else:
        n_layers = 0 # For slater

    # Read transpile optimization level (default: 0)
    transpile_ol = int(config.get("transpile_optimization_level", 0))

    task = config["task"]
    # -------- simulate --------
    if task == "simulate":
        backend_config = BackendConfig(kind="aer", transpile_ol=transpile_ol, default_precision=1e-2)
        job, results = run_QuenchSpectroscopy(L, N_f, n_layers, dt, J, U, N_Trotter, backend_config=backend_config)

        # Extract counts and shot info for each time step
        counts_list = []
        shots_list = []

        for result in results:
            counts = result.join_data().get_counts()
            shots = sum(counts.values())
            counts_list.append(counts)
            shots_list.append(shots)


        compute_spectrum_and_plot(counts_list, shots_list, L, N_Trotter, dt)

    # -------- run_qc --------
    elif task == "run_qc":
        backend_config = BackendConfig(kind="ibm", transpile_ol=transpile_ol, default_precision=1e-2)
        # job, results = run_QuenchSpectroscopy(Q_mat, dt, J, U, N_Trotter, backend_config=backend_config)
        job, results = run_QuenchSpectroscopy(L, N_f, n_layers, dt, J, U, N_Trotter, backend_config=backend_config)

    # -------- post_process --------
    elif task == "post_process":
        if data_source == "simulation":
            compute_spectrum_and_plot(counts_list, shots_list, L, N_Trotter, dt)
        else:
            post_process_ibm(config["job_id"], L, N_Trotter, dt)
    else:
        raise ValueError(f"Unknown task: {task}")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Config 
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    
    # Jobs list
    parser.add_argument(
        "--jobs",
        type=str,
        help="Which job log to print: 'simulation' or 'ibm'"
    )
    args = parser.parse_args()

    if args.jobs:
        opt = args.jobs.lower()
        if opt == "simulation":
            print_jobs_jsonl(LOG_FILE_SIMULATION)
            sys.exit(0)
        elif opt == "ibm":
            print_jobs_jsonl(LOG_FILE_IBM)
            sys.exit(0)
        else:
            print("Invalid value for --jobs. Choices are: 'simulation' or 'ibm'.")
            sys.exit(2)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config_parser(config)