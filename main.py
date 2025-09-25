# main.py

import yaml
import argparse
# import numpy as np

from post_process import post_process, compute_Sx_i
from utils.utils import get_job_by_number, print_job_summary, print_jobs_jsonl
from utils.plotting import plot_Sx_t_and_Qwk

from backend.backend import BackendConfig
from runners.run_quench import run_QuenchSpectroscopy, LOG_FILE


def config_parser(config):
    
    # If job number given, load data from job log instead of config file
    if "job_num" in config and config["task"].startswith("post_process"):

        job_data = get_job_by_number(config["job_num"])
        print_job_summary(config["job_num"])


        config.setdefault("L", job_data["L"])
        config.setdefault("J", job_data["J"])
        config.setdefault("U", job_data["U"])
        config.setdefault("N_Trotter", job_data["N_Trotter"])
        config.setdefault("T", job_data["T"])
        config.setdefault("job_id", job_data["job_id"])

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


        print("Computing ⟨Sx_i(t)⟩ as function of time...\n")

        # Build Sx_t (⟨Sx_i(t)⟩ matrix)
        Sx_t = []
        for t in range(N_Trotter):
            counts = counts_list[t]
            shots  = shots_list[t]
            Sx_vals = [compute_Sx_i(counts, i, L, shots) for i in range(L)]
            Sx_t.append(Sx_vals)

        plot_Sx_t_and_Qwk(Sx_t, dt)

    # -------- run_qc --------
    elif task == "run_qc":
        backend_config = BackendConfig(kind="ibm", transpile_ol=transpile_ol, default_precision=1e-2)
        # job, results = run_QuenchSpectroscopy(Q_mat, dt, J, U, N_Trotter, backend_config=backend_config)
        job, results = run_QuenchSpectroscopy(L, N_f, n_layers, dt, J, U, N_Trotter, backend_config=backend_config)

    # -------- post_process --------
    elif task == "post_process":
        post_process(config["job_id"], L, N_Trotter, dt)
    else:
        raise ValueError(f"Unknown task: {task}")


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    parser.add_argument("--jobs", action="store_true", help="Print list of logged jobs")
    args = parser.parse_args()

    if args.jobs:
        print_jobs_jsonl(LOG_FILE)
        raise SystemExit(0)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config_parser(config)