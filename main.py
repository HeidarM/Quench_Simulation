# main.py

import yaml
import argparse
import numpy as np

from run_simulation import test_run
from run_on_quantum_computer import Quantum_HW_run
from post_process import post_process
from utils.utils import get_job_by_number, print_job_summary


def generate_Q_mat(L, N_f):
    # Wave function for OBC
    j  = np.arange(L)                      # site indices 0 … L−1
    n  = np.arange(1, N_f + 1)[:, None]    # mode numbers 1 … N_f (column)
    k  = n * np.pi / (L + 1)               # quantised momenta (N_f×1)

    Q_mat  = np.sqrt(2.0 / (L + 1)) * np.sin(k * (j + 1))

    return Q_mat

def main(config):
    
    # If job number given, load data from job log instead of config file
    if "job_num" in config and config["task"].startswith("post_process"):

        job_data = get_job_by_number(config["job_num"])
        print_job_summary(config["job_num"])


        config.setdefault("L", job_data["L"])
        config.setdefault("J", job_data["J"])
        config.setdefault("U", job_data["U"])
        config.setdefault("N_Trotter", job_data["N_Trotter"])
        config.setdefault("dt", job_data["dt"])
        config.setdefault("job_id", job_data["job_id"])

    L = config["L"]
    dt, J, U = config["dt"], config["J"], config["U"]
    N_Trotter = config["N_Trotter"]
    N_f = int(L * config.get("fill_fraction", 1.0 / 3))
    Q_mat = generate_Q_mat(L, N_f)

    task = config["task"]
    if task == "simulate":
        test_run(Q_mat, dt, J, U, N_Trotter)
    elif task == "run_qc":
        Quantum_HW_run(Q_mat, dt, J, U, N_Trotter)
    elif task == "post_process":
        post_process(config["job_id"], L, N_Trotter, dt)
    else:
        raise ValueError(f"Unknown task: {task}")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    main(config)