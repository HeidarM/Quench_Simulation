# utils.py

import json

def get_job_by_number(job_num, log_file="job_log.jsonl"):
    with open(log_file, "r") as f:
        for line in f:
            entry = json.loads(line)
            if entry["job_num"] == job_num:
                return entry
    raise ValueError(f"Job number {job_num} not found in log.")


def print_job_summary(job_num, log_file="job_log.jsonl"):
    try:
        job = get_job_by_number(job_num, log_file)
    except ValueError as e:
        print(e)
        return

    print(f"\n Job #{job['job_num']} summary from {log_file}:")
    print("-" * 40)
    print(f"  Job ID      : {job['job_id']}")
    print(f"  Timestamp   : {job['timestamp']}")
    print(f"  Backend     : {job['backend']}")
    print(f"  System size : L = {job['L']}, N_f = {job['N_f']}")
    print(f"  Trotter steps: {job['N_Trotter']}")
    print(f"  Time step   : dt = {job['dt']}")
    print(f"  Couplings   : J = {job['J']}, U = {job['U']}")
    print(f"  Shots       : {job['shots']}")
    print("-" * 40)