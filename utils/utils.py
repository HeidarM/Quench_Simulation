# utils/utils.py

import os
import json
from datetime import datetime

import json
from pathlib import Path



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


def append_to_job_log(entry, LOG_FILE):
    """
    Appends an entry to a JSONL log file.
    """
    job_num = 1
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, "r") as f:
            job_num = sum(1 for _ in f) + 1
    entry["job_num"] = job_num
    entry["timestamp"] = datetime.now().isoformat()

    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return job_num



def print_circuit_summary(qc, circuit_name: str):
    """
    Prints a summary of a quantum circuit.
    """
    ops = qc.decompose(reps=10).count_ops()
    twoq = sum(1 for inst, qargs, _ in qc.data if len(qargs) >= 2)
    print()
    print("=== " + circuit_name + " summary ===")
    print("qubits:            ", qc.num_qubits)
    print("depth:             ", qc.depth())
    print("instructions:      ", qc.size())
    print("multi-qubit gates: ", twoq)
    print("SWAPs:             ", ops.get('swap', 0))
    print("ops breakdown:     ", dict(ops))
    print()



def print_jobs_jsonl(LOG_FILE):
    """
    Read a JSONL file where each line is a job record and print a compact table.
    """
    path = Path(LOG_FILE)
    if not path.exists():
        raise FileNotFoundError(path)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {ln}: {e}") from e

            # Pull fields with safe defaults
            rows.append({
                "job_num": rec.get("job_num", ""),
                "job_id": rec.get("job_id", ""),
                "backend": rec.get("backend", ""),
                "L": rec.get("L", ""),
                "N_f": rec.get("N_f", ""),
                "dt": rec.get("dt", ""),
                "J": rec.get("J", ""),
                "U": rec.get("U", ""),
                "N_Trotter": rec.get("N_Trotter", ""),
                "shots": rec.get("shots", ""),
                "initial_state": rec.get("initial_state", ""),
                "n_layers": rec.get("n_layers", 0 if str(rec.get("initial_state","")).lower()=="slater" else ""),
                "fidelity": rec.get("initial_state fidelity", ""),
                "timestamp": rec.get("timestamp", ""),
            })

    # Sort by job_num if available and comparable
    try:
        rows.sort(key=lambda r: (r["job_num"] is "", r["job_num"]))  # blanks last
    except TypeError:
        pass  # mixed types, skip sorting

    # Columns to show (header, key, formatter)
    cols = [
        ("#", "job_num", str),
        ("backend", "backend", str),
        ("L", "L", str),
        ("N_f", "N_f", str),
        ("dt", "dt", lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)),
        ("J", "J", lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)),
        ("U", "U", lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)),
        ("N_Trot", "N_Trotter", str),
        ("shots", "shots", str),
        ("state", "initial_state", str),
        ("layers", "n_layers", str),
        ("fidelity", "fidelity", lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else str(v)),
        ("job_id", "job_id", str),
        ("timestamp", "timestamp", str),
    ]

    # Compute column widths
    def cell(row, key, fmt): 
        v = row.get(key, "")
        try:
            return fmt(v)
        except Exception:
            return str(v)

    data = [[cell(r, k, f) for _, k, f in cols] for r in rows]
    headers = [h for h, _, _ in cols]
    widths = [max(len(h), *(len(r[i]) for r in data)) for i, h in enumerate(headers)]

    # Print header
    line_sep = "  ".join("-" * w for w in widths)
    header_row = "  ".join(h.ljust(w) for h, w in zip(headers, widths))
    print()
    print(header_row)
    print(line_sep)

    # Print rows
    for r in data:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))

    print()