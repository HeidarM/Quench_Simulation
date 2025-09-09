Reproducing some of the results from arXiv:2501.04649. Using TeNPy for the classical simulation and QiSkit for the quantum algorithms.


## Notes
- V1 and V2 of the paper appears to have some typos in the construction of G(theta), H12(theta) and other details. Mathematica file checks some of these formulas.

## How to run
Can be run by using a config file as

python main.py --config config.yaml

configs/ folder has a few examples.

type of tasks: simulate | run_qc | post_process
  - simulate: simulates quantum computer locally
  - run_qc: run on IBM quantum computer, will store job id and data used in job_log.jsonl file
  - post_process: post process code already run on IBM quantum computer by providing job_id or job_num (from the job_log.jsonl)

Post process computes <S^x_i> and Quench Spectral Function Q(omega,k) from raw measurement bitstrings

Classical simulation (using TeNPy) can be run from root folder as: python -m classical_simulation.1d_Fermi_Hubbard_quench_spectroscopy
Parameters are hardcoded in the classical_simulation/1d_Fermi_Hubbard_quench_spectroscopy.py