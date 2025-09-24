# backend/backend.py

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any, Sequence, Union

from qiskit import transpile, QuantumCircuit
from qiskit.providers.backend import BackendV2
from qiskit.quantum_info import SparsePauliOp
# from qiskit_algorithms import VQE

# simulators
from qiskit_aer import AerSimulator
from qiskit_aer.primitives import SamplerV2 as AerSampler, EstimatorV2 as AerEstimator

# IBM hardware
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime import SamplerV2 as IBMSampler, EstimatorV2 as IBMEstimator

# Bug fixed VQE class
from BUG_FIXES.VQE_CALLBACK_BUG_FIX import VQE_CALLBACK_BUG_FIX as VQE




@dataclass          # @dataclass auto-generates __init__, __repr__, __eq__, etc. for this config class
class BackendConfig:

    # "aer" (simulation) or "ibm" (quantum hardware)
    kind: str = "aer"
    # Backend (if None, choose least_busy())
    backend_name: Optional[str] = None
    # transpile settings
    transpile_ol: int = 0
    transpile_kwargs: Optional[Dict[str, Any]] = None

    # sampling / estimation
    shots: Optional[int] = None                 # if None on Aer Sampler → quasi-probs
    default_precision: Optional[float] = 1e-2  # Aer Estimator "shots-like"

    aer_method: Optional[str] = None               # e.g. "matrix_product_state", "statevector", ...
    aer_options: Optional[Dict[str, Any]] = None  # any other Aer options (e.g. mps params)

    # runtime options for IBM (optional)
    runtime_options: Optional[Dict[str, Any]] = None


class BackendManager:
    def __init__(self, config: BackendConfig):
        self.config = config
        self.service = None

        if config.kind == "aer":
            self.backend: BackendV2 = AerSimulator()
            
            if config.aer_method is not None:
                self.backend.set_options(method=config.aer_method)
            if config.aer_options:
                self.backend.set_options(**config.aer_options)

            self._sampler = AerSampler.from_backend(self.backend)
            
            estimator_options = {}
            if config.shots is not None:
                estimator_options["shots"] = config.shots
            elif config.default_precision is not None:
                estimator_options["default_precision"] = config.default_precision
            self._estimator = AerEstimator.from_backend(self.backend, options=estimator_options)


        elif config.kind == "ibm":
            self.service = QiskitRuntimeService()
            if config.backend_name:
                self.backend = self.service.backend(config.backend_name)
            else:
                self.backend = self.service.least_busy(simulator=False, operational=True)

            # Build IBM Runtime primitives bound to the backend (mode=backend)
            sampler_opts = {} if config.runtime_options is None else dict(config.runtime_options)
            if config.shots is not None:
                sampler_opts["shots"] = config.shots
            self._sampler = IBMSampler(mode=self.backend, options=sampler_opts)

            estimator_opts = {} if config.runtime_options is None else dict(config.runtime_options)
            if config.shots is not None:
                estimator_opts["shots"] = config.shots
            self._estimator = IBMEstimator(mode=self.backend, options=estimator_opts)
        else:
            raise ValueError("BackendConfig.kind must be 'aer' or 'ibm'")


    def transpile(self, circuits: Union[QuantumCircuit, Sequence[QuantumCircuit]]) \
        -> Union[QuantumCircuit, Sequence[QuantumCircuit]]:
        tk = self.config.transpile_kwargs or {}
        
        # To avoid issues with large systems
        if self.config.kind == "aer":
            # Simulator: no backend
            return transpile(
            circuits,
            optimization_level=self.config.transpile_ol,
            basis_gates=['u', 'cx'],
            coupling_map=None,
            **tk,
        )
        
        # IBM hardware: device-aware transpilation
        return transpile(circuits, backend=self.backend,
                     optimization_level=self.config.transpile_ol, **tk)

    def sampler(self):
        return self._sampler

    def estimator(self):
        return self._estimator

    # -------- Run/retrieve methods for various tasks -----------------------------

    # Run sampler on one or more circuits
    def run_sampler(self, circuits: Sequence[QuantumCircuit], shots: Optional[int] = None):
        # Transpile circuits to backend
        circ_t = self.transpile(circuits)
        if shots is not None:
            return self._sampler.run(circ_t, shots=shots)
        return self._sampler.run(circ_t)

    # Run estimator on one or more (circuit, operator) pairs
    def run_vqe(self, ansatz: QuantumCircuit, hamiltonian: SparsePauliOp,
                optimizer, initial_point, callback=None):
        ansatz_t = self.transpile(ansatz)
        vqe = VQE(estimator=self._estimator, ansatz=ansatz_t,
                  optimizer=optimizer, initial_point=initial_point, callback=callback)
        return vqe.compute_minimum_eigenvalue(hamiltonian)

    # Retrieve a Sampler/Estimator job result by job id
    def retrieve_job(self, job_id: str):
        if self.service is None:
            raise RuntimeError("retrieve_job is only available for IBM backends.")
        job = self.service.job(job_id)
        return job.result()
