# utils/optimize_fidelity.py

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from concurrent.futures import ProcessPoolExecutor, as_completed

from utils.fast_fidelity import _build_B_from_brick, dga_overlap_and_fidelity




# Instead of maximizing fidelity, we miniminze -log(fidelity)
def _log_fidelity_objective(thetas, Q, n_layers):
    N_f, L = Q.shape
    B = _build_B_from_brick(L=L, N_f=N_f, thetas=thetas, n_layers=n_layers)
    sign, lad = np.linalg.slogdet(Q @ B)
    
    if (sign == 0) or (not np.isfinite(lad)):
        return np.inf  # repel singular/ill-conditioned cases
    return float(-4.0 * lad)  # minimizing -log fidelity


def _optimize_one(init, Q, n_layers, maxiter):
    '''
    Minimizes -log(fidelity) using L-BFGS-B: https://en.wikipedia.org/wiki/Limited-memory_BFGS
    '''
    N_f, L = Q.shape
    n_params = n_layers * (L - 1)

    bounds = [(-np.pi, np.pi)] * n_params     # parameters are angles

    return minimize(_log_fidelity_objective, init, args=(Q, n_layers),
                    method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter})


def _make_starts_sobol(n_params, n_starts, seed):
    # Sobol in [0,1]^d → scale to [-pi,pi]
    # scramble for randomness/repeatability
    sampler = qmc.Sobol(d=n_params, scramble=True, seed=seed)
    x01 = sampler.random(n_starts)
    return qmc.scale(x01, l_bounds=[-np.pi]*n_params, u_bounds=[np.pi]*n_params)



def optimize_thetas_multistart(Q, n_layers, n_starts=64, maxiter=500, seed=0, n_jobs=None):
    ''''
    Minimizes -log(fidelity) using L-BFGS-B: https://en.wikipedia.org/wiki/Limited-memory_BFGS

    Runs optimization with many different initial positions, spread using sobol sequence

    Runs multi-threaded and at the end choose best point found
    '''

    N_f, L = Q.shape
    n_params = n_layers * (L - 1)

    # Sobol sequence in [0,1]^d: https://en.wikipedia.org/wiki/Sobol_sequence
    inits = qmc.Sobol(d=n_params, scramble=True, seed=seed).random(n_starts)
    # Scale to [-pi,pi]ˆd
    inits = qmc.scale(inits, l_bounds=[-np.pi]*n_params, u_bounds=[np.pi]*n_params)


    best = None

    # Create pool of worker processes
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        
        # Schedule one job per init
        jobs = [ex.submit(_optimize_one, init, Q, n_layers, maxiter) for init in inits]

        # Iterate over jobs as they complete
        for job in as_completed(jobs):
            res = job.result()

            # Find best result
            if best is None or res.fun < best.fun:
                best = res

    
    best_theta = np.asarray(best.x, dtype=float)
    best_fidelity = dga_overlap_and_fidelity(Q, best_theta, n_layers)[1]
    return best_theta, best_fidelity, best