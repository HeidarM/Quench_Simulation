# utils/VQE_convergence.py

import numpy as np
import matplotlib.pyplot as plt
from collections import deque

from utils.fast_fidelity import dga_overlap_and_fidelity


# --- Early termination criterion for the SPSA  ---
# Implement as functor instead of function, to keep track of history
class SPSAConvergence:
    """Stop when the max |ΔE| over a trailing window falls below tolerance."""
    def __init__(self, n_layers: int, Q, fidelity_goal: float = 0.9):


        self.n_layers = n_layers
        self.Q = Q
        self.fidelity_goal = fidelity_goal

        self.fidelities = []     # (iteration, fidelity) log

    # NOTE: For SPSA.termination_checker the arg order is:
    # (nfev, parameters, value, stepsize, accepted)
    # # See https://qiskit-community.github.io/qiskit-algorithms/stubs/qiskit_algorithms.optimizers.SPSA.html
    def __call__(self, nfev, parameters, value, stepsize, accepted) -> bool:
        
        _, fidelity = dga_overlap_and_fidelity(self.Q, parameters, self.n_layers)
        self.fidelities.append((nfev, fidelity))

        if fidelity >= self.fidelity_goal:
            return True     # Terminate optimization
        else:
            return False    # Continue optimization




# Callback to keep track of lowest energy observed during optimization and the corresponding parameters - optimizer results last point not the best found
class KeepBestCallback:
    def __init__(self):
        self.min_energy = np.inf
        self.min_energy_parameter = None

    def __call__(self, nfev, x, fx, stepsize, accepted):
        if fx < self.min_energy:
            self.min_energy = float(fx)
            self.min_energy_parameter = np.array(x, copy=True)
            print(f"\nNew min energy: {self.min_energy:.6f} at iter {nfev}")
            print(f"New min parameters: {self.min_energy_parameter}")


# Fuctor for monitoring VQE convergence with live plotting and storing best parameters reached so far
# See https://qiskit-community.github.io/qiskit-algorithms/tutorials/02_vqe_advanced_options.html
class VQETrackProgressCallback:
    def __init__(self, history_window: int,
                 n_layers: int | None = None, Q: np.ndarray | None = None,
                 live_plot: bool = False,
                 print_best_energy: bool = False,
                 print_best_parameter: bool = False,
                 print_best_fidelity: bool = False):

        self.live_plot = live_plot
        self.print_best_energy = print_best_energy
        self.print_best_parameter = print_best_parameter
        self.print_best_fidelity = print_best_fidelity
        self.history_window = history_window

        # For tracking best parameters so far
        self.min_energy = np.inf
        self.min_energy_parameter = None

        # fidelity config + logs
        self.Q = Q
        self.n_layers = n_layers
        self.fidelities = []
        self.best_fidelity = -np.inf

        if self.live_plot:
            self.history = deque(maxlen=history_window + 1)      # store last window+1 energies

            self.max_deltas = deque(maxlen=2000)                 # track ΔE series for plotting
            self.values = deque(maxlen=2000)                     # track current value series
            self.iters = deque(maxlen=2000)                      # track iteration numbers

            self.fid_vals = deque(maxlen=2000)                   # track fidelity series (if Q provided)


            # -------- Set up plot --------
            # Set up figure: add a third axis for fidelity if we can compute it
            if self.Q is not None and self.n_layers is not None:
                ncols = 3
            else:
                ncols = 2
            
            plt.ion()
            self.fig, axes = plt.subplots(1, ncols, figsize=(15 if ncols==3 else 10, 4))
            if ncols == 2:
                self.ax, self.ax_val = axes
                self.ax_fid = None
            else:
                self.ax, self.ax_val, self.ax_fid = axes

            (self.line,) = self.ax.plot([], [], "-o")
            (self.line_val,) = self.ax_val.plot([], [], "-o")

            self.ax.set_xlabel("Number of Evaluations")
            self.ax.set_ylabel("max |ΔE|")
            self.ax.set_yscale("log")
            self.ax.grid(True)

            self.ax_val.set_xlabel("Number of Evaluations")
            self.ax_val.set_ylabel("E")
            self.ax_val.grid(True)

            if self.ax_fid is not None:
                (self.line_fid,) = self.ax_fid.plot([], [], "-o")
                self.ax_fid.set_xlabel("Number of Evaluations")
                self.ax_fid.set_ylabel("Fidelity")
                self.ax_fid.set_ylim(0.0, 1.0)
                self.ax_fid.grid(True)



    # Callback function for VQE - live plotting
    def __call__(self, eval_count: int, parameters, energy, metadata):


        current_energy = float(energy)

        if current_energy < self.min_energy:
            self.min_energy = current_energy
            self.min_energy_parameter = np.array(parameters, copy=True)

            if self.print_best_energy:
                print(f"New min energy: {self.min_energy:.6f} at iter {eval_count}")
            if self.print_best_parameter:
                print(f"New min parameters: {self.min_energy_parameter}\n")

        # Fidelity testing
        if self.Q is not None and self.n_layers is not None:
            _, fidelity = dga_overlap_and_fidelity(self.Q, parameters, self.n_layers)
            self.fidelities.append((eval_count, float(fidelity)))

            if fidelity > self.best_fidelity:
                self.best_fidelity = float(fidelity)
                self.best_fidelity_at = int(eval_count)
                self.best_fidelity_params = np.array(parameters, copy=True)
                if self.print_best_fidelity:
                    print(f"New best fidelity: {self.best_fidelity:.6f} at eval {self.best_fidelity_at}")


        # Plotting
        if self.live_plot:
            self.history.append(current_energy)

            # update ΔE window metric
            if len(self.history) == self.history.maxlen:
                # compute max absolute difference across the window
                diffs = [abs(self.history[i+1] - self.history[i])
                        for i in range(len(self.history)-1)]
                max_delta = float(np.max(diffs))
                self.max_deltas.append(max_delta)
                self.line.set_data(range(len(self.max_deltas)), self.max_deltas)
                self.ax.relim(); self.ax.autoscale_view()

            # update raw energy trace
            self.values.append(current_energy); 
            self.iters.append(eval_count)
            self.line_val.set_data(self.iters, self.values)
            self.ax_val.relim(); self.ax_val.autoscale_view()

            # update fidelity plot
            if self.Q is not None and self.n_layers is not None and hasattr(self, "ax_fid") and self.ax_fid is not None:
                self.fid_vals.append(self.fidelities[-1][1] if len(self.fidelities) else np.nan)
                self.line_fid.set_data(self.iters, self.fid_vals)
                self.ax_fid.relim(); self.ax_fid.autoscale_view()

            plt.pause(0.00000001)



def cosine_anneal_with_restarts(
    max_iter,
    lr_max, lr_min,
    eps_max, eps_min,
    T0=100, T_mult=1.5
):
    """
    Returns arrays eta (learning_rate) and eps (perturbation) of length max_iter
    using cosine annealing with warm restarts. Period lengths grow by T_mult.
    """
    eta = np.zeros(max_iter, dtype=float)
    eps = np.zeros(max_iter, dtype=float)
    t = 0
    Ti = int(T0)
    while t < max_iter:
        for i in range(Ti):
            if t >= max_iter:
                break
            cosw = 0.5 * (1 + np.cos(np.pi * i / Ti))
            eta[t] = lr_min + (lr_max - lr_min) * cosw
            eps[t] = eps_min + (eps_max - eps_min) * cosw
            t += 1
        Ti = int(np.ceil(Ti * T_mult))  # lengthen cycles
    return eta, eps


def powerlaw_with_restarts(
    max_iter: int,
    A: float, a: float, alpha: float,
    c: float, gamma: float,
    T0: int = 150, T_mult: float = 1.3,
):
    """
    Piecewise power-law schedules with warm restarts.
    Each cycle resets i->0, so early-iteration 'large steps' reappear.
    """
    eta  = np.zeros(max_iter, dtype=float)
    eps  = np.zeros(max_iter, dtype=float)
    t    = 0
    Ti   = int(T0)

    while t < max_iter:
        # indices within this cycle
        i_cycle = np.arange(Ti, dtype=float)
        # clip last cycle if max_iter not multiple of Ti
        i_cycle = i_cycle[:max_iter - t]

        # power-law (your original) but reset each cycle
        eta[t:t+len(i_cycle)] = a / ((A + i_cycle + 1.0) ** alpha)
        eps[t:t+len(i_cycle)] = c / ((i_cycle + 1.0) ** gamma)

        t  += len(i_cycle)
        Ti  = int(np.ceil(Ti * T_mult))  # lengthen cycles (optional)

    return eta, eps