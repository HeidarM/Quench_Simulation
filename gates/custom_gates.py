# gates/custom_gates.py

import numpy as np
from numbers import Real

from qiskit import QuantumCircuit
from qiskit.circuit import Parameter,ParameterExpression, Gate
from qiskit.circuit.tools.pi_check import pi_check
from qiskit.circuit.library import SwapGate, CPhaseGate, CHGate

from .two_qubit_gates import RXX, RYY, RZZ

# -------------- Helper functions --------------
# Check if x is a parameter or parameter expression
def _is_param(x):
    return isinstance(x, (ParameterExpression, Parameter))

# Pretty print for angles in gate labels if numerical
def _fmt_angle_for_label(x, symbol='θ'):
    # Pretty label: π-fraction if numeric, else the parameter's repr
    if _is_param(x):
        return f"{x}"
    if isinstance(x, Real):
        return pi_check(float(x), ndigits=3, output='text')
    return str(x)

def _is_zero(x, tol=1e-12):
    return isinstance(x, Real) and abs(float(x)) < tol
# ----------------------------------------------


def N(qc: QuantumCircuit, q1: int, q2: int, alpha: float, beta: float, gamma: float):
    """
    N(alpha, 0, gamma) on qubits (q1, q2) as defined in paper
    """
    if alpha != 0:
        RXX(qc, q1, q2, alpha)
    if beta != 0:
        RYY(qc, q1, q2, beta)
    if gamma != 0:
        RZZ(qc, q1, q2, gamma)


def G_real(theta_val: float) -> Gate:
    """
        Implement the Givens rotation G(theta) on qubits (q1, q2).

        The generator in the paper appears to be wrong, this is the correct version:
            G(θ) = e^{iπ/4 Z_2} N(θ/2, θ/2, 0) e^{-iπ/4 Z_2}.

                 = [[1,    0,         0,       0],
                    [0,    cos(θ), -sin(θ),    0],
                    [0,    sin(θ),  cos(θ),    0],
                    [0,    0,         0,       1]].
    """

    # Try to get symbolic π representation
    # symbolic_label = pi_check(theta_val, ndigits=3, output='text')
    symbolic_label = _fmt_angle_for_label(theta_val, 'θ')

    theta = Parameter("θ")
    qc = QuantumCircuit(2)

    # Define the circuit
    qc.rz(-np.pi/2, 1)                          # e^{iπ/4 Z_2} = rz(-pi/2)
    #TODO: should this be N(qc, 0, 1, -theta/2, -theta/2, 0)? Double check
    N(qc, 0, 1, -theta/2, -theta/2, 0)
    qc.rz(np.pi/2, 1)                           # e^{-iπ/4 Z_2} = = rz(pi/2)

    # Assign the numeric value but retain the symbolic display
    bound_qc = qc.assign_parameters({theta: theta_val})
    return bound_qc.to_gate(label=f"G({symbolic_label})")

def G(theta_val: float, phi_val: float) -> Gate:
    if _is_param(phi_val):
        raise ValueError("Only real-valued rotations supported: phi must be numeric ≈ 0.")
    
    if np.isclose(phi_val, 0):
        return G_real(theta_val)
    else:
        raise ValueError("Only real-valued rotation gates (phi=0) are supported, but phi =", phi_val, "was given.")


def FSWAP() -> SwapGate:
    """
        Fermionic qubit swap
    """

    # Define FSWAP logic as a decomposition
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 0)
    qc.h(1)

    # Copy the standard SwapGate to keep it's symbol
    fswap = SwapGate().to_mutable()

    # Override the SWAP definition to get FSWAP
    fswap.definition = qc

    return fswap

def Q(theta_val: float) -> Gate:
    """
        Two–qubit local-quench gate
            Q(θ) = exp[ -i θ/4 (XX + YY) ]
        with θ = π/4 in the paper.
    """
    # pretty label for the drawer
    # symbolic_label = pi_check(theta_val, ndigits=3, output='text')
    symbolic_label = _fmt_angle_for_label(theta_val, 'θ')

    # theta = Parameter("θ")
    theta = theta_val if _is_param(theta_val) else Parameter("θ")
    qc = QuantumCircuit(2, name=f"Q({symbolic_label})")

    # Q = e^{-i θ/4 X⊗X} · e^{-i θ/4 Y⊗Y}
    N(qc, 0, 1, -theta/4, -theta/4, 0)

    return qc.assign_parameters({theta: theta_val}).to_gate(label=f"Q")




def H(theta_val: float) -> Gate:
    """
        Two–qubit  hopping  gate  H(θ) := exp[-i θ/2 (XX + YY)].
    """
    # pretty π label for the drawer
    # symbolic_label = pi_check(theta_val, ndigits=3, output='text')
    symbolic_label = _fmt_angle_for_label(theta_val, 'θ')

    # theta = Parameter("θ")                        # symbolic parameter
    theta = theta_val if _is_param(theta_val) else Parameter("θ") # symbolic parameter
    qc  = QuantumCircuit(2)
 
    N(qc, 0, 1, -theta/2, -theta/2, 0)

    # return qc.assign_parameters({theta: theta_val}).to_gate(label=f"H({symbolic_label})")
    if _is_param(theta_val):
        return qc.to_gate(label=f"H({symbolic_label})")
    else:
        return qc.assign_parameters({theta: float(theta_val)}).to_gate(label=f"H({symbolic_label})")



def O(phi_val: float) -> Gate:
    """
        O(φ) = exp[-i φ n↑ n↓]  implemented with the library CPhaseGate(-φ).
    """
    # symbolic_label = pi_check(phi_val, ndigits=3, output='text')
    symbolic_label = _fmt_angle_for_label(phi_val, 'φ')
    qc  = QuantumCircuit(2, name=f"O({symbolic_label})")
    qc.append(CPhaseGate(-phi_val), [0, 1])
    return qc.to_gate()



def BXY() -> Gate:
    """
        Local basis-change block used in the paper to diagonalise
        S^x = (X_up X_down + Y_up Y_down)/2
        before measuring both qubits in Z.

        q0 = ↑, q1 = ↓ ordering assumed
    """
    qc = QuantumCircuit(2, name="BXY")
    qc.cx(0, 1)               
    qc.append(CHGate(), [1, 0])   
    qc.cx(0, 1)               
    return qc.to_gate()