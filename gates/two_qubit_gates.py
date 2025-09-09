# gates/two_qubit_gates.py
from qiskit import QuantumCircuit


def RXX(qc: QuantumCircuit, q1: int, q2: int, alpha: float):
    """
    Exp(i alpha * X1 * X2)
    """
    qc.cx(q1,q2)
    qc.rx(-2*alpha,q1)
    qc.cx(q1,q2)

def RYY(qc: QuantumCircuit, q1: int, q2: int, beta: float):
    """
    Exp(i beta * Y1 * Y2) = (S1 S2)^dagger RXX(beta) S1 S2
    """
    qc.s(q1)
    qc.s(q2)
    RXX(qc, q1, q2, beta)
    qc.sdg(q1)
    qc.sdg(q2)


def RZZ(qc: QuantumCircuit, q1: int, q2: int, gamma: float):
    """
    Exp(i gamma * Z1 * Z2)
    """
    qc.cx(q1,q2)
    qc.rz(-2*gamma,q2)
    qc.cx(q1,q2)