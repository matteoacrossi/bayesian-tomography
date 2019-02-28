import qiskit
import qiskit.ignis
import numpy
from bme_fit import *
import time
import qiskit
from qiskit import QuantumRegister, QuantumCircuit, ClassicalRegister, BasicAer
from qiskit.quantum_info import state_fidelity
import qiskit.ignis.verification.tomography as tomo

#number of trials
SHOTS = 10

#number of qubits
n = 2

#create a qubit register
q_reg = QuantumRegister(n)
c_reg = ClassicalRegister(n)

#create a quantum circuit
circuit = QuantumCircuit(q_reg)


#append gates to the quantum circuit
circuit.h(q_reg[0])

print(circuit.qasm())

#run the circuit
job = qiskit.execute(circuit, BasicAer.get_backend('statevector_simulator'))

#get wave function at the end of the circuit
psi = job.result().get_statevector(circuit)
print('final state of the circuit',psi)


# Generate circuits for tomography and run tomography on simulator
t = time.time()
qst = tomo.state_tomography_circuits(circuit, q_reg)
job = qiskit.execute(qst, BasicAer.get_backend('qasm_simulator'), shots=SHOTS)
print('Time taken:', time.time() - t)

# Extract tomography data so that counts are indexed by measurement configuration
tomo_counts = tomo.tomography_data(job.result(), qst)

# Generate fitter data and reconstruct density matrix
probs, basis_matrix, weights = tomo.fitter_data(tomo_counts)
print('tomography results',tomo_counts)

#bayesian reconstruction of the density matrix and error on observable
rho_fit, error_rho = bme_fit(probs, basis_matrix, SHOTS, n)

#maximum likelihood reconstruction of density matrix
rho_mle = tomo.state_mle_fit(probs, basis_matrix, weights)


print("MLE:", rho_mle)
print("BME", rho_fit)
