import numpy as np

def bme_fit(probs, basis_matrix, shots, nqubits):

    N = 2 ** nqubits
    SAMPLES = 400

    def logl(rho, meas_res):

        logl = 0.0
        for result in meas_res:
            logl += result[1] * np.log(np.real(np.trace(np.matmul(np.reshape(result[0], (N,N)), rho))))

        return logl

    def montecarlo(N, likelihood, nsamples):
        result = np.zeros((N, N), dtype='complex')
        error = np.kron(result, result)

        # Estimated density operator
        for i in range(nsamples):
            rho = ginibre(N)
            L = np.exp(likelihood(rho))
            result += L * rho
            error += L * np.kron(rho, rho)

        result /= nsamples
        trace = np.trace(result)
        result /= trace

        error /= nsamples
        error /= trace
        error -= np.kron(result, result)

        return result, error


    # print(basis_matrix)
    # print(probs)

    counts = [i * shots for i in probs]

    # print(counts)

    meas_res = list(zip(basis_matrix,counts))
    print('measurement results',meas_res)

    res = montecarlo(N, lambda rho: logl(rho, meas_res), SAMPLES)
    return res

def compute_std_error(obs, superoperator):
   return np.sqrt(np.trace(np.matmul(np.kron(obs,obs),superoperator))).real

def ginibre(N, rank=None):
   """
   Returns a Ginibre random density operator of dimension
   ``dim`` and rank ``rank`` by using the algorithm of
   [BCSZ08]_. If ``rank`` is `None`, a full-rank
   (Hilbert-Schmidt ensemble) random density operator will be
   returned.

   Parameters
   ----------
   N : int
       Dimension of the density operator to be returned.

   rank : int or None
       Rank of the sampled density operator. If None, a full-rank
       density operator is generated.

   Returns
   -------
   rho : Qobj
       An N × N density operator sampled from the Ginibre
       or Hilbert-Schmidt distribution.
   """
   if rank is None:
       rank = N
   if rank > N:
       raise ValueError("Rank cannot exceed dimension.")
   shape = (N, rank)
   X = np.sum(np.random.randn(*(shape + (2,)))* np.array([1, 1j]), axis=-1)
   rho = np.dot(X, X.T.conj())
   rho /= np.trace(rho)

   return rho
