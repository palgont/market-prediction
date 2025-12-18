# hmm_gaussian.py
import numpy as np

def logsumexp(a, axis=None):
    a = np.asarray(a)
    amax = np.max(a, axis=axis, keepdims=True)
    out = amax + np.log(np.sum(np.exp(a - amax), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def log_norm_pdf_1d(y, mean, var):
    # y: (T,)
    # mean,var: scalar
    return -0.5 * (np.log(2*np.pi*var) + (y - mean)**2 / var)

class GaussianHMM:
    """
    HMM with K hidden states and 1D Gaussian emissions:
        z_t ~ Markov(pi, A)
        y_t | z_t=k ~ Normal(mu_k, var_k)
    Fit with EM (Baum-Welch).
    """
    def __init__(self, K=3, n_iter=50, seed=0, min_var=1e-6):
        self.K = int(K)
        self.n_iter = int(n_iter)
        self.seed = int(seed)
        self.min_var = float(min_var)

        self.pi = None   # (K,)
        self.A = None    # (K,K)
        self.mu = None   # (K,)
        self.var = None  # (K,)

    def _init_params(self, y):
        rng = np.random.default_rng(self.seed)
        K = self.K
        self.pi = rng.random(K); self.pi /= self.pi.sum()
        self.A  = rng.random((K, K)); self.A /= self.A.sum(axis=1, keepdims=True)

        # init mu around data
        self.mu = rng.normal(loc=np.mean(y), scale=np.std(y) + 1e-8, size=K)
        self.var = np.full(K, np.var(y) + 1e-6)

    def _e_step(self, y):
        T = y.shape[0]
        K = self.K

        log_pi = np.log(self.pi + 1e-12)
        log_A  = np.log(self.A  + 1e-12)

        # logB[t,k] = log p(y_t | z_t=k)
        logB = np.zeros((T, K))
        for k in range(K):
            logB[:, k] = log_norm_pdf_1d(y, self.mu[k], self.var[k])

        # Forward pass (log alpha)
        log_alpha = np.zeros((T, K))
        log_alpha[0] = log_pi + logB[0]
        for t in range(1, T):
            log_alpha[t] = logB[t] + logsumexp(log_alpha[t-1][:, None] + log_A, axis=0)

        # Backward pass (log beta)
        log_beta = np.zeros((T, K))
        log_beta[T-1] = 0.0
        for t in range(T-2, -1, -1):
            log_beta[t] = logsumexp(log_A + logB[t+1][None, :] + log_beta[t+1][None, :], axis=1)

        # Log-likelihood
        loglik = logsumexp(log_alpha[T-1], axis=0)

        # gamma[t,k] ∝ alpha[t,k]*beta[t,k]
        log_gamma = log_alpha + log_beta
        log_gamma = log_gamma - logsumexp(log_gamma, axis=1)[:, None]
        gamma = np.exp(log_gamma)  # (T,K)

        # xi[t,i,j] ∝ alpha[t,i] A[i,j] B[t+1,j] beta[t+1,j]
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            log_xi_t = (log_alpha[t][:, None] + log_A + logB[t+1][None, :] + log_beta[t+1][None, :])
            log_xi_t = log_xi_t - logsumexp(log_xi_t.reshape(-1), axis=0)
            xi[t] = np.exp(log_xi_t)

        return loglik, gamma, xi

    def _m_step(self, y, gamma, xi):
        T, K = gamma.shape

        # pi
        self.pi = gamma[0] + 1e-12
        self.pi /= self.pi.sum()

        # A
        A_num = xi.sum(axis=0) + 1e-12               # (K,K)
        A_den = gamma[:-1].sum(axis=0) + 1e-12       # (K,)
        self.A = A_num / A_den[:, None]

        # mu, var
        Nk = gamma.sum(axis=0) + 1e-12
        self.mu = (gamma * y[:, None]).sum(axis=0) / Nk
        self.var = (gamma * (y[:, None] - self.mu[None, :])**2).sum(axis=0) / Nk
        self.var = np.maximum(self.var, self.min_var)

    def fit(self, y, verbose=True):
        y = np.asarray(y, dtype=float).reshape(-1)
        if self.pi is None:
            self._init_params(y)

        prev = -np.inf
        for it in range(self.n_iter):
            loglik, gamma, xi = self._e_step(y)
            self._m_step(y, gamma, xi)

            if verbose and (it % 5 == 0 or it == self.n_iter - 1):
                print(f"iter {it:03d}  loglik={loglik:.3f}")

            # (optional) monotonicity check
            if loglik < prev - 1e-6 and verbose:
                print("  warning: log-likelihood decreased (can happen with numerical issues)")
            prev = loglik

        return self

    def viterbi(self, y):
        """Most likely state sequence (Viterbi decoding)."""
        y = np.asarray(y, dtype=float).reshape(-1)
        T = y.shape[0]
        K = self.K

        log_pi = np.log(self.pi + 1e-12)
        log_A  = np.log(self.A  + 1e-12)

        logB = np.zeros((T, K))
        for k in range(K):
            logB[:, k] = log_norm_pdf_1d(y, self.mu[k], self.var[k])

        dp = np.zeros((T, K))
        ptr = np.zeros((T, K), dtype=int)

        dp[0] = log_pi + logB[0]
        for t in range(1, T):
            scores = dp[t-1][:, None] + log_A
            ptr[t] = np.argmax(scores, axis=0)
            dp[t] = logB[t] + np.max(scores, axis=0)

        states = np.zeros(T, dtype=int)
        states[T-1] = np.argmax(dp[T-1])
        for t in range(T-2, -1, -1):
            states[t] = ptr[t+1, states[t+1]]
        return states
