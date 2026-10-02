"""
Synthetic RRI-folded occupancy processes and churn estimators.

Occupancy model (Assumptions A1-A2 of the letter):
  s_r(m): stationary two-state Markov chain, P(0->1)=alpha, P(1->0)=beta,
          pi = alpha/(alpha+beta), lambda = 1-alpha-beta
  y_r(m) = s_r(m) * (1 - e_r(m)),  e ~ i.i.d. Bernoulli(p)   (missed detections / flicker)
"""
import numpy as np
from . import analysis as an

SQ2 = np.sqrt(2.0)


def gen_occupancy(n_res, n_t, pi, lam, p_md, rng, pi_t=None):
    """Return y (n_res x n_t) and hidden s. If pi_t is given (length n_t) the chain is
    non-stationary with alpha_t = (1-lam) pi_t, beta_t = (1-lam)(1-pi_t)."""
    s = np.empty((n_res, n_t), dtype=bool)
    p0 = pi if pi_t is None else pi_t[0]
    cur = rng.random(n_res) < p0
    for t in range(n_t):
        pit = pi if pi_t is None else pi_t[t]
        a, b = (1 - lam) * pit, (1 - lam) * (1 - pit)
        u = rng.random(n_res)
        cur = np.where(cur, u >= b, u < a)
        s[:, t] = cur
    y = s & (rng.random((n_res, n_t)) >= p_md)
    return y.astype(float), s


def haar_details(y, J=3):
    """Undecimated (MODWT-type, orthonormal DWT scaling) Haar details, all shifts with full support.
    y: (..., n_t). Returns list of arrays d_j with shape (..., n_t - 2^j + 1)."""
    out = []
    c = np.cumsum(np.concatenate([np.zeros(y.shape[:-1] + (1,)), y], axis=-1), axis=-1)
    n = y.shape[-1]
    for j in range(1, J + 1):
        Lj = 2 ** (j - 1)
        w = 2 * Lj
        m = np.arange(w - 1, n)                      # window end index
        new = c[..., m + 1] - c[..., m + 1 - Lj]     # most recent L_j samples
        old = c[..., m + 1 - Lj] - c[..., m + 1 - w]
        out.append((new - old) / np.sqrt(w))
    return out


def est_wavelet(y, J=3):
    d = haar_details(y, J)
    nu = [np.mean(dj ** 2) for dj in d]
    e = an.invert_wavelet(nu[0], nu[1], nu[2], y.mean())
    return float(e['beta']), e, nu


def est_wavelet_struct(y):
    d = haar_details(y, 2)
    nu = [np.mean(dj ** 2) for dj in d]
    e = an.invert_structural(nu[0], nu[1], y.mean())
    return float(e['beta']), e, nu


def est_tc(y):
    """Transition-count (Markov MLE) estimator; limit beta + p(1-beta)."""
    prev, now = y[:, :-1], y[:, 1:]
    n1 = prev.sum()
    return float(((prev == 1) & (now == 0)).sum() / max(n1, 1))


def _acov(y, lag, mean=None):
    mu = y.mean() if mean is None else mean
    z = y - mu
    if lag == 0:
        return np.mean(z * z)
    return np.mean(z[:, :-lag] * z[:, lag:])


def est_acf1(y):
    """Lag-1 autocorrelation with global mean (no flicker correction)."""
    lam = np.clip(_acov(y, 1) / max(_acov(y, 0), 1e-12), 0, 0.999)
    return float((1 - lam) * (1 - y.mean()))


def est_lagratio(y):
    """ARMA(1,1) moment estimator: lambda = gamma(2)/gamma(1), flicker-corrected via gamma(0)."""
    g0, g1, g2 = _acov(y, 0), _acov(y, 1), _acov(y, 2)
    lam = np.clip(g2 / g1, 0.0, 0.995) if g1 > 1e-12 else 0.0
    sM2 = g1 / lam if lam > 1e-9 else 0.0
    sW2 = max(g0 - sM2, 0.0)
    mu = y.mean()
    p = np.clip(sW2 / max(mu, 1e-9), 0, 0.5)
    pi = np.clip(mu / (1 - p), 1e-4, 0.99)
    return float((1 - lam) * (1 - pi))
