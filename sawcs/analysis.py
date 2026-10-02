"""
Analytical results of Scale-Adaptive Wavelet Collision Scheduling (SA-WCS).

  Theorem 1  : closed-form Haar wavelet spectrum of the RRI-folded occupancy process
  Prop. 1    : moment estimator (lambda, sigma_M^2, sigma_W^2, p, pi, beta) from 3 scales
  Theorem 2  : renewal-reward collision intensity of SPS (BRC + MD + ENC terms)
  Theorem 3  : optimal reservation persistence (closed form b* and exact renewal argmin)
  Cor. 1/2   : regret (cosh law) and AoI of bursty losses
  Prop. 3    : distance-dependent PRR approximation
"""
import numpy as np
from scipy.stats import norm

SQ2 = np.sqrt(2.0)


# ----------------------------------------------------------------------------------
# Theorem 1: Haar wavelet variance of AR(1)-type occupancy + white flicker
# ----------------------------------------------------------------------------------
def h_closed(lam, L):
    """h_j(lambda) with L = 2^(j-1):
       (1+l)/(1-l) - l (1-l^L)(3-l^L) / (L (1-l)^2);  series (1-l)(2L^2+1)/3 as l->1."""
    lam = np.asarray(lam, dtype=float)
    L = np.asarray(L, dtype=float)
    small = (1.0 - lam) < 1e-4
    ls = np.where(small, 0.5, lam)
    val = (1 + ls) / (1 - ls) - ls * (1 - ls ** L) * (3 - ls ** L) / (L * (1 - ls) ** 2)
    ser = (1 - lam) * (2 * L ** 2 + 1) / 3.0
    return np.where(small, ser, val)


def h_direct(lam, L):
    """Direct evaluation from the autocovariance sums (verification of Theorem 1)."""
    L = int(L)
    t_in = np.arange(1, L)
    S = L + 2.0 * np.sum((L - t_in) * lam ** t_in)
    t_x = np.arange(1, 2 * L)
    C = np.sum((L - np.abs(t_x - L)) * lam ** t_x)
    return (S - C) / L


def wavelet_variance(j, lam, sM2, sW2):
    j = np.asarray(j, dtype=float)
    return sW2 + sM2 * h_closed(lam, 2.0 ** (j - 1))


def G_ratio(lam):
    """G(lambda) = (h3-h2)/(h2-h1): strictly increasing from 1/2 (lambda->0) to 4 (lambda->1)."""
    h1, h2, h3 = h_closed(lam, 1), h_closed(lam, 2), h_closed(lam, 4)
    return (h3 - h2) / (h2 - h1)


_LAM_GRID = np.linspace(1e-6, 0.9995, 40001)
_G_GRID = G_ratio(_LAM_GRID)


def invert_wavelet(nu1, nu2, nu3, mu, lam_max=0.995):
    """Proposition 1: invert the three-scale Haar spectrum.
    Returns dict with lam, sM2, sW2, p (flicker), pi (true occupancy), beta (1->0 hazard)."""
    nu1, nu2, nu3, mu = (np.asarray(v, dtype=float) for v in (nu1, nu2, nu3, mu))
    d1 = nu2 - nu1
    d2 = nu3 - nu2
    R = np.where(d1 > 1e-12, d2 / np.maximum(d1, 1e-12), 0.5)
    lam = np.clip(np.interp(R, _G_GRID, _LAM_GRID), 0.0, lam_max)
    lam = np.where(d1 > 1e-12, lam, 0.0)
    dh = h_closed(lam, 2) - h_closed(lam, 1)
    sM2 = np.where(dh > 1e-12, np.maximum(d1, 0.0) / np.maximum(dh, 1e-12), 0.0)
    sW2 = np.maximum(nu1 - sM2 * h_closed(lam, 1), 0.0)
    p = np.clip(sW2 / np.maximum(mu, 1e-9), 0.0, 0.5)
    pi = np.clip(mu / (1.0 - p), 1e-4, 0.99)
    beta = (1.0 - lam) * (1.0 - pi)
    return dict(lam=lam, sM2=sM2, sW2=sW2, p=p, pi=pi, beta=beta)


def invert_structural(nu1, nu2, mu, lam_max=0.995):
    """Proposition 1 (structural two-scale Haar estimator). For binary occupancy the Bernoulli
    identity gamma(0) = mu(1-mu) closes the system:  Q = (nu2-nu1)/(mu(1-mu)-nu1) = (1-l)(3+l)/2
    =>  lambda = sqrt(4-2Q) - 1,  sigma_M^2 = (mu(1-mu)-nu1)/lambda,  sigma_W^2 = mu(1-mu) - sigma_M^2."""
    nu1, nu2, mu = (np.asarray(v, dtype=float) for v in (nu1, nu2, mu))
    V = mu * (1.0 - mu)
    den = V - nu1
    Q = np.where(den > 1e-12, (nu2 - nu1) / np.maximum(den, 1e-12), 1.5)
    lam = np.clip(np.sqrt(np.clip(4.0 - 2.0 * Q, 1.0, 4.0)) - 1.0, 0.0, lam_max)
    sM2 = np.where(lam > 1e-6, np.maximum(den, 0.0) / np.maximum(lam, 1e-6), 0.0)
    sM2 = np.minimum(sM2, V)
    sW2 = np.maximum(V - sM2, 0.0)
    p = np.clip(sW2 / np.maximum(mu, 1e-9), 0.0, 0.5)
    pi = np.clip(mu / (1.0 - p), 1e-4, 0.99)
    beta = (1.0 - lam) * (1.0 - pi)
    return dict(lam=lam, sM2=sM2, sW2=sW2, p=p, pi=pi, beta=beta)


def gof_third_scale(nu3, lam, sM2, sW2):
    """Goodness of fit: predicted minus measured level-3 wavelet variance (Theorem 1)."""
    return wavelet_variance(3, lam, sM2, sW2) - np.asarray(nu3, dtype=float)


def flux_from_count_spectrum(nu1, nu2, nu3, lam_lo=0.6, lam_hi=0.995):
    """Corollary 1 (conserved inner-zone count): entry flux E = sigma_M^2 (1-lambda)
    = 2 (nu2 - nu1) / (lambda (3 + lambda)), lambda clipped to the physically admissible range."""
    nu1, nu2, nu3 = (np.asarray(v, dtype=float) for v in (nu1, nu2, nu3))
    d1, d2 = nu2 - nu1, nu3 - nu2
    R = np.where(d1 > 1e-12, d2 / np.maximum(d1, 1e-12), 4.0)
    lam = np.clip(np.interp(R, _G_GRID, _LAM_GRID), lam_lo, lam_hi)
    return np.maximum(2.0 * d1 / (lam * (3.0 + lam)), 0.0)


# ----------------------------------------------------------------------------------
# Reservation-lifetime renewal functions (RC ~ U{Cmin..Cmax}, keep probability pk)
# ----------------------------------------------------------------------------------
def lifetime_pmf(pk, cmin=5, cmax=15, nfft=1 << 14):
    """pmf of L = sum_{g=1}^{G} C_g, G ~ Geom(1-pk) on {1,2,...}: F_L = (1-pk)F_C/(1-pk F_C)."""
    fc = np.zeros(nfft)
    fc[cmin:cmax + 1] = 1.0 / (cmax - cmin + 1)
    FC = np.fft.rfft(fc)
    FL = (1 - pk) * FC / (1 - pk * FC)
    fl = np.clip(np.fft.irfft(FL, nfft), 0.0, None)
    return fl / fl.sum()


def renewal_moments(pk, cmin=5, cmax=15):
    """b = 1/E[L]; m1 = E[min(L1,L2)], m2 = E[min(L,R)], m3 = E[min(R1,R2)] (R: equilibrium residual)."""
    fl = lifetime_pmf(pk, cmin, cmax)
    k = np.arange(len(fl))
    EL = float(np.sum(k * fl))
    SL = (1.0 - np.cumsum(fl) + fl)[1:]            # P(L >= k), k >= 1
    SR = np.cumsum(SL[::-1])[::-1] / EL            # P(R >= k)
    return 1.0 / EL, float(np.sum(SL ** 2)), float(np.sum(SL * SR)), float(np.sum(SR ** 2))


def renewal_moments2(pk, cmin=5, cmax=15):
    """Second moments E[min(L1,L2)^2], E[min(L,R)^2], E[min(R1,R2)^2] (Theorem 4)."""
    fl = lifetime_pmf(pk, cmin, cmax)
    k = np.arange(len(fl))
    EL = float(np.sum(k * fl))
    SL = (1.0 - np.cumsum(fl) + fl)[1:]
    SR = np.cumsum(SL[::-1])[::-1] / EL
    w = 2.0 * np.arange(1, len(SL) + 1) - 1.0          # E[X^2] = sum_k (2k-1) P(X >= k)
    return float(np.sum(w * SL ** 2)), float(np.sum(w * SL * SR)), float(np.sum(w * SR ** 2))


def geometric_moments(b):
    m = 1.0 / (b * (2.0 - b))
    return b, m, m, m


PK_GRID = np.round(np.arange(0.0, 0.8001, 0.05), 2)            # standard range [0, 0.8]
_TAB = np.array([renewal_moments(p) for p in PK_GRID])        # columns: b, m1, m2, m3
TAB_B, TAB_M1, TAB_M2, TAB_M3 = _TAB.T
_TAB2 = np.array([renewal_moments2(p) for p in PK_GRID])
TAB_Q1, TAB_Q2, TAB_Q3 = _TAB2.T


# ----------------------------------------------------------------------------------
# Theorem 2: collision intensity  Lambda = Lambda_BRC + Lambda_MD + Lambda_ENC
# ----------------------------------------------------------------------------------
def collision_intensity(pk, N_I, xA, Nr, px, beta_mob, omega=1.0, zeta=1.0,
                        cmin=5, cmax=15, memoryless=False):
    if memoryless:
        b, m1, m2, m3 = geometric_moments((1 - pk) / (0.5 * (cmin + cmax)))
    else:
        b, m1, m2, m3 = renewal_moments(pk, cmin, cmax)
    SA = xA * Nr
    l_brc = N_I * omega * b * b * m1 / SA
    l_md = N_I * 2.0 * b * px * m2 / SA
    l_enc = zeta * N_I * beta_mob * m3 / Nr
    lam = l_brc + l_md + l_enc
    return 1.0 - np.exp(-lam), dict(BRC=l_brc, MD=l_md, ENC=l_enc, total=lam)


def collision_age(pk, N_I, xA, Nr, px, beta_mob, omega=1.0, zeta=1.0, cmin=5, cmax=15):
    """Theorem 4: Gamma = sum_type (formation rate) x E[D_type^2]; AoI/T ~ c0 + (phi/2)(Lambda + Gamma)."""
    b = renewal_moments(pk, cmin, cmax)[0]
    q1, q2, q3 = renewal_moments2(pk, cmin, cmax)
    SA = xA * Nr
    return N_I * omega * b * b * q1 / SA + N_I * 2.0 * b * px * q2 / SA + zeta * N_I * beta_mob * q3 / Nr


def b_star_aoi(beta_mob, px, xA, eta, omega=1.0, zeta=1.0):
    """Memoryless AoI-aware law: unique positive root of A b^3 - (B + eta C1) b - 2 eta B = 0
    (normalised by N_I/N_r: A = omega/(2 xA), B = zeta beta_mob/2, C1 = px/xA)."""
    A, B, C1 = omega / (2 * xA), zeta * beta_mob / 2, px / xA
    r = np.roots([A, 0.0, -(B + eta * C1), -2 * eta * B])
    r = r[np.isreal(r)].real
    return float(r[r > 0].max())


# ----------------------------------------------------------------------------------
# Theorem 3: optimal persistence
# ----------------------------------------------------------------------------------
def b_star_closed(beta_mob, xA, omega=1.0, zeta=1.0):
    """Memoryless closed form: b* = sqrt(zeta * beta_mob * xA / omega)."""
    return np.sqrt(zeta * np.asarray(beta_mob) * np.asarray(xA) / omega)


def pk_from_b(b, EC=10.0):
    return np.clip(1.0 - np.asarray(b) * EC, 0.0, 0.8)


def objective_grid(beta_mob, px, xA, omega=None, zeta=1.0, eta=0.0):
    """Persistence-dependent collision intensity (normalised by N_I/N_r) on PK_GRID."""
    beta_mob = np.atleast_1d(np.asarray(beta_mob, float))
    px = np.atleast_1d(np.asarray(px, float))
    xA = np.atleast_1d(np.asarray(xA, float))
    if omega is None:
        omega = 1.0 - (1.0 - xA) / 4.0
    omega = np.atleast_1d(np.asarray(omega, float)) * np.ones_like(xA)
    F = (omega[:, None] * TAB_B[None, :] ** 2 * TAB_M1[None, :]
         + 2.0 * px[:, None] * TAB_B[None, :] * TAB_M2[None, :]) / xA[:, None] \
        + zeta * beta_mob[:, None] * TAB_M3[None, :]
    if eta > 0:
        F = F + eta * ((omega[:, None] * TAB_B[None, :] ** 2 * TAB_Q1[None, :]
                        + 2.0 * px[:, None] * TAB_B[None, :] * TAB_Q2[None, :]) / xA[:, None]
                       + zeta * beta_mob[:, None] * TAB_Q3[None, :])
    return F


def pk_star(beta_mob, px, xA, omega=None, zeta=1.0, eta=0.0):
    F = objective_grid(beta_mob, px, xA, omega, zeta, eta)
    return PK_GRID[np.argmin(F, axis=1)]


def regret_ratio(kappa):
    """Corollary 1: persistence-dependent cost inflation when beta_mob is mis-estimated by kappa."""
    k = np.asarray(kappa, dtype=float)
    return 0.5 * (np.sqrt(k) + 1.0 / np.sqrt(k))


# ----------------------------------------------------------------------------------
# Corollary 2: AoI of bursty losses (renewal formula)
# ----------------------------------------------------------------------------------
def aoi_from_gaps(gap_hist, T=0.1):
    """Mean AoI (excluding a constant access delay) from an inter-reception gap histogram."""
    g = np.arange(len(gap_hist), dtype=float)
    n = gap_hist.sum()
    if n == 0:
        return np.nan
    E1 = (g * gap_hist).sum() / n
    E2 = (g * g * gap_hist).sum() / n
    return T * E2 / (2.0 * E1)


def aoi_gilbert(eps, Pc, q, T=0.1):
    """Delta/T = (1+eps)/(2(1-eps)) + Pc/q  (i.i.d. losses eps + collision episodes, end prob q)."""
    return T * ((1 + eps) / (2 * (1 - eps)) + Pc / q)


# ----------------------------------------------------------------------------------
# Proposition 3: distance-dependent PRR approximation
# ----------------------------------------------------------------------------------
def pathloss_db(d, fc_GHz=5.9, d_bp=100.0, n_exp=3.5):
    d = np.maximum(np.asarray(d, dtype=float), 1.0)
    fs = 32.45 + 20 * np.log10(fc_GHz)
    return np.where(d <= d_bp, fs + 20 * np.log10(d),
                    fs + 20 * np.log10(d_bp) + 10 * n_exp * np.log10(d / d_bp))


def prr_analytic(d, P_RC, cfg, Rs):
    d = np.atleast_1d(np.asarray(d, dtype=float))
    sig = np.sqrt(cfg.sigma_S ** 2 + cfg.sigma_F ** 2)
    N0dB = -174 + 10 * np.log10(cfg.B_Hz) + cfg.NF
    pl = lambda z: pathloss_db(z, cfg.fc_GHz, cfg.d_bp, cfg.n_exp)
    snr = cfg.P_tx - pl(d) - N0dB
    P0 = norm.sf((cfg.gamma_data - snr) / sig)
    rho = cfg.density / 1000.0
    Nr = cfg.N_sub * cfg.N_slot
    xin = np.linspace(-Rs, Rs, 801)
    xo = np.linspace(Rs, cfg.L_road / 2, 1201)
    out = np.empty_like(d)
    for i, di in enumerate(d):
        ff = lambda xs: norm.cdf((cfg.gamma_data - (pl(np.maximum(np.abs(xs - di), 3.0)) - pl(di)))
                                 / (SQ2 * sig))
        F_in = ff(xin).mean()
        integ = np.trapezoid(ff(xo), xo) + np.trapezoid(ff(-xo), xo)
        hidden = np.exp(-rho / Nr * integ)
        out[i] = (1 - 1.0 / cfg.N_slot) * P0[i] * (1 - P_RC * F_in) * hidden
    return out
