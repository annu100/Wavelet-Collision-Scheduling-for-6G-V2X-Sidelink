"""
SL-Mode2Sim: slot-level NR-V2X sidelink Mode 2 system-level simulator.

3GPP procedures
  * TS 38.214 Sec. 8.1.4 sensing-based selection: candidate single-slot resources of L_subCH contiguous
    subchannels in the selection window [n+T1, n+T2]; exclusion of candidates in non-monitored slots
    (half-duplex) for the configured RRI list; exclusion of candidates overlapping reservations announced
    in SCI with RSRP above the threshold (100-ms RRI, Q=1); +3 dB threshold steps until X*M_total
    candidates remain; uniform selection (the risk ranking of SA-WCS refines this last step).
  * TS 38.321 SPS: SL_RESOURCE_RESELECTION_COUNTER ~ U{5..15} (100-ms RRI), sl-ProbResourceKeep,
    re-evaluation T3 slots before a newly selected, not yet announced resource.
  * Dynamic scheduling: per-packet sensing-based selection without reservation announcements.
Evaluation assumptions (TR 37.885, highway)
  * 3 lanes per direction, 4-m lanes, wrap-around, inter-vehicle distance 2.5 s x speed (or given density)
  * 23 dBm, 3-dBi antennas, 9-dB noise figure, 5.9 GHz, 15-kHz SCS (1-ms slots), 10 subchannels x 10 PRBs
  * V2V highway LOS path loss 32.4 + 20 log10(d) + 20 log10(fc), highway LOS probability, NLOSv vehicle
    blockage loss (mean 5 + max(0, 15 log10 d - 41) dB, std 4 dB), 3-dB shadowing with 25-m decorrelation
PHY abstraction
  * block Rician (K = 9 dB, LOS) / Rayleigh (NLOSv) fading per subchannel and transmission
  * MIESM (QPSK) effective SINR and AWGN BLER curves for PSSCH (300-byte TB on 2 subchannels) and PSCCH
  * in-band emission per the TS 38.101-1 general IBE formula (EVM 17.5 %)
Large-scale parameters (positions, path loss, shadowing, LOS state) are updated every 100 ms.
"""
import warnings
from dataclasses import dataclass, asdict
import numpy as np
from . import analysis as an

warnings.filterwarnings("ignore", category=RuntimeWarning)
SQ2 = np.sqrt(2.0)

SCHEMES = {
    'DS': dict(ds=True), 'SPS': dict(pk=0.0), 'SPS-P8': dict(pk=0.8), 'SPS-LTE': dict(pk=0.0, rssi=True),
    'SPS-TC': dict(adapt='tc'), 'SA-WCS-P': dict(adapt='wav'), 'SA-WCS-S': dict(pk=0.0, wiener=True),
    'SA-WCS': dict(adapt='wav', wiener=True), 'SA-WCS-A': dict(adapt='wav', wiener=True, eta=0.01),
    'Oracle': dict(adapt='oracle'),
}


@dataclass
class SLConfig:
    L_road: float = 5000.0
    lanes_per_dir: int = 3
    lane_w: float = 4.0
    v_kmh: float = 140.0
    v_spread: float = 0.1
    spacing_s: float = 2.5
    density: float = 0.0          # veh/km; 0 -> TR 37.885 rule (2.5 s x speed per lane)
    fc_GHz: float = 5.9
    P_tx: float = 23.0
    G_ant: float = 3.0
    NF: float = 9.0
    scs_khz: float = 15.0
    n_subch: int = 10
    subch_prb: int = 10
    L_sub: int = 2
    rri: int = 100
    T1: int = 1
    T2: int = 100
    Th0: float = -110.0           # dBm per RE
    X_min: float = 0.2
    C_min: int = 5
    C_max: int = 15
    sigma_S: float = 3.0
    d_corr: float = 25.0
    K_los_dB: float = 9.0
    pssch_g50: float = 1.0
    pssch_slope: float = 0.35
    pscch_g50: float = -3.0
    pscch_slope: float = 0.5
    ibe: bool = True
    evm: float = 0.175
    reeval: bool = True
    T3: int = 3
    M: int = 8
    alpha_ew: float = 0.005
    dth_in: float = 21.0
    D_pir: float = 300.0
    d_bin: float = 50.0
    d_max: float = 1000.0
    gap_max: int = 400
    n_sec: float = 60.0
    warm_sec: float = 20.0
    adapt_sec: float = 6.0
    speed_profile: tuple = ()     # ((t_sec, v_kmh), ...)


# ----------------------------------------------------------------------------------- PHY abstraction
_H1, _H2, _H3 = 0.3073, 0.8935, 1.1064


def J(s):
    return (1.0 - 2.0 ** (-_H1 * s ** (2 * _H2))) ** _H3


def Jinv(I):
    I = np.clip(I, 1e-9, 1 - 1e-9)
    return (-np.log2(1.0 - I ** (1.0 / _H3)) / _H1) ** (1.0 / (2 * _H2))


def miesm_qpsk(sinr, axis=-1):
    """Effective SINR (linear) by mutual-information averaging of QPSK bit channels."""
    return Jinv(J(np.sqrt(4.0 * np.maximum(sinr, 0.0))).mean(axis=axis)) ** 2 / 4.0


def p_success(snr_db, g50, slope):
    return 1.0 / (1.0 + np.exp(-(snr_db - g50) / slope))


def pl_los_highway(d, fc):
    return 32.4 + 20 * np.log10(np.maximum(d, 1.0)) + 20 * np.log10(fc)


def p_los_highway(d):
    return np.where(d <= 475, np.minimum(1.0, 2.1013e-6 * d * d - 0.002 * d + 1.0193),
                    np.maximum(0.0, 0.54 - 0.001 * (d - 475)))


def blockage_mu(d):
    return 5.0 + np.maximum(0.0, 15 * np.log10(np.maximum(d, 1.0)) - 41)


def ibe_table(cfg):
    """Linear leakage factor (relative to in-band per-RE power) for each allocation start and subchannel."""
    L, P = cfg.L_sub, cfg.subch_prb
    nrb, lcrb = cfg.n_subch * P, L * P
    base = -25 - 10 * np.log10(nrb / lcrb)
    S0 = cfg.n_subch - L + 1
    tab = np.zeros((S0, cfg.n_subch))
    for x in range(S0):
        for k in range(cfg.n_subch):
            if x <= k < x + L:
                tab[x, k] = 1.0
                continue
            first = (x - k - 1) * P if k < x else (k - (x + L - 1) - 1) * P
            drb = first + np.arange(1, P + 1)
            leak = np.maximum(base, 20 * np.log10(cfg.evm) - 3 - 5 * (drb - 1) / lcrb)
            tab[x, k] = np.mean(10 ** (leak / 10)) if cfg.ibe else 0.0
    return tab


def wiener_gain(lam, sM2, sW2, p, pi, M):
    k = np.arange(M)
    Sig = sM2 * lam ** np.abs(k[:, None] - k[None, :]) + sW2 * np.eye(M) + 1e-9 * np.eye(M)
    c = (1 - p) * pi * (1 - pi) * lam ** (M - k)
    return np.linalg.solve(Sig, c)


# ----------------------------------------------------------------------------------- simulator
def simulate(cfg, opts, seed=0, record_ts=False):
    rng = np.random.default_rng(seed)
    R, L, nsub = cfg.rri, cfg.L_sub, cfg.n_subch
    S0 = nsub - L + 1
    Mtot = R * S0
    ovb = np.mean([min(x + L - 1, S0 - 1) - max(x - L + 1, 0) + 1 for x in range(S0)])
    Nr_eff = Mtot / ovb                              # equivalent number of orthogonal resources
    T_rri = R * 1e-3 * 15.0 / cfg.scs_khz
    if cfg.density > 0:
        N = int(round(cfg.density * cfg.L_road / 1000.0))
    else:
        N = int(round(2 * cfg.lanes_per_dir * cfg.L_road / (cfg.spacing_s * cfg.v_kmh / 3.6)))
    Lr, M, a = cfg.L_road, cfg.M, cfg.alpha_ew
    EC = 0.5 * (cfg.C_min + cfg.C_max)
    ds, adapt = opts.get('ds', False), opts.get('adapt')
    wiener, rssi = opts.get('wiener', False), opts.get('rssi', False)
    need_est = adapt is not None or wiener
    eta = opts.get('eta', 0.0)

    # ---- mobility
    direction = np.where(rng.permutation(N) % 2 == 0, 1.0, -1.0)
    lane = rng.integers(0, cfg.lanes_per_dir, N)
    ypos = direction * (2.0 + cfg.lane_w * (lane + 0.5))
    x = rng.uniform(0, Lr, N)
    vfac = np.clip(1 + cfg.v_spread * rng.standard_normal(N), 0.6, 1.4)
    dy2 = (ypos[:, None] - ypos[None, :]) ** 2
    eye = np.eye(N, dtype=bool)
    prof = {int(round(ts / T_rri)): v for ts, v in cfg.speed_profile} if cfg.speed_profile else {}

    def set_speed(vk):
        vel = direction * vfac * vk / 3.6
        rho = np.exp(-np.abs(vel[:, None] - vel[None, :]) * T_rri / cfg.d_corr)
        return vel, rho, np.sqrt(1 - rho ** 2)

    vel, rho_sh, sq_sh = set_speed(prof.get(0, cfg.v_kmh))
    sym = lambda Z: (np.triu(Z, 1) + np.triu(Z, 1).T)
    U = sym(rng.random((N, N)))
    Zb = sym(rng.standard_normal((N, N)))
    G = rng.standard_normal((N, N))
    Sh = cfg.sigma_S * (G + G.T) / SQ2

    N0 = 10 ** ((-174 + 10 * np.log10(cfg.scs_khz * 1e3) + cfg.NF) / 10)      # mW per RE
    P_RE = cfg.P_tx - 10 * np.log10(12 * L * cfg.subch_prb) + 2 * cfg.G_ant
    Klin = 10 ** (cfg.K_los_dB / 10)
    mu_l, sc_l = np.sqrt(Klin / (Klin + 1)), np.sqrt(1 / (2 * (Klin + 1)))
    IBE = ibe_table(cfg)
    Th0 = cfg.Th0

    def large_scale():
        dx = np.abs(x[:, None] - x[None, :])
        dx = np.minimum(dx, Lr - dx)
        d = np.sqrt(dx * dx + dy2)
        los = U < p_los_highway(d)
        pl = pl_los_highway(d, cfg.fc_GHz) + np.where(los, 0.0, np.maximum(blockage_mu(d) + 4.0 * Zb, 0.0))
        Pd = P_RE - pl - Sh
        np.fill_diagonal(Pd, -np.inf)
        return d, los, Pd, 10 ** (Pd / 10)

    Dm, LOS, Pdb, Pm = large_scale()
    zone = Pdb >= Th0
    inr_prev = (Dm <= cfg.D_pir) & ~eye

    # ---- MAC state
    arr_phase = rng.integers(0, R, N)
    next_tx = np.full(N, -1, dtype=np.int64)
    sub = np.zeros(N, dtype=np.int64)
    RC = np.zeros(N, dtype=np.int64)
    need_dec = np.zeros(N, dtype=bool)
    need_init = np.ones(N, dtype=bool)
    pk = np.full(N, float(opts.get('pk', 0.0)))
    grp = 'frac' in opts
    adopt = np.ones(N, dtype=bool)
    if grp:
        adopt[:] = False
        adopt[rng.permutation(N)[:int(round(opts['frac'] * N))]] = True
    reeval_at = np.full(N, -1, dtype=np.int64)
    reeval_th = np.zeros(N)
    arr_slot = np.zeros(N, dtype=np.int64)
    Rmap = np.full((N, R, nsub), -np.inf, dtype=np.float32)
    if rssi:
        Smap = np.full((N, R, nsub), N0, dtype=np.float64)
    Hc = np.full((N, R, S0, M), -np.inf, dtype=np.float16)
    hptr, filled = 0, 0
    nu, nuc = np.zeros((3, N)), np.zeros((3, N))
    mu, n1, n10 = np.zeros(N), np.zeros(N), np.zeros(N)
    est_init, n_est = False, 0
    bbar = np.full(N, 1.0 / EC)
    bmob_ew, E_true_ew = np.zeros(N), np.zeros(N)
    px_ew = np.full(N, 0.05)

    # ---- metrics
    n_slots = int(round(cfg.n_sec / T_rri)) * R
    warm = int(round(cfg.warm_sec / T_rri)) * R
    adapt_slot = int(round(cfg.adapt_sec / T_rri)) * R
    nb = int(cfg.d_max // cfg.d_bin)
    att, suc = np.zeros(nb), np.zeros(nb)
    prc_num = prc_den = 0
    NI_acc = NI_cnt = 0.0
    gap_hist = np.zeros(cfg.gap_max + 1)
    gap_ms = []
    last = np.full((N, N), -1, dtype=np.int64)
    bm_num = bm_den = 0.0
    px_num = px_den = xa_acc = xa_cnt = 0.0
    n_resel = n_reeval = 0
    est_log = dict(beta_wav=[], beta_tc=[], p_wav=[], lam_wav=[])
    Ew_log, Etc_log, Et_log = [], [], []
    pk_hist = np.zeros(len(an.PK_GRID))
    g_prc, g_att, g_suc = np.zeros(4), np.zeros((2, nb)), np.zeros((2, nb))
    g_gap = [[], []]
    ts = dict(prc=[], beta_wav=[], beta_tc=[], bmob=[], pk=[]) if record_ts else None
    coll_period = [0, 0]

    def cand_rsrp(Rm):
        """candidate-level RSRP (max over the L subchannels); NaN for non-monitored slots"""
        out = Rm[..., 0:S0]
        for l in range(1, L):
            out = np.maximum(out, Rm[..., l:l + S0])
        return out

    def select(u, t_lo, t_hi, t_now):
        nonlocal px_num, px_den, xa_acc, xa_cnt
        slots = np.arange(t_lo, t_hi + 1)
        sidx = slots % R
        Rm = Rmap[u]
        Yw = cand_rsrp(Rm)[sidx]
        hdw = np.isnan(Rm[sidx, 0])
        need = cfg.X_min * slots.size * S0
        th = Th0
        for _ in range(60):
            excl = hdw[:, None] | (Yw >= th)
            if slots.size * S0 - excl.sum() >= need:
                break
            th += 3.0
        cand = ~excl
        ci = np.flatnonzero(cand.ravel())
        if ci.size == 0:
            ci = np.arange(slots.size * S0)
        uw = wiener and est_init and t_now >= adapt_slot and adopt[u] and filled == M
        if uw:
            e = an.invert_structural(nu[0, u], nu[1, u], mu[u])
            lam_u, sM2_u, sW2_u, p_u, pi_u = (float(e[k]) for k in ('lam', 'sM2', 'sW2', 'p', 'pi'))
            if sM2_u > 1e-8:
                g = wiener_gain(lam_u, sM2_u, sW2_u, p_u, pi_u, M)
                order = (hptr + np.arange(M)) % M
                hist = Hc[u][:, :, order[1:]].astype(np.float32)          # M-1 older snapshots
                hist = np.concatenate([hist, cand_rsrp(Rm)[:, :, None]], axis=2)[sidx]
                hv = hist.reshape(-1, M)[ci]
                yv = np.where(np.isnan(hv), mu[u], (hv >= Th0).astype(float))
                shat = pi_u + (yv - mu[u]) @ g + 1e-7 * rng.random(ci.size)
                o = np.argsort(shat)
                pref = np.cumsum(shat[o]) / np.arange(1, ci.size + 1)
                Ks = np.arange(1, ci.size + 1)
                Jk = pref + pi_u * slots.size * S0 * bbar[u] / Ks
                Jk[:int(np.ceil(need)) - 1] = np.inf
                ci = ci[o[:int(np.argmin(Jk)) + 1]]
        if rssi:
            rs = Smap[u][sidx]
            rc = rs[:, 0:S0].copy()
            for l in range(1, L):
                rc += rs[:, l:l + S0]
            rv = rc.ravel()[ci]
            ci = ci[np.argsort(rv, kind='stable')[:int(cfg.X_min * slots.size * S0)]]
        pick = ci[rng.integers(ci.size)]
        row, xx = divmod(int(pick), S0)
        if t_now >= warm and zone[u].any():
            nbr = np.flatnonzero(zone[u] & (next_tx >= t_lo) & (next_tx <= t_hi) & ~need_dec & ~need_init)
            if nbr.size and not ds:
                rows = next_tx[nbr] - t_lo
                m_ = np.zeros(slots.size * S0, dtype=bool)
                m_[ci] = True
                m_ = m_.reshape(slots.size, S0)
                px_num += m_[rows, sub[nbr]].sum()
                px_den += nbr.size
            xa_acc += cand.sum() / (slots.size * S0)
            xa_cnt += 1
        return int(slots[row]), xx, th

    def start_new(u, t):
        nonlocal n_resel
        y, xx, th = select(u, t + cfg.T1, t + cfg.T2, t)
        next_tx[u], sub[u] = y, xx
        arr_slot[u] = t
        if cfg.reeval and y - cfg.T3 > t:
            reeval_at[u], reeval_th[u] = y - cfg.T3, th
        if t >= warm and not need_init[u]:
            n_resel += 1

    for t in range(n_slots):
        si = t % R
        meas = t >= warm
        # -------- re-evaluation of newly selected, not yet announced resources
        rv_u = np.flatnonzero(reeval_at == t)
        for u in rv_u:
            reeval_at[u] = -1
            s_ = next_tx[u] % R
            if np.nanmax(Rmap[u, s_, sub[u]:sub[u] + L]) >= reeval_th[u] and next_tx[u] > t + 1:
                y, xx, th = select(u, t + 1, min(arr_slot[u] + cfg.T2, next_tx[u] + R - 1), t)
                next_tx[u], sub[u] = y, xx
                if meas:
                    n_reeval += 1
        # -------- transmissions in slot t
        Tx = np.flatnonzero(next_tx == t)
        Rmap[:, si, :] = -np.inf
        Islot = None
        if Tx.size:
            nt = Tx.size
            xs = sub[Tx]
            Pr = Pm[Tx]
            hr = rng.standard_normal((nt, N, L))
            hi = rng.standard_normal((nt, N, L))
            g = np.where(LOS[Tx][:, :, None], (mu_l + sc_l * hr) ** 2 + (sc_l * hi) ** 2, 0.5 * (hr * hr + hi * hi))
            S_own = Pr[:, :, None] * g
            W = Pr[:, :, None] * IBE[xs][:, None, :]
            cols = xs[:, None] + np.arange(L)
            W[np.arange(nt)[:, None, None], np.arange(N)[None, :, None], cols[:, None, :]] = S_own
            Islot = W.sum(axis=0) + N0
            I_at = Islot[:, cols].transpose(1, 0, 2)
            sinr = S_own / np.maximum(I_at - S_own, 1e-30)
            ok_c = rng.random((nt, N)) < p_success(10 * np.log10(sinr[:, :, 0]), cfg.pscch_g50, cfg.pscch_slope)
            geff = miesm_qpsk(sinr, axis=2)
            ok_d = ok_c & (rng.random((nt, N)) < p_success(10 * np.log10(geff), cfg.pssch_g50, cfg.pssch_slope))
            ok_c[:, Tx] = False
            ok_d[:, Tx] = False
            if not ds:
                rs = (10 * np.log10(S_own.mean(axis=2))).astype(np.float32)
                for k_ in range(nt):
                    j = np.flatnonzero(ok_c[k_])
                    if j.size:
                        blk = Rmap[j, si, xs[k_]:xs[k_] + L]
                        Rmap[j, si, xs[k_]:xs[k_] + L] = np.maximum(blk, rs[k_, j][:, None])
            if meas:
                dT = Dm[Tx]
                bins = (dT / cfg.d_bin).astype(np.int64)
                valid = bins < nb
                valid[np.arange(nt), Tx] = False
                att += np.bincount(bins[valid], minlength=nb)[:nb]
                suc += np.bincount(bins[valid & ok_d], minlength=nb)[:nb]
                if nt > 1:
                    ov = np.abs(xs[:, None] - xs[None, :]) < L
                    np.fill_diagonal(ov, False)
                    coll = (zone[np.ix_(Tx, Tx)] & ov).any(axis=1)
                else:
                    coll = np.zeros(1, dtype=bool)
                prc_num += coll.sum()
                prc_den += nt
                coll_period[0] += coll.sum()
                coll_period[1] += nt
                for k_ in range(nt):
                    i = Tx[k_]
                    jj = np.flatnonzero(inr_prev[i])
                    if not jj.size:
                        continue
                    vj = jj[ok_d[k_, jj]]
                    lv = last[i, vj]
                    gg = t - lv[lv >= 0]
                    if gg.size:
                        gap_ms.append(gg)
                        gap_hist[:] += np.bincount(np.minimum(np.rint(gg / R).astype(np.int64), cfg.gap_max),
                                                   minlength=cfg.gap_max + 1)
                        if grp:
                            g_gap[0 if adopt[i] else 1].append(gg)
                    last[i, vj] = t
                if grp:
                    for gi, gm in enumerate((adopt[Tx], ~adopt[Tx])):
                        vg = valid & gm[:, None]
                        g_att[gi] += np.bincount(bins[vg], minlength=nb)[:nb]
                        g_suc[gi] += np.bincount(bins[vg & ok_d], minlength=nb)[:nb]
                        g_prc[2 * gi] += coll[gm].sum()
                        g_prc[2 * gi + 1] += gm.sum()
            Rmap[Tx, si, :] = np.nan
            RC[Tx] -= 1
            if ds:
                next_tx[Tx] = -1
            else:
                next_tx[Tx] += R
                need_dec[Tx[RC[Tx] <= 0]] = True
        if rssi:
            Smap[:, si, :] = 0.99 * Smap[:, si, :] + 0.01 * (Islot if Islot is not None else N0)
        # -------- packet arrivals
        A = np.flatnonzero(arr_phase == si)
        if A.size:
            if ds:
                for u in A:
                    start_new(u, t)
                    need_init[u] = False
            else:
                dec = A[need_dec[A]]
                if dec.size:
                    active = est_init and t >= adapt_slot
                    us = dec[adopt[dec]] if adapt is not None and active else dec[:0]
                    if us.size:
                        xA = np.clip(1.0 - mu[us] - 1.0 / R, cfg.X_min, 1.0)
                        if adapt == 'wav':
                            e = an.invert_structural(nu[0, us], nu[1, us], mu[us])
                            Eh = an.flux_from_count_spectrum(nuc[0, us], nuc[1, us], nuc[2, us])
                            bm = np.maximum(Eh / np.maximum(e['pi'] * Nr_eff, 1.0), 1e-4)
                            pxx = e['p']
                        elif adapt == 'tc':
                            bm = np.maximum(nuc[0, us] / np.maximum(mu[us] * Nr_eff, 1.0), 1e-4)
                            pxx = np.zeros(us.size)
                        else:
                            bm = np.maximum(bmob_ew[us], 1e-4)
                            pxx = px_ew[us]
                        pk[us] = an.pk_star(bm, pxx, xA, eta=eta)
                        if meas:
                            pk_hist += np.bincount(np.searchsorted(an.PK_GRID, pk[us] - 1e-9),
                                                   minlength=len(an.PK_GRID))[:len(an.PK_GRID)]
                    keep = rng.random(dec.size) < pk[dec]
                    RC[dec] = rng.integers(cfg.C_min, cfg.C_max + 1, dec.size)
                    need_dec[dec] = False
                    for u in dec[~keep]:
                        start_new(u, t)
                ini = A[need_init[A]]
                for u in ini:
                    start_new(u, t)
                    RC[u] = rng.integers(cfg.C_min, cfg.C_max + 1)
                    need_init[u] = False
        # -------- end of an RRI: mobility, large-scale channel, statistics
        if si == R - 1:
            p_idx = t // R + 1
            if p_idx in prof:
                vel, rho_sh, sq_sh = set_speed(prof[p_idx])
            x = (x + vel * T_rri) % Lr
            G = rng.standard_normal((N, N))
            Sh = rho_sh * Sh + sq_sh * cfg.sigma_S * (G + G.T) / SQ2
            zone_prev = zone
            Dm, LOS, Pdb, Pm = large_scale()
            zone = Pdb >= Th0
            inr = (Dm <= cfg.D_pir) & ~eye
            last[inr & ~inr_prev] = -1
            inr_prev = inr
            base = zone_prev.sum(1)
            exits = (zone_prev & ~zone).sum(1)
            entries = (~zone_prev & zone).sum(1)
            bmob_ew = (1 - a) * bmob_ew + a * exits / np.maximum(base, 1)
            E_true_ew = (1 - a) * E_true_ew + a * entries
            if meas:
                NI_acc += zone.sum()
                NI_cnt += N
                bm_num += exits.sum()
                bm_den += base.sum()
            Yc = cand_rsrp(Rmap)                                     # (N, R, S0)
            Hc[:, :, :, hptr] = Yc.astype(np.float16)
            hptr = (hptr + 1) % M
            filled = min(filled + 1, M)
            if need_est and filled == M:
                order = (hptr + np.arange(M)) % M
                Hw = Hc[:, :, :, order].reshape(N, R * S0, M).astype(np.float32)
                nanm = np.isnan(Hw)
                Y = np.where(nanm, np.nan, (Hw >= Th0).astype(np.float32))
                d1 = (Y[..., 7] - Y[..., 6]) / SQ2
                d2 = (Y[..., 7] + Y[..., 6] - Y[..., 5] - Y[..., 4]) / 2.0
                d3 = (Y[..., 4:].sum(-1) - Y[..., :4].sum(-1)) / (2 * SQ2)
                inst = np.nan_to_num(np.stack([np.nanmean(d1 ** 2, 1), np.nanmean(d2 ** 2, 1), np.nanmean(d3 ** 2, 1)]))
                mu_i = np.nan_to_num(np.nanmean(Y[..., 7], 1))
                prv, now = Y[..., 6], Y[..., 7]
                n1_i = ((prv == 1) & ~np.isnan(now)).sum(1)
                n10_i = ((prv == 1) & (now == 0)).sum(1)
                allmon = ~nanm.any(axis=2)
                Cn = ((Hw >= Th0 + cfg.dth_in) & allmon[..., None]).sum(axis=1).astype(float) / ovb
                instc = np.stack([((Cn[:, 7] - Cn[:, 6]) / SQ2) ** 2,
                                  ((Cn[:, 7] + Cn[:, 6] - Cn[:, 5] - Cn[:, 4]) / 2.0) ** 2,
                                  ((Cn[:, 4:].sum(1) - Cn[:, :4].sum(1)) / (2 * SQ2)) ** 2])
                n_est += 1
                at = max(a, 1.0 / n_est)
                est_init = True
                nu = (1 - at) * nu + at * inst
                nuc = (1 - at) * nuc + at * instc
                mu = (1 - at) * mu + at * mu_i
                n1 = (1 - at) * n1 + at * n1_i
                n10 = (1 - at) * n10 + at * n10_i
                bbar = (1 - at) * bbar + at * (1 - pk) / EC
                if adapt == 'oracle':
                    sj = np.where(next_tx >= 0, next_tx % R, 0)
                    F = np.nan_to_num(Yc[:, sj, sub], nan=-np.inf) >= Th0
                    nm_ = np.isnan(Rmap[:, :, 0])
                    own_s = np.where(nm_.any(1), nm_.argmax(1), -1)
                    mk = zone & (next_tx >= 0)[None, :] & (sj[None, :] != own_s[:, None])
                    cnt = mk.sum(1)
                    px_ew = np.where(cnt > 0, (1 - a) * px_ew + a * (1 - (F & mk).sum(1) / np.maximum(cnt, 1)), px_ew)
                if meas and (t // R) % 5 == 0:
                    e = an.invert_structural(nu[0], nu[1], mu)
                    est_log['beta_wav'].append(float(np.mean(e['beta'])))
                    est_log['p_wav'].append(float(np.mean(e['p'])))
                    est_log['lam_wav'].append(float(np.mean(e['lam'])))
                    est_log['beta_tc'].append(float(np.mean(n10 / np.maximum(n1, 1e-9))))
                    Ew_log.append(float(np.mean(an.flux_from_count_spectrum(nuc[0], nuc[1], nuc[2]))))
                    Etc_log.append(float(np.mean(nuc[0])))
                    Et_log.append(float(np.mean(E_true_ew)))
                if record_ts:
                    e = an.invert_structural(nu[0], nu[1], mu)
                    ts['beta_wav'].append(float(np.mean(an.flux_from_count_spectrum(nuc[0], nuc[1], nuc[2])
                                                        / np.maximum(e['pi'] * Nr_eff, 1))))
                    ts['beta_tc'].append(float(np.mean(nuc[0] / np.maximum(mu * Nr_eff, 1))))
                    ts['bmob'].append(float(bmob_ew.mean()))
                    ts['pk'].append(float(pk.mean()))
            if record_ts and meas:
                ts['prc'].append(coll_period[0] / max(coll_period[1], 1))
            coll_period = [0, 0]

    n_meas = (n_slots - warm) // R
    gm = np.concatenate(gap_ms) if gap_ms else np.zeros(1)
    aoi = float(np.sum(gm.astype(float) ** 2) / (2 * np.sum(gm)) * 1e-3) if gm.sum() > 0 else float('nan')
    NI = float(NI_acc / max(NI_cnt, 1))
    rho_m = N / Lr
    out = dict(
        cfg=asdict(cfg), opts=opts, seed=seed, N=N, Nr=float(Nr_eff), Mtot=Mtot, Rs=float(NI / (2 * rho_m)),
        d_centers=((np.arange(nb) + 0.5) * cfg.d_bin).tolist(), prr=(suc / np.maximum(att, 1)).tolist(),
        att=att.tolist(), P_RC=float(prc_num / max(prc_den, 1)), N_I=NI, gap_hist=gap_hist.tolist(), aoi=aoi,
        pir05=float(np.mean(gm > 500)), pir1=float(np.mean(gm > 1000)),
        beta_mob=float(bm_num / max(bm_den, 1)), px=float(px_num / max(px_den, 1)),
        xA=float(xa_acc / max(xa_cnt, 1)), omega=float('nan'), b_meas=float(n_resel / (N * max(n_meas, 1))),
        reeval_rate=float(n_reeval / (N * max(n_meas, 1))),
        est={k: float(np.mean(v)) if len(v) else float('nan') for k, v in est_log.items()},
        pk_hist=pk_hist.tolist(),
        flux=dict(E_wav=float(np.mean(Ew_log)) if Ew_log else float('nan'),
                  E_tc=float(np.mean(Etc_log)) if Etc_log else float('nan'),
                  E_true=float(np.mean(Et_log)) if Et_log else float('nan')))
    if grp:
        ga = [np.concatenate(g) if g else np.zeros(1) for g in g_gap]
        out['groups'] = dict(P_RC_A=float(g_prc[0] / max(g_prc[1], 1)), P_RC_L=float(g_prc[2] / max(g_prc[3], 1)),
                             prr_A=(g_suc[0] / np.maximum(g_att[0], 1)).tolist(),
                             prr_L=(g_suc[1] / np.maximum(g_att[1], 1)).tolist(),
                             aoi_A=float(np.sum(ga[0] ** 2.0) / (2 * max(ga[0].sum(), 1)) * 1e-3),
                             aoi_L=float(np.sum(ga[1] ** 2.0) / (2 * max(ga[1].sum(), 1)) * 1e-3))
    if need_est and est_init:
        e = an.invert_structural(nu[0], nu[1], mu)
        nu3p = an.wavelet_variance(3, e['lam'], e['sM2'], e['sW2'])
        out['scalogram'] = dict(nu=[float(v) for v in nu.mean(1)], mu=float(mu.mean()),
                                nuc=[float(v) for v in nuc.mean(1)], nu3_pred=float(np.mean(nu3p)),
                                gof=float(np.median(np.abs(nu3p - nu[2]) / np.maximum(nu[2], 1e-9))))
    if record_ts:
        out['ts'] = ts
    return out
