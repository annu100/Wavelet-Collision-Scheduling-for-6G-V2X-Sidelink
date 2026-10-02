"""Synthetic / analysis-only experiments. Output: results/theory.json"""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sawcs import analysis as an, estimators as es

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rng = np.random.default_rng(2026)
out = {}

# (a) Theorem 1: wavelet spectrum, Monte Carlo vs closed form
out['thm1'] = []
for lam in (0.5, 0.8, 0.95):
    for p in (0.0, 0.1):
        pi = 0.4
        y, _ = es.gen_occupancy(400, 4000, pi, lam, p, rng)
        d = es.haar_details(y, 6)
        emp = [float(np.mean(x ** 2)) for x in d]
        sM2, sW2 = (1 - p) ** 2 * pi * (1 - pi), pi * p * (1 - p)
        th = [float(an.wavelet_variance(j, lam, sM2, sW2)) for j in range(1, 7)]
        out['thm1'].append(dict(lam=lam, p=p, emp=emp, th=th))

# (b) estimator bias / RMSE vs flicker probability (W = 200 periods x 200 resources)
ps = np.round(np.arange(0, 0.201, 0.025), 3)
EST = {'wavelet': lambda y: es.est_wavelet_struct(y)[0], 'wavelet3': lambda y: es.est_wavelet(y)[0],
       'tc': es.est_tc, 'acf1': es.est_acf1, 'lagratio': es.est_lagratio}
est_names = list(EST)
res_b = {k: dict(mean=[], rmse=[]) for k in est_names}
pi, lam = 0.4, 0.9
beta0 = (1 - lam) * (1 - pi)
for p in ps:
    vals = {k: [] for k in est_names}
    for _ in range(150):
        y, _ = es.gen_occupancy(200, 200, pi, lam, p, rng)
        for k in est_names:
            vals[k].append(EST[k](y))
    for k in est_names:
        v = np.array(vals[k])
        res_b[k]['mean'].append(float(v.mean()))
        res_b[k]['rmse'].append(float(np.sqrt(np.mean((v - beta0) ** 2))))
out['est_flicker'] = dict(p=ps.tolist(), beta0=beta0, **res_b)

# (c) non-stationary occupancy level: pi(t) square wave of amplitude A (period 40)
amps = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25]
res_c = {k: dict(mean=[], rmse=[]) for k in est_names}
for A in amps:
    t = np.arange(200)
    pit = 0.4 + A * np.sign(np.sin(2 * np.pi * t / 40 + 0.1))
    vals = {k: [] for k in est_names}
    btrue = float(np.mean((1 - lam) * (1 - pit)))
    for _ in range(150):
        y, _ = es.gen_occupancy(200, 200, 0.4, lam, 0.05, rng, pi_t=pit)
        for k in est_names:
            vals[k].append(EST[k](y))
    for k in est_names:
        v = np.array(vals[k])
        res_c[k]['mean'].append(float(v.mean() / btrue))
        res_c[k]['rmse'].append(float(np.sqrt(np.mean((v / btrue - 1) ** 2))))
out['est_nonstat'] = dict(A=amps, **res_c)

# (d) estimator RMSE vs observation horizon W (p = 0.05)
Ws = [25, 50, 100, 200, 400, 800]
res_d = {k: [] for k in est_names}
for W in Ws:
    vals = {k: [] for k in est_names}
    for _ in range(120):
        y, _ = es.gen_occupancy(200, W, pi, lam, 0.05, rng)
        for k in est_names:
            vals[k].append(EST[k](y))
    for k in est_names:
        v = np.array(vals[k])
        res_d[k].append(float(np.sqrt(np.mean((v - beta0) ** 2)) / beta0))
out['est_horizon'] = dict(W=Ws, **res_d)

# (e) G(lambda) and h_j(lambda)
lg = np.linspace(0.001, 0.999, 400)
out['G'] = dict(lam=lg.tolist(), G=an.G_ratio(lg).tolist(),
                h=[an.h_closed(lg, 2 ** (j - 1)).tolist() for j in (1, 2, 3, 4)])

# (f) renewal moments: exact vs memoryless
pkg = np.round(np.arange(0, 0.951, 0.05), 3)
ren = np.array([an.renewal_moments(p) for p in pkg])
geo = np.array([an.geometric_moments((1 - p) / 10.0) for p in pkg])
out['renewal'] = dict(pk=pkg.tolist(), b=ren[:, 0].tolist(), m1=ren[:, 1].tolist(), m2=ren[:, 2].tolist(),
                      m3=ren[:, 3].tolist(), geo_m=geo[:, 1].tolist())

# (g) optimal keep probability map and closed form b*
bm = np.logspace(-4, -1.7, 60)
xa = np.linspace(0.2, 0.9, 50)
PK = np.array([[an.pk_star(b, 0.03, x)[0] for b in bm] for x in xa])
out['pkmap'] = dict(beta_mob=bm.tolist(), xA=xa.tolist(), pk=PK.tolist(),
                    b_closed=[an.b_star_closed(b, 0.5, omega=0.875).tolist() for b in bm],
                    b_exact=[float(an.renewal_moments(an.pk_star(b, 0.03, 0.5)[0])[0]) for b in bm])

# (h) regret (cosh law) analytic + exact renewal regret for kappa-misestimation
kap = np.logspace(-1.3, 1.3, 81)
out['regret'] = dict(kappa=kap.tolist(), cosh=an.regret_ratio(kap).tolist())
F_true = lambda bmob, pkv: an.objective_grid(bmob, 0.03, 0.5)[0][np.searchsorted(an.PK_GRID, pkv - 1e-9)]
exact = {}
for bmob in (0.001, 0.004, 0.008):
    opt = F_true(bmob, an.pk_star(bmob, 0.03, 0.5)[0])
    exact[str(bmob)] = [float(F_true(bmob, an.pk_star(bmob * k, 0.03, 0.5)[0]) / opt) for k in kap]
out['regret']['exact'] = exact

# (i) conserved-count flux: Haar (Cor. 1) vs lag-domain (gamma1-gamma2) vs naive first difference
def gen_count(T, E, beta, sw2, drift, rng):
    lvl = rng.poisson(E / beta)
    n = np.zeros(T)
    for t in range(T):
        lvl = lvl - rng.binomial(lvl, beta) + rng.poisson(E * (1 + drift * np.sin(2 * np.pi * t / 600)))
        n[t] = lvl
    return n + np.sqrt(sw2) * rng.standard_normal(T)

flux = {}
for drift in (0.0, 0.5):
    for W in (200, 400, 800, 1600):
        r = {'haar': [], 'lag': [], 'naive': []}
        for _ in range(100):
            n = gen_count(W, 0.3, 0.015, 2.5, drift, rng)[None, :]
            nu = [float(np.mean(x ** 2)) for x in es.haar_details(n, 3)]
            r['haar'].append(float(an.flux_from_count_spectrum(*nu)))
            z = n - n.mean()
            r['lag'].append(float(np.mean(z[:, :-1] * z[:, 1:]) - np.mean(z[:, :-2] * z[:, 2:])))
            r['naive'].append(nu[0])
        flux[f"d{drift}_W{W}"] = {k: dict(bias=float(np.mean(v) / 0.3 - 1),
                                           rmse=float(np.sqrt(np.mean((np.array(v) / 0.3 - 1) ** 2))))
                                  for k, v in r.items()}
out['flux_synth'] = flux

with open(os.path.join(ROOT, 'results', 'theory.json'), 'w') as f:
    json.dump(out, f)
print("theory done")
for k in est_names:
    print(k.ljust(9), 'bias@p=0.1 %+.3f' % (res_b[k]['mean'][4] / beta0 - 1), ' rmse@p=0.1 %.3f' % (res_b[k]['rmse'][4] / beta0),
          ' rmse@W=200 %.3f' % res_d[k][3], ' rmse@W=50 %.3f' % res_d[k][1], ' nonstat bias@A=0.2 %+.3f' % (res_c[k]['mean'][4] - 1))
for key, v in flux.items():
    print('flux', key, {k: (round(x['bias'], 2), round(x['rmse'], 2)) for k, x in v.items()})
