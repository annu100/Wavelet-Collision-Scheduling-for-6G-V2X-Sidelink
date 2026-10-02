"""Figures and numeric summary from the 3GPP slot-level campaign (results/sl). Every legend is placed
outside the plotting area (above the panel, in a shared row, or to the right), so no data are hidden.
Outputs: figures/*.pdf and results/summary_sl.txt"""
import glob, json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sawcs import analysis as an

RES, FIG = os.path.join(ROOT, 'results'), os.path.join(ROOT, 'figures')
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'Nimbus Roman', 'DejaVu Serif'],
                     'mathtext.fontset': 'stix', 'font.size': 7, 'axes.labelsize': 7, 'legend.fontsize': 6,
                     'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'lines.linewidth': 1.0, 'lines.markersize': 3.2,
                     'axes.grid': True, 'grid.alpha': 0.3, 'grid.linewidth': 0.4, 'legend.handlelength': 1.8,
                     'legend.columnspacing': 0.9, 'legend.labelspacing': 0.25, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.02, 'pdf.fonttype': 42})
WF, W1 = 7.16, 3.5
ST = {'DS': dict(label='DS', color='0.5', marker='x', ls='--'),
      'SPS': dict(label=r'SPS ($P_k{=}0$)', color='k', marker='o', ls='-'),
      'SPS-P8': dict(label=r'SPS ($P_k{=}0.8$)', color='tab:blue', marker='s', ls='-'),
      'SPS-LTE': dict(label='SPS-RSSI', color='tab:purple', marker='^', ls='-.'),
      'SPS-TC': dict(label='SPS-TC', color='tab:orange', marker='v', ls='-.'),
      'Oracle': dict(label='Genie pers.', color='tab:green', marker='*', ls='--'),
      'SA-WCS-P': dict(label='SA-WCS-P', color='tab:olive', marker='d', ls=':'),
      'SA-WCS-S': dict(label='SA-WCS-S', color='tab:cyan', marker='<', ls=':'),
      'SA-WCS': dict(label='SA-WCS', color='tab:red', marker='D', ls='-'),
      'SA-WCS-A0.01': dict(label=r'SA-WCS-A ($\eta{=}0.01$)', color='tab:pink', marker='P', ls='--'),
      'SA-WCS-A0.03': dict(label=r'SA-WCS-A ($\eta{=}0.03$)', color='tab:brown', marker='X', ls=':')}
ESTN = {'wavelet': 'Haar (Prop. 1)', 'lagratio': 'Lag-ratio', 'wavelet3': 'Haar 3-scale', 'tc': 'Trans. count',
        'acf1': 'Lag-1 ACF'}
EMK = {'wavelet': ('tab:red', 'D', '-'), 'wavelet3': ('tab:pink', 'o', ':'), 'tc': ('tab:orange', 'v', '-.'),
       'acf1': ('tab:brown', '^', '--'), 'lagratio': ('tab:blue', 's', '-')}
SPC = {30: 'tab:blue', 140: 'tab:green', 250: 'tab:red'}
R = {}
for f in glob.glob(os.path.join(RES, 'sl', '*.json')):
    r = json.load(open(f))
    R.setdefault((r['tag'], r['scheme']), []).append(r)
TH = json.load(open(os.path.join(RES, 'theory.json')))
OUT = []


def say(*a):
    s = ' '.join(str(x) for x in a)
    OUT.append(s)
    print(s)


prr_at = lambda r, d: float(np.interp(d, r['d_centers'], r['prr']))
mean_pk = lambda r: float(np.asarray(r['pk_hist']) @ an.PK_GRID / max(np.sum(r['pk_hist']), 1)) if np.sum(r['pk_hist']) else float('nan')


def A(tag, sch, fn):
    v = np.array([fn(r) for r in R.get((tag, sch), [])], float)
    return (float(np.nanmean(v)), float(np.nanstd(v)), len(v)) if len(v) else (np.nan, np.nan, 0)


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + '.pdf'))
    plt.close(fig)


def noleg(s):
    return {k: v for k, v in ST[s].items() if k != 'label'}


def leg_above(ax, ncol, fs=5.6, **kw):
    ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.01), ncol=ncol, fontsize=fs, frameon=False,
              handlelength=1.5, columnspacing=0.7, **kw)


def top_legend(fig, handles, labels, ncol, y, fs=6):
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, y), ncol=ncol, fontsize=fs, frameon=False)


# ------------------------------------------------------------------ summary tables
ALL = ['DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SPS-TC', 'Oracle', 'SA-WCS-P', 'SA-WCS-S', 'SA-WCS', 'SA-WCS-A0.01']
for tag in ('v70_d123', 'v140_d62'):
    say(f"== MAIN {tag} (N={R[(tag, 'SPS')][0]['N']}, N_I={A(tag, 'SPS', lambda r: r['N_I'])[0]:.0f}, "
        f"Rs={A(tag, 'SPS', lambda r: r['Rs'])[0]:.0f} m)")
    for s in ALL:
        p = A(tag, s, lambda r: r['P_RC'])
        say(f"  {s:13s} P_RC={p[0]*100:.2f}+-{p[1]*100:.2f}(n={p[2]}) PRR300={A(tag, s, lambda r: prr_at(r, 300))[0]*100:.1f}"
            f" PRR500={A(tag, s, lambda r: prr_at(r, 500))[0]*100:.1f} AoI={A(tag, s, lambda r: r['aoi'])[0]*1e3:.1f}"
            f"+-{A(tag, s, lambda r: r['aoi'])[1]*1e3:.1f} PIR05={A(tag, s, lambda r: r['pir05'])[0]*100:.2f}"
            f" px={A(tag, s, lambda r: r['px'])[0]:.3f} xA={A(tag, s, lambda r: r['xA'])[0]:.2f}"
            f" b={A(tag, s, lambda r: r['b_meas'])[0]:.3f} pk={A(tag, s, mean_pk)[0]:.2f}")

pks = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
grid = np.round(np.arange(0, 0.901, 0.025), 3)
par = {}
for v in (30, 140, 250):
    runs = [r for pk in pks for r in R.get((f"v{v}_d62", f"PK{pk:.2f}"), [])]
    par[v] = {k: float(np.nanmean([r[k] for r in runs])) for k in ('N_I', 'xA', 'px', 'beta_mob', 'Nr')}
    par[v]['omega'] = 1 - (1 - par[v]['xA']) / 4
    par[v]['beta_zone'] = par[v]['beta_mob']
    par[v]['beta_mob'] = A(f"v{v}_d62", 'SA-WCS', lambda r: r['flux']['E_wav'])[0] / par[v]['N_I']   # Cor. 1 flux
Lam = lambda v, pk: an.collision_intensity(pk, par[v]['N_I'], par[v]['xA'], par[v]['Nr'], par[v]['px'],
                                           par[v]['beta_mob'], par[v]['omega'])
Gam = lambda v, pk: an.collision_age(pk, par[v]['N_I'], par[v]['xA'], par[v]['Nr'], par[v]['px'], par[v]['beta_mob'],
                                     par[v]['omega'])
say("== SWEEP (Theorem 2 validation)")
for v in (30, 140, 250):
    sim = [A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['P_RC'])[0] for pk in pks]
    mod = [Lam(v, pk)[0] for pk in pks]
    std = grid <= 0.8 + 1e-9
    mg = np.array([Lam(v, pk)[0] for pk in grid])
    say(f"  v={v} params " + ' '.join(f"{k}={x:.4g}" for k, x in par[v].items()) +
        f" | model opt={grid[std][np.argmin(mg[std])]:.3f} sim argmin(<=0.8)={pks[int(np.argmin(sim[:5]))]}")
    say("     sim " + ' '.join(f"{x*100:.2f}" for x in sim) + " | model " + ' '.join(f"{x*100:.2f}" for x in mod)
        + " | rel.err " + ' '.join(f"{m/s-1:+.2f}" for m, s in zip(mod, sim)))
    say("     AoI(ms) " + ' '.join(f"{A(f'v{v}_d62', f'PK{pk:.2f}', lambda r: r['aoi'])[0]*1e3:.1f}" for pk in pks))

speeds = [30, 70, 140, 250]
say("== SPEED (d=61.7 veh/km): P_RC(%) | PRR300 | AoI(ms) | mean pk")
for s in ['DS', 'SPS', 'SPS-P8', 'SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01']:
    say(f"  {s:13s} " + ' '.join(f"{A(f'v{v}_d62', s, lambda r: r['P_RC'])[0]*100:.2f}" for v in speeds) + " | " +
        ' '.join(f"{A(f'v{v}_d62', s, lambda r: prr_at(r, 300))[0]*100:.1f}" for v in speeds) + " | " +
        ' '.join(f"{A(f'v{v}_d62', s, lambda r: r['aoi'])[0]*1e3:.1f}" for v in speeds) + " | " +
        ' '.join(f"{A(f'v{v}_d62', s, mean_pk)[0]:.2f}" for v in speeds))
dens = [31, 62, 93]
say("== DENSITY (140 km/h): P_RC(%) | PRR300 | AoI")
for s in ['DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SA-WCS']:
    say(f"  {s:13s} " + ' '.join(f"{A(f'v140_d{d}', s, lambda r: r['P_RC'])[0]*100:.2f}" for d in dens) + " | " +
        ' '.join(f"{A(f'v140_d{d}', s, lambda r: prr_at(r, 300))[0]*100:.1f}" for d in dens) + " | " +
        ' '.join(f"{A(f'v140_d{d}', s, lambda r: r['aoi'])[0]*1e3:.1f}" for d in dens) +
        f" | xA {' '.join(f'{A(f"v140_d{d}", s, lambda r: r["xA"])[0]:.2f}' for d in dens)}")
say("== FRESHNESS WEIGHT: P_RC | PRR300 | AoI | PIR05 | pk")
for v in (30, 140):
    for s in ('SPS', 'SA-WCS', 'SA-WCS-A0.01', 'SA-WCS-A0.03'):
        say(f"  v{v} {s:13s} {A(f'v{v}_d62', s, lambda r: r['P_RC'])[0]*100:.2f} {A(f'v{v}_d62', s, lambda r: prr_at(r, 300))[0]*100:.1f}"
            f" {A(f'v{v}_d62', s, lambda r: r['aoi'])[0]*1e3:.1f} {A(f'v{v}_d62', s, lambda r: r['pir05'])[0]*100:.2f}"
            f" {A(f'v{v}_d62', s, mean_pk)[0]:.2f}")
fr = [0.0, 0.25, 0.5, 0.75, 1.0]
gv = lambda f, key: A('pen_v140', f"PEN{f}", lambda r: r['groups'][key])[0]
gp = lambda f, key: A('pen_v140', f"PEN{f}", lambda r: float(np.interp(300, r['d_centers'], r['groups'][key])))[0]
say("== PEN: f | all P_RC | adopters P_RC PRR300 AoI | legacy P_RC PRR300 AoI")
for f in fr:
    say(f"  {f}: {A('pen_v140', f'PEN{f}', lambda r: r['P_RC'])[0]*100:.2f} | "
        + (f"{gv(f, 'P_RC_A')*100:.2f} {gp(f, 'prr_A')*100:.1f} {gv(f, 'aoi_A')*1e3:.1f}" if f > 0 else '-') + ' | '
        + (f"{gv(f, 'P_RC_L')*100:.2f} {gp(f, 'prr_L')*100:.1f} {gv(f, 'aoi_L')*1e3:.1f}" if f < 1 else '-'))
say("== NONSTAT / SENS: P_RC | PRR300 | AoI")
for tag, schs in (('nonstat', ('SPS', 'SPS-TC', 'SA-WCS')), ('th100_v140', ('SPS', 'SA-WCS')), ('noibe_v140', ('SPS', 'SA-WCS'))):
    for s in schs:
        if (tag, s) in R:
            say(f"  {tag:11s} {s:8s} {A(tag, s, lambda r: r['P_RC'])[0]*100:.2f} {A(tag, s, lambda r: prr_at(r, 300))[0]*100:.1f}"
                f" {A(tag, s, lambda r: r['aoi'])[0]*1e3:.1f}")
say("== FLUX (SA-WCS, d62): speed | E_wav | E_tc | E_true | scalogram gof (SA-WCS-A)")
for v in speeds:
    say(f"  v={v} " + ' '.join(f"{A(f'v{v}_d62', 'SA-WCS', lambda r, k=k: r['flux'][k])[0]:.3f}" for k in ('E_wav', 'E_tc', 'E_true'))
        + f" | gof {A(f'v{v}_d62', 'SA-WCS-A0.01', lambda r: r['scalogram']['gof'])[0]:.3f}")

# Theorem 4 fit on the sweep
X, Y, Vv = [], [], []
for v in (30, 140, 250):
    for pk in pks:
        X.append(Lam(v, pk)[1]['total'] + Gam(v, pk))
        Y.append(A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['aoi'])[0])
        Vv.append(v)
X, Y, Vv = np.array(X), np.array(Y), np.array(Vv)
D = np.column_stack([Vv == 30, Vv == 140, Vv == 250, X]).astype(float)
coef = np.linalg.lstsq(D, Y, rcond=None)[0]
pred = D @ coef
R2 = 1 - np.sum((Y - pred) ** 2) / np.sum((Y - Y.mean()) ** 2)
say(f"== Theorem 4 fit: phi={2*coef[3]/0.1:.4f} R2={R2:.3f} max|err|={np.max(np.abs(Y-pred))*1e3:.1f} ms")
opt = []
for v in speeds:
    q = {k: A(f"v{v}_d62", 'SPS', lambda r, k=k: r[k])[0] for k in ('N_I', 'xA', 'px')}
    q['beta_mob'] = A(f"v{v}_d62", 'SA-WCS', lambda r: r['flux']['E_wav'])[0] / q['N_I']
    opt.append(an.pk_star(q['beta_mob'], q['px'], q['xA'])[0])
say("== pk vs speed: Thm3 " + ' '.join(f"{x:.2f}" for x in opt) + " | SA-WCS " +
    ' '.join(f"{A(f'v{v}_d62', 'SA-WCS', mean_pk)[0]:.2f}" for v in speeds) + " | SA-WCS-A " +
    ' '.join(f"{A(f'v{v}_d62', 'SA-WCS-A0.01', mean_pk)[0]:.2f}" for v in speeds) + " | Genie " +
    ' '.join(f"{A(f'v{v}_d62', 'Oracle', mean_pk)[0]:.2f}" for v in speeds))

# ================================================================== LETTER FIG. 1
fig, ax = plt.subplots(1, 4, figsize=(WF, 1.8))
cols = {0.5: 'tab:blue', 0.8: 'tab:green', 0.95: 'tab:red'}
for c in TH['thm1']:
    j = np.arange(1, 7)
    ax[0].semilogy(j, c['th'], color=cols[c['lam']], ls='-' if c['p'] == 0 else '--', lw=0.9)
    ax[0].semilogy(j, c['emp'], color=cols[c['lam']], marker='o' if c['p'] == 0 else '^', ls='none', ms=2.6,
                   mfc=cols[c['lam']] if c['p'] == 0 else 'none')
for lam, col in cols.items():
    ax[0].plot([], [], color=col, label=rf'$\lambda={lam}$')
ax[0].plot([], [], 'k-', label='$p=0$')
ax[0].plot([], [], 'k--', label='$p=0.1$')
leg_above(ax[0], 2, fs=5.2)
ax[0].set_xlabel('Scale $j$\n(a) Haar spectrum')
ax[0].set_ylabel(r'$\nu_j$')
e = TH['est_flicker']
for k in ['wavelet', 'lagratio', 'wavelet3', 'tc', 'acf1']:
    c, m, l = EMK[k]
    ax[1].plot(e['p'], np.array(e[k]['mean']) / e['beta0'], color=c, marker=m, ls=l, ms=2.6, label=ESTN[k])
leg_above(ax[1], 2, fs=5.2)
ax[1].set_xlabel('Flicker probability $p$\n(b) Churn estimation')
ax[1].set_ylabel(r'$\hat\beta/\beta$')
for v in (30, 140, 250):
    mg = np.array([Lam(v, pk)[0] for pk in grid])
    sim = np.array([A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['P_RC'])[0] for pk in pks])
    ax[2].plot(grid, mg * 100, color=SPC[v], lw=1.0)
    ax[2].plot(pks, sim * 100, color=SPC[v], marker='o', ls='none', ms=3, label=f'{v} km/h')
    std = grid <= 0.8 + 1e-9
    i = np.argmin(mg[std])
    ax[2].plot(grid[std][i], mg[std][i] * 100, marker='*', color=SPC[v], ms=8, mec='k', mew=0.4, ls='none')
ax[2].axvspan(0.8, 0.92, color='0.9', zorder=0)
ax[2].plot([], [], 'k-', label='Thm. 2')
ax[2].plot([], [], 'k*', ms=6, ls='none', label='Thm. 3 opt.')
ax[2].set_xlim(-0.02, 0.92)
leg_above(ax[2], 3, fs=5.2)
ax[2].set_xlabel('Keep probability $P_k$\n(c) Collisions vs $P_k$')
ax[2].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
sw = np.array([[A(f"v30_d62", f"PK{pk:.2f}", lambda r: r['aoi'])[0] * 1e3,
                A(f"v30_d62", f"PK{pk:.2f}", lambda r: r['P_RC'])[0] * 100] for pk in pks])
ax[3].plot(sw[:, 0], sw[:, 1], 'k-o', ms=2.8, label=r'SPS, $P_k\in[0,0.9]$')
for s_ in ('SA-WCS', 'SA-WCS-A0.01', 'SA-WCS-A0.03'):
    ax[3].plot(A("v30_d62", s_, lambda r: r['aoi'])[0] * 1e3, A("v30_d62", s_, lambda r: r['P_RC'])[0] * 100,
               ls='none', ms=5, **{k: vv for k, vv in ST[s_].items() if k != 'ls'})
leg_above(ax[3], 2, fs=5.2)
ax[3].set_xlabel('Mean AoI (ms)\n(d) Collisions vs AoI (30 km/h)')
ax[3].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
fig.tight_layout(w_pad=0.6)
save(fig, 'fig1_theory')
say("== Pareto 30 km/h SPS sweep (AoI ms, P_RC %): " + ' '.join(f"({x:.1f},{y:.2f})" for x, y in sw))

# ================================================================== LETTER FIG. 2
fig, ax = plt.subplots(1, 4, figsize=(WF, 1.62))
order = ['DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01']
hs = []
for s in order:
    rs = R[("v140_d62", s)]
    d = np.array(rs[0]['d_centers'])
    P = np.array([r['prr'] for r in rs]).mean(0)
    hs.append(ax[0].plot(d, P, markevery=3, ms=2.6, **ST[s])[0])
ax[0].set_xlabel('Tx-Rx distance (m)\n(a) PRR vs distance (140 km/h)')
ax[0].set_ylabel('PRR')
for s in ['SPS', 'SPS-P8', 'SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01']:
    pr = np.array([A(f"v{v}_d62", s, lambda r: r['P_RC']) for v in speeds])
    ax[1].plot(speeds, pr[:, 0] * 100, ms=2.8, **noleg(s))
for s in ['DS', 'SPS', 'SPS-P8', 'SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01']:
    ao = np.array([A(f"v{v}_d62", s, lambda r: r['aoi']) for v in speeds])
    ax[2].plot(speeds, ao[:, 0] * 1e3, ms=2.8, **noleg(s))
ax[1].set_ylim(0, None)
ax[1].set_xlabel('Speed (km/h)\n' + r'(b) $P_{\mathrm{RC}}$ vs speed')
ax[1].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
ax[2].set_xlabel('Speed (km/h)\n(c) AoI vs speed')
ax[2].set_ylabel('Mean AoI (ms)')
h1 = ax[3].plot(fr, [A('pen_v140', f"PEN{f}", lambda r: r['P_RC'])[0] * 100 for f in fr], 'k-o', ms=3)[0]
h2 = ax[3].plot(fr[1:], [gv(f, 'P_RC_A') * 100 for f in fr[1:]], color='tab:red', marker='D', ms=3)[0]
h3 = ax[3].plot(fr[:-1], [gv(f, 'P_RC_L') * 100 for f in fr[:-1]], color='0.4', marker='s', ls='--', ms=3)[0]
ax[3].set_xlabel('SA-WCS penetration\n(d) Mixed deployment')
ax[3].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.83))
top_legend(fig, hs + [h1, h2, h3], [h.get_label() for h in hs] + ['(d) all UEs', '(d) SA-WCS UEs', '(d) legacy UEs'],
           ncol=6, y=0.83, fs=5.8)
save(fig, 'fig2_system')

# ================================================================== SUPPLEMENTARY FIGURES
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
gg = TH['G']
for j, h in enumerate(gg['h'], 1):
    ax[0].semilogy(gg['lam'], h, label=f'$j={j}$')
leg_above(ax[0], 4, fs=6)
ax[0].set_xlabel(r'$\lambda$' + '\n(a) Scale functions $h_j$')
ax[0].set_ylabel(r'$h_j(\lambda)$')
ax[1].plot(gg['lam'], gg['G'], 'k')
ax[1].set_xlabel(r'$\lambda$' + '\n' + r'(b) Three-scale ratio $G(\lambda)$')
ax[1].set_ylabel(r'$G(\lambda)$')
fig.tight_layout()
save(fig, 'figS1_h_G')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
eh, en = TH['est_horizon'], TH['est_nonstat']
hs = []
for k in ['wavelet', 'lagratio', 'wavelet3', 'tc', 'acf1']:
    c, m, l = EMK[k]
    hs.append(ax[0].loglog(eh['W'], eh[k], color=c, marker=m, ls=l, label=ESTN[k])[0])
    ax[1].plot(en['A'], en[k]['mean'], color=c, marker=m, ls=l)
ax[0].set_xlabel('Observation horizon $W$ (periods)\n(a) RMSE vs horizon ($p=0.05$)')
ax[0].set_ylabel(r'Relative RMSE of $\hat\beta$')
ax[1].set_xlabel(r'Amplitude $A$ of the $\pi(t)$ square wave' + '\n(b) Non-stationary occupancy')
ax[1].set_ylabel(r'$\hat\beta/\bar\beta$')
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, hs, [h.get_label() for h in hs], 5, 0.87)
save(fig, 'figS2_estimators')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
Ws = [200, 400, 800, 1600]
hs = []
for i, dr in enumerate((0.0, 0.5)):
    for k, c, m, lab in (('haar', 'tab:red', 'D', 'Haar (Cor. 1)'), ('lag', 'tab:blue', 's', r'Lag domain $\hat\gamma_1-\hat\gamma_2$'),
                         ('naive', 'tab:orange', 'v', 'First difference')):
        h = ax[i].loglog(Ws, [TH['flux_synth'][f"d{dr}_W{W}"][k]['rmse'] for W in Ws], color=c, marker=m, label=lab)[0]
        if i == 0:
            hs.append(h)
    ax[i].set_ylabel(r'Relative RMSE of $\hat E$')
    ax[i].set_xlabel('Horizon $W$ (periods)\n' + ('(a) stationary density' if dr == 0 else '(b) density varying by 50 %'))
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, hs, [h.get_label() for h in hs], 3, 0.87)
save(fig, 'figS3_flux_synth')

fig, ax = plt.subplots(figsize=(W1, 2.1))
rn = TH['renewal']
ax.semilogy(rn['pk'], rn['m1'], 'r-', label=r'$m_1$')
ax.semilogy(rn['pk'], rn['m2'], 'g-', label=r'$m_2$')
ax.semilogy(rn['pk'], rn['m3'], 'b-', label=r'$m_3$')
ax.semilogy(rn['pk'], rn['geo_m'], 'k--', label=r'memoryless $1/(b(2-b))$')
ax.set_xlabel('$P_k$')
ax.set_ylabel('Periods')
leg_above(ax, 4, fs=6)
save(fig, 'figS4_renewal')

fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.1))
pm = TH['pkmap']
cs = ax[0].contourf(pm['beta_mob'], pm['xA'], pm['pk'], levels=np.arange(-0.025, 0.85, 0.05), cmap='viridis')
ax[0].set_xscale('log')
ax[0].grid(False)
ax[0].set_xlabel(r'Mobility churn $\beta_{\mathrm{mob}}$' + '\n(a) Exact $P_k^*$')
ax[0].set_ylabel('$x_A$')
cb = fig.colorbar(cs, ax=ax[0])
cb.set_label('$P_k^*$')
cb.ax.tick_params(labelsize=6)
ax[1].loglog(pm['beta_mob'], pm['b_exact'], 'r-', label='exact renewal optimum')
ax[1].loglog(pm['beta_mob'], pm['b_closed'], 'k--', label=r'$b^*=\sqrt{\zeta\beta_{\mathrm{mob}}x_A/\omega}$')
ax[1].axhline(0.02, color='0.5', lw=0.6, ls=':')
ax[1].axhline(0.1, color='0.5', lw=0.6, ls=':')
ax[1].set_xlabel(r'$\beta_{\mathrm{mob}}$' + '\n(b) Square-root law')
ax[1].set_ylabel('Optimal reselection rate $b^*$')
leg_above(ax[1], 1, fs=6)
fig.tight_layout()
save(fig, 'figS5_pkmap')

fig, ax = plt.subplots(figsize=(W1, 2.2))
rg = TH['regret']
ax.semilogx(rg['kappa'], rg['cosh'], 'k-', lw=1.2, label=r'Cor. 2: $(\sqrt{\kappa}+1/\sqrt{\kappa})/2$')
for (bm, vals), c in zip(rg['exact'].items(), ('tab:blue', 'tab:green', 'tab:red')):
    ax.semilogx(rg['kappa'], vals, color=c, ls='--', label=rf'exact, $\beta_{{\mathrm{{mob}}}}={bm}$')
ax.set_xlim(0.1, 10)
ax.set_ylim(0.98, 1.8)
ax.set_xlabel(r'Misestimation factor $\kappa$')
ax.set_ylabel('Cost / optimal cost')
leg_above(ax, 2, fs=6)
save(fig, 'figS6_regret')

fig, ax = plt.subplots(figsize=(W1, 2.3))
hs = []
for s in order:
    rs = R[("v70_d123", s)]
    d = np.array(rs[0]['d_centers'])
    P = np.array([r['prr'] for r in rs]).mean(0)
    hs.append(ax.plot(d, P, markevery=3, ms=2.6, **ST[s])[0])
ax.set_xlabel('Tx-Rx distance (m)')
ax.set_ylabel('PRR')
leg_above(ax, 3, fs=5.8)
save(fig, 'figS7_prr_v70')

fig, ax = plt.subplots(figsize=(W1, 2.2))
fx = np.array([[A(f"v{v}_d62", 'SA-WCS', lambda r, k=k: r['flux'][k])[0] for k in ('E_wav', 'E_tc', 'E_true')] for v in speeds])
ax.semilogy(speeds, fx[:, 2], 'k-o', label='zone entries (genie)')
ax.semilogy(speeds, fx[:, 0], 'r-D', label='Haar (Cor. 1)')
ax.semilogy(speeds, fx[:, 1], color='tab:orange', marker='v', ls='-.', label='first difference')
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel('Entries per period')
leg_above(ax, 3, fs=6)
save(fig, 'figS9_flux_system')

fig, ax = plt.subplots(figsize=(W1, 2.3))
for s in ('SA-WCS', 'SA-WCS-A0.01', 'Oracle', 'SPS-TC'):
    ax.plot(speeds, [A(f"v{v}_d62", s, mean_pk)[0] for v in speeds], **ST[s])
ax.plot(speeds, opt, 'k:', marker='*', ms=6, label=r'Thm. 3 (flux input)')
ax.set_ylim(0, 0.85)
ax.set_xlabel('Speed (km/h)')
ax.set_ylabel(r'Mean selected $P_k$')
leg_above(ax, 3, fs=5.8)
save(fig, 'figS10_pk_speed')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
hs = []
for s in ['DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SA-WCS']:
    hs.append(ax[0].plot(dens, [A(f"v140_d{d}", s, lambda r: r['P_RC'])[0] * 100 for d in dens], **ST[s])[0])
    ax[1].plot(dens, [A(f"v140_d{d}", s, lambda r: prr_at(r, 300))[0] * 100 for d in dens], **noleg(s))
ax[0].set_yscale('log')
ax[0].set_xlabel('Density (veh/km)\n(a) Collisions')
ax[0].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
ax[1].set_xlabel('Density (veh/km)\n(b) Reliability')
ax[1].set_ylabel('PRR at 300 m (%)')
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, hs, [h.get_label() for h in hs], 5, 0.87)
save(fig, 'figS11_density')

fig, ax = plt.subplots(3, 1, figsize=(4.9, 3.8), sharex=True)
rw, rt = R[('nonstat', 'SA-WCS')][0]['ts'], R[('nonstat', 'SPS-TC')][0]['ts']
te = (8 + np.arange(len(rw['beta_wav']))) * 0.1
ax[0].semilogy(te, rw['bmob'], 'k-', label='zone churn (genie)')
ax[0].semilogy(te, rw['beta_wav'], 'r-', label='Haar estimate')
ax[0].semilogy(te, rw['beta_tc'], color='tab:orange', ls='-.', label='first difference')
ax[0].set_ylabel(r'$\beta_{\mathrm{mob}}$')
ax[1].plot(te, rw['pk'], 'r-', label='SA-WCS')
ax[1].plot((8 + np.arange(len(rt['pk']))) * 0.1, rt['pk'], color='tab:orange', ls='-.', label='SPS-TC')
ax[1].set_ylabel(r'Mean $P_k$')
for s in ('SPS', 'SPS-TC', 'SA-WCS'):
    xx = np.convolve(R[('nonstat', s)][0]['ts']['prc'], np.ones(100) / 100, mode='valid')
    ax[2].plot((100 + 50 + np.arange(len(xx))) * 0.1, xx * 100, color=ST[s]['color'], ls=ST[s]['ls'], label=ST[s]['label'])
ax[2].set_ylabel(r'$P_{\mathrm{RC}}$ (%, 10-s avg.)')
ax[2].set_xlabel('Time (s); shaded: 140 km/h, otherwise 30 km/h')
for a_ in ax:
    a_.axvspan(40, 80, color='0.92', zorder=0)
    a_.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=6, frameon=False)
fig.tight_layout(h_pad=0.4)
save(fig, 'figS12_nonstat')

fig, ax = plt.subplots(figsize=(4.8, 2.1))
for s in order:
    h = np.sum([np.asarray(r['gap_hist']) for r in R[("v140_d62", s)]], axis=0)
    cc = 1 - np.cumsum(h) / h.sum()
    gi = np.arange(len(h))
    k = (gi >= 1) & (gi <= 15)
    ax.semilogy(gi[k] * 0.1, np.maximum(cc[k], 1e-7), **{kk: vv for kk, vv in ST[s].items() if kk != 'marker'})
ax.set_xlabel('Packet inter-reception time $x$ (s)')
ax.set_ylabel(r'$\Pr(\mathrm{PIR}>x)$')
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=6, frameon=False)
save(fig, 'figS13_pir_ccdf')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
q = par[140]
lc = np.array([[Lam(140, pk)[1][c] for c in ('BRC', 'MD', 'ENC')] for pk in grid])
bq = np.array([an.renewal_moments(pk)[0] for pk in grid])
Q2 = np.array([an.renewal_moments2(pk) for pk in grid])
SA = q['xA'] * q['Nr']
gc = np.column_stack([q['N_I'] * q['omega'] * bq ** 2 * Q2[:, 0] / SA, q['N_I'] * 2 * bq * q['px'] * Q2[:, 1] / SA,
                      q['N_I'] * q['beta_mob'] * Q2[:, 2] / q['Nr']])
hs = []
for i, (c, lab) in enumerate(zip(('tab:blue', 'tab:green', 'tab:red'), ('BRC', 'MD', 'ENC'))):
    hs.append(ax[0].plot(grid, lc[:, i], color=c, label=lab)[0])
    ax[1].semilogy(grid, gc[:, i], color=c)
hs.append(ax[0].plot(grid, lc.sum(1), 'k-', lw=1.3, label='total')[0])
ax[1].semilogy(grid, gc.sum(1), 'k-', lw=1.3)
ax[0].set_xlabel('$P_k$\n' + r'(a) Collision intensity $\Lambda$')
ax[0].set_ylabel(r'$\Lambda$ components')
ax[1].set_xlabel('$P_k$\n' + r'(b) Collision age $\Gamma$ (Thm. 4)')
ax[1].set_ylabel(r'$\Gamma$ components')
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, hs, [h.get_label() for h in hs], 4, 0.87)
save(fig, 'figS14_components')

fig, ax = plt.subplots(1, 2, figsize=(5.8, 2.2))
hs = []
for i, v in enumerate((30, 140)):
    swv = np.array([[A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['aoi'])[0] * 1e3,
                     A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['P_RC'])[0] * 100] for pk in pks])
    h = ax[i].plot(swv[:, 0], swv[:, 1], 'k-o', ms=3, label=r'SPS, fixed $P_k\in[0,0.9]$')[0]
    if i == 0:
        hs.append(h)
    for s in ('SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01', 'SA-WCS-A0.03'):
        h = ax[i].plot(A(f"v{v}_d62", s, lambda r: r['aoi'])[0] * 1e3, A(f"v{v}_d62", s, lambda r: r['P_RC'])[0] * 100,
                       ls='none', ms=6, **{k: vv for k, vv in ST[s].items() if k != 'ls'})[0]
        if i == 0:
            hs.append(h)
    ax[i].set_xlabel(f'Mean AoI (ms)\n({"ab"[i]}) {v} km/h')
    ax[i].set_ylabel(r'$P_{\mathrm{RC}}$ (%)')
fig.tight_layout(rect=(0, 0, 1, 0.83))
top_legend(fig, hs, [h.get_label() for h in hs], 3, 0.83)
save(fig, 'figS15_pareto')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
h1 = ax[0].plot(fr[1:], [gp(f, 'prr_A') * 100 for f in fr[1:]], color='tab:red', marker='D')[0]
h2 = ax[0].plot(fr[:-1], [gp(f, 'prr_L') * 100 for f in fr[:-1]], color='0.4', marker='s', ls='--')[0]
ax[1].plot(fr[1:], [gv(f, 'aoi_A') * 1e3 for f in fr[1:]], color='tab:red', marker='D')
ax[1].plot(fr[:-1], [gv(f, 'aoi_L') * 1e3 for f in fr[:-1]], color='0.4', marker='s', ls='--')
ax[0].set_xlabel('SA-WCS penetration\n(a) Reliability by group')
ax[0].set_ylabel('PRR at 300 m (%)')
ax[1].set_xlabel('SA-WCS penetration\n(b) Freshness by group')
ax[1].set_ylabel('Mean AoI (ms)')
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, [h1, h2], ['SA-WCS UEs', 'legacy UEs'], 2, 0.87)
save(fig, 'figS16_penetration')

fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.0))
cm = plt.get_cmap('viridis')
hs = []
for i, v in enumerate(speeds):
    rs = [r for r in R.get((f"v{v}_d62", 'SA-WCS-A0.01'), []) if 'scalogram' in r]
    nu = np.mean([r['scalogram']['nu'] for r in rs], 0)
    nup = np.mean([r['scalogram']['nu3_pred'] for r in rs])
    nc = np.mean([r['scalogram']['nuc'] for r in rs], 0)
    col = cm(i / (len(speeds) - 1))
    hs.append(ax[0].plot([1, 2, 3], nu, 'o', color=col, label=f'{v} km/h')[0])
    ax[0].plot([1, 2, 3], [nu[0], nu[1], nup], '-', color=col, lw=0.8)
    ax[1].semilogy([1, 2, 3], nc, '-o', color=col)
for a_ in ax:
    a_.set_xticks([1, 2, 3])
ax[0].set_xlabel('Scale $j$\n(a) Sensed occupancy: data vs Thm. 1')
ax[0].set_ylabel(r'$\hat\nu_j$ (per resource)')
ax[1].set_xlabel('Scale $j$\n(b) Conserved count')
ax[1].set_ylabel(r'$\hat\nu^{(n)}_j$ (inner count)')
fig.tight_layout(rect=(0, 0, 1, 0.87))
top_legend(fig, hs, [h.get_label() for h in hs], 4, 0.87)
save(fig, 'figS17_scalogram')

fig, ax = plt.subplots(figsize=(W1, 2.3))
fine = np.round(np.arange(0, 0.901, 0.025), 3)
for i, v in enumerate((30, 140, 250)):
    sim = np.array([A(f"v{v}_d62", f"PK{pk:.2f}", lambda r: r['aoi'])[0] for pk in pks])
    mod = np.array([coef[i] + coef[3] * (Lam(v, pk)[1]['total'] + Gam(v, pk)) for pk in fine])
    ax.plot(fine, mod * 1e3, color=SPC[v], lw=1.0)
    ax.plot(pks, sim * 1e3, color=SPC[v], marker='o', ls='none', ms=3, label=f'{v} km/h (sim.)')
ax.plot([], [], 'k-', label=r'Thm. 4, fitted $\varphi$')
ax.set_xlabel(r'Keep probability $P_k$')
ax.set_ylabel('Mean AoI (ms)')
leg_above(ax, 2, fs=6)
save(fig, 'figS19_aoi_model')

with open(os.path.join(RES, 'summary_sl.txt'), 'w') as f:
    f.write('\n'.join(OUT))
print("figures written")
