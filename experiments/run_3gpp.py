"""Campaign on the 3GPP slot-level simulator SL-Mode2Sim (resumable).
Usage: python experiments/run_3gpp.py [group ...]; results -> results/sl/*.json"""
import json, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from sawcs.sl_simulator import simulate, SLConfig, SCHEMES

OUT = os.path.join(ROOT, 'results', 'sl')
os.makedirs(OUT, exist_ok=True)


def opts_of(name):
    if name in SCHEMES:
        return dict(SCHEMES[name])
    if name.startswith('PK'):
        return dict(pk=float(name[2:]))
    if name.startswith('SA-WCS-A'):
        return dict(SCHEMES['SA-WCS'], eta=float(name[8:]) if len(name) > 8 else 0.01)
    if name.startswith('PEN'):
        return dict(SCHEMES['SA-WCS'], frac=float(name[3:]))
    raise ValueError(name)


def job(tag, kw, scheme, seed, ts=False):
    fn = os.path.join(OUT, f"{tag}__{scheme}__s{seed}.json")
    if os.path.exists(fn):
        return
    t = time.time()
    r = simulate(SLConfig(**kw), opts_of(scheme), seed=seed, record_ts=ts)
    r['tag'], r['scheme'] = tag, scheme
    with open(fn, 'w') as f:
        json.dump(r, f)
    print(f"{tag:12s} {scheme:13s} s{seed} P_RC={r['P_RC']:.4f} AoI={r['aoi']:.4f} ({time.time() - t:.0f}s)", flush=True)


D140, D70 = 61.7, 123.4                # TR 37.885 highway: 6 lanes, 2.5 s x speed spacing
B = dict(n_sec=45.0, warm_sec=15.0)
ALL = ['DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SPS-TC', 'Oracle', 'SA-WCS-P', 'SA-WCS-S', 'SA-WCS', 'SA-WCS-A0.01']
G = {}
G['main140'] = [("v140_d62", dict(B, v_kmh=140, density=D140), s, sd) for sd in (1, 2) for s in ALL]
G['main70'] = ([("v70_d123", dict(B, v_kmh=70, density=D70), s, 1) for s in ALL]
               + [("v70_d123", dict(B, v_kmh=70, density=D70), s, 2) for s in ('SPS', 'SA-WCS', 'SA-WCS-A0.01')])
G['sweep'] = [(f"v{v}_d62", dict(B, v_kmh=v, density=D140), f"PK{pk:.2f}", 1)
              for v in (30, 140, 250) for pk in (0.0, 0.2, 0.4, 0.6, 0.8, 0.9)]
G['speed'] = [(f"v{v}_d62", dict(B, v_kmh=v, density=D140), s, 1) for v in (30, 70, 250)
              for s in ('DS', 'SPS', 'SPS-P8', 'SPS-TC', 'Oracle', 'SA-WCS', 'SA-WCS-A0.01')]
G['density'] = [(f"v140_d{int(round(d))}", dict(B, v_kmh=140, density=d), s, 1) for d in (30.9, 92.6)
                for s in ('DS', 'SPS', 'SPS-P8', 'SPS-LTE', 'SA-WCS')]
G['pen'] = [("pen_v140", dict(B, v_kmh=140, density=D140), f"PEN{f}", 1) for f in (0.0, 0.25, 0.5, 0.75, 1.0)]
G['aoi'] = [(f"v{v}_d62", dict(B, v_kmh=v, density=D140), "SA-WCS-A0.03", 1) for v in (30, 140)]
G['nonstat'] = [("nonstat", dict(v_kmh=30, density=D140, speed_profile=((0, 30), (40, 140), (80, 30)),
                                 n_sec=120.0, warm_sec=10.0), s, 1, True) for s in ('SPS', 'SA-WCS', 'SPS-TC')]
G['sens'] = ([("th100_v140", dict(B, v_kmh=140, density=D140, Th0=-100.0), s, 1) for s in ('SPS', 'SA-WCS')]
             + [("noibe_v140", dict(B, v_kmh=140, density=D140, ibe=False), s, 1) for s in ('SPS', 'SA-WCS')])

if __name__ == '__main__':
    for g in (sys.argv[1:] or list(G)):
        for j in G[g]:
            job(*j)
    print("DONE", sys.argv[1:], flush=True)
