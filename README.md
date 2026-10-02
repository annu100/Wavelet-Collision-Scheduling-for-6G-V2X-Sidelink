# SA-WCS: Scale-Adaptive Wavelet Collision Scheduling for NR-V2X / 6G sidelink

Code and results for the letter "Scale-Adaptive Wavelet Collision Scheduling for 6G V2X Sidelink: A Stochastic
Framework for Optimal Semi-Persistent Resource Allocation" (Annu, P. Rajalakshmi, IIT Hyderabad).

| Path | Purpose |
|---|---|
| `sawcs/analysis.py` | Theorems 1-4, Propositions 1-2, Corollaries 1-3 (wavelet spectrum, estimators, collision intensity, square-root and collision-age laws) |
| `sawcs/estimators.py` | synthetic flickering-Markov occupancy and the five estimators compared in the paper |
| `sawcs/sl_simulator.py` | **SL-Mode2Sim**: slot-level NR-V2X Mode 2 simulator (TS 38.214 Sec. 8.1.4 selection, TS 38.321 SPS and re-evaluation, TR 37.885 highway scenario and channel, MIESM PHY abstraction, TS 38.101-1 in-band emission) with all schemes (DS, SPS, SPS-RSSI, SPS-TC, genie, SA-WCS variants, mixed deployment) |
| `experiments/exp_theory.py` | synthetic and analytical studies -> `results/theory.json` |
| `experiments/run_3gpp.py` | resumable system-level campaign -> `results/sl/*.json` (96 runs) |
| `experiments/make_figures_sl.py` | all figures and the numeric summary `results/summary_sl.txt` (every number in the papers) |
| `paper/` | LaTeX sources of the letter, supplementary material and cover letter |

Requirements: Python >= 3.10 with numpy, scipy, matplotlib; TeX with IEEEtran.
Reproduce everything: `pip install -r requirements.txt && bash run_all.sh` (about 80 min on one CPU core; the
campaign is resumable, e.g. `python3 experiments/run_3gpp.py main140`).

Scope: SL-Mode2Sim models periodic 300-byte traffic with a 100-ms RRI and no HARQ retransmissions; the BLER-curve
parameters, Rician K-factor and in-band-emission model are stated assumptions (supplement, Sec. S-III and S-VI).
Cross-validation on ns-3 5G-LENA is recommended before publication.
