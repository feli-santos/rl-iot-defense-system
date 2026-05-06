# F4 — Action-distribution evolution

Two-panel diagnostic of the **best-performing algo** (PPO, by D5.11:
highest mean eval reward over the last 10 % of training, averaged
across 5 seeds).

**(a) Top — marginal action share over training timesteps.**
Stacked-area chart with 25 K-step bins. Each band is the mean across
5 seeds of the fraction of *all* decisions in that bin that picked
`OBSERVE` (grey), `LOG` (green), `THROTTLE` (amber), `BLOCK`
(orange), or `ISOLATE` (red). The expected steady-state under the
oracle recommended-action policy is roughly LOG ≫ OBSERVE > BLOCK >
THROTTLE > ISOLATE because the LSTM Red Team spends most of its
pre-attack time in BENIGN/RECON. The agent recovers this ordering by
~50 K timesteps and holds it through training.

**(b) Bottom — per-stage histograms at three checkpoints.**
Five small panels (one per *decision-time* attack stage), each
showing three side-by-side bars per action: early (5 % of training,
faded), mid (50 %, faded), late (100 %, solid + black border). The
late-checkpoint argmax in every panel matches the oracle recommended
action at that stage (LOG@RECON, BLOCK@MANEUVER, …), demonstrating
that the agent learned **stage-action proportionality** rather than
collapsing to a single action.

**G5.5 PASS.** No per-stage action share exceeds 70 % at the late
checkpoint:

| Stage | argmax action | max share |
|---|---|---:|
| BENIGN   | LOG     | 0.45 |
| RECON    | LOG     | 0.34 |
| ACCESS   | LOG     | 0.30 |
| MANEUVER | BLOCK   | 0.40 |
| IMPACT   | (no decisions in late window — env terminates at IMPACT) | n/a |

The IMPACT row is empty by env design: the lifecycle terminates the
episode at IMPACT and the agent's IMPACT-step action is recorded as
the *previous* decision-stage's action, not as a separate IMPACT
decision. The breakdown of agent behaviour at the IMPACT step
(via `end_outcome`) is the G5.4 narrative in RESULTS.md §4.

Per-stage and per-bin counts in `F4_summary.json`; SHA-256
inputs/outputs in `F4_manifest.json`.
