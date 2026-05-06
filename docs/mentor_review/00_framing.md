# Step 00 — Thesis Framing & Scope

**Mentor memo. Locks the thesis narrative for the MSc defense at
Unicamp/FEEC. Updates downstream artifacts to match.**

---

## 1. Audience and scope

| Decision | Value |
|---|---|
| **Primary audience** | MSc defense committee, Faculdade de Engenharia Elétrica e de Computação (FEEC), UNICAMP |
| **Document language** | English |
| **Document class** | Final dissertation (`\dissertacaoMestrado` in `tex/thesis.cls`); the qualification class flag will be flipped during Step 9 (LaTeX rebuild) |
| **Advisor** | Prof. Dr. Denis Fantinato, FEEC/UNICAMP |
| **IEEE conference paper** | Out of scope for now. May be derived from the thesis later, but no narrative or scope decision is made for an IEEE submission first. |
| **Open-source release** | In scope. Code is already MIT-licensed; Step 10 polishes the public surface. |

The thesis is a Master's dissertation, **not** an extended abstract or
a paper-equivalent monograph. The committee will read all five chapters
and the appendices, ask questions about every figure, and probe the
methodology more than the framing. We optimise for that audience.

## 2. The role of IoTWarden in this thesis

> **IoTWarden (Alam et al., 2024) is an *inspiration*, not a baseline.**

This is the central framing decision of Step 0c. It supersedes any
earlier wording in the repository.

What that means concretely:

- IoTWarden is cited in the **related work** section as a key
  motivating paper that established the trigger-action-attack /
  RL-defense paradigm.
- IoTWarden's **stage-action recommended-action mapping**
  (BENIGN→OBSERVE, RECON→LOG, ACCESS→THROTTLE, MANEUVER→BLOCK,
  IMPACT→ISOLATE) is **adopted** as a design choice in our
  environment. We acknowledge the source. We do not present this as
  "matching IoTWarden"; it is a sensible default mapping that we
  inherited, evaluated, and kept.
- Where individual figures (F3, F4, F7, F10) were *originally*
  conceived as visual analogues to IoTWarden Fig. 4(a), Fig. 5,
  Fig. 4(b) and Fig. 6 respectively, we drop the "reproduces / is
  aligned with / replicates" framing. Those figures stand on their
  own as **direct empirical results on CICIoT2023**. IoTWarden's
  figures are a different experiment on a different environment;
  apples-to-apples reproduction was never the contract.
- We do **not** quote "X% of the IoTWarden ceiling" or any framing
  that compares numerical results to IoTWarden. The committee can
  trivially dismiss any such claim with "different dataset,
  different MDP, different action space" and they would be right.
- The qualitative observation that *PPO mean reward grows
  monotonically with the defender de-escalation probability* is a
  **finding on our environment** (Phase 7, G7.3 PASS). It happens
  to share the qualitative shape with IoTWarden's Fig. 6, which we
  may note in passing as a sanity check. It is not a thesis claim.

## 3. Locked thesis claims

The thesis carries **three primary claims** (P1–P3) plus **two
pre-registered findings** (R1, R2). Every claim has a gate-passing or
gate-failing-with-finding artifact under `docs/results/`. None of them
are framed against IoTWarden.

### P1 — RL learns proportional kill-chain-aware defense on real IoT traffic

> On the held-out `test_balanced` split of CICIoT2023, trained DQN, PPO,
> and A2C agents earn mean episodic rewards of **+1336 / +1313 / +1297**
> respectively, statistically separated from every non-RL deployable
> baseline (random, always-OBSERVE, always-BLOCK, RandomForest-acting)
> with non-overlapping 95% bootstrap CIs. F6 confusion matrices show
> proportional behaviour on non-IMPACT stages (G6.3 PASS, scores
> 0.71–0.79). The agents observe a 29-feature CICIoT2023 vector and
> never see the true attack stage.

Evidence: `docs/results/06_benchmark/{F5,F6,F7,F8}_*` and
`G6_scoreboard.json`.

### P2 — RL is robust to attacker aggressiveness in a controlled, monotone way

> Holding the Phase-3 reward fixed, PPO mean reward grows monotonically
> with the defender de-escalation success probability `p_defender_de-escalation`
> from p=0.0 (CI 134, 141) to p=0.6 (CI 1280, 1359), i.e. a ~10×
> increase. The trend is monotone non-decreasing across the full sweep
> (G7.3 PASS).

Evidence: `docs/results/07_ablation/F10_aggressiveness.png` and
`F10_summary.json`.

### P3 — The reward function admits a structural lever that closes most of the gap to the oracle ceiling

> No single-axis 0.5×/2× perturbation of any reward coefficient closes
> the +288 gap to the oracle recommended-action ceiling. The gap is
> closed by a **structural** environment change — making IMPACT
> non-terminal (`impact_is_terminal=False`) — which lifts mean test
> reward from DQN's +1336 to +1542 and the **mitigated-impact rate**
> from 0.153 to **0.900** (a 5.9× improvement) while keeping reward
> within seed-noise of the oracle ceiling.

Evidence: `docs/results/07_ablation/F9_reward_ablation.png` and
`F9_summary.json`. G7.2 verdict: PASS-WITHOUT-STRETCH.

### R1 — RL is *robust to* but not *better at* an out-of-distribution attack class

> On the held-out OOD class `VulnerabilityScan`, DQN's mean reward
> (+1313) is within seed-noise of its in-distribution mean (+1336):
> the policy does not collapse. RandomForest-acting's higher OOD
> reward (+1611) is not evidence of RF's competence — RF recall on
> `VulnerabilityScan` is 0.001 — but evidence that "do nothing" is
> locally optimal under the Phase-3 reward when the OOD class is
> dominated by the disproportionate-penalty cost. This finding was
> **pre-registered** as a possible outcome in the Phase-7 plan
> (D7.9.1).

Evidence: `docs/results/07_ablation/F15_ood_robustness.png` and
`F15_summary.json`. G7.9 verdict: FAIL-WITH-FINDING (pre-registered).

### R2 — The security-vs-availability trade-off in this environment is approximately linear

> Across 32 reward-perturbed cells, only one Pareto-dominant point
> emerges — the trade-off surface is approximately linear under the
> Phase-3 reward formulation. A non-trivial Pareto front would require
> **non-linear** reward composition (e.g. a hard mit-rate constraint).
> This finding was **pre-registered** in the Phase-7 plan (R7.3).

Evidence: `docs/results/07_ablation/F12_pareto.png` and
`F12_summary.json`. G7.4 verdict: FAIL-WITH-FINDING (pre-registered).

## 4. Locked thesis chapter outline (5 chapters + appendices)

The dissertation follows the standard FEEC five-chapter structure.
The qualification draft used the same skeleton; content will be
rebuilt in Step 9 to align with this outline.

### Chapter 1 — Introduction
- Motivation: IoT attack surface, kill-chain framing, why supervised
  detection alone is insufficient, why RL.
- Threat model.
- Problem statement.
- Contributions (the five P/R bullets above, in plain English).
- Thesis structure.

### Chapter 2 — Background and Related Work
- IoT security primer.
- Cyber kill chain and related staged-attack models.
- CICIoT2023 dataset.
- Reinforcement learning fundamentals (MDP, DQN, PPO, A2C — the
  exact algorithms used).
- Related work in DRL-based intrusion detection / defense, including
  IoTWarden as inspiration; Tharewal et al.; the surveys in
  `docs/papers/`.

### Chapter 3 — Methodology
- §3.1 Dataset preparation and kill-chain projection (Phase 1).
- §3.2 Red team: Markov sequence generator + LSTM next-token
  predictor (Phase 2).
- §3.3 Stage detector (Phase 4).
- §3.4 Adversarial RL environment (Phase 3): MDP, action space,
  observation, reward.
- §3.5 Blue team training protocol (Phase 5): algorithms,
  hyperparameters, seeds.
- §3.6 Baseline policies (Phase 6).
- §3.7 Reproducibility framework (manifest hash chain).

### Chapter 4 — Results and Discussion
- §4.1 Red team validation (Phase 2; F1, F2).
- §4.2 Stage detector validation (Phase 4; F11).
- §4.3 Blue team training (Phase 5; F3, F4, T1).
- §4.4 Final benchmarks against baselines (Phase 6; F5, F6, F7, F8).
- §4.5 Reward and aggressiveness ablations (Phase 7; F9, F10, F12).
- §4.6 Out-of-distribution robustness (Phase 7; F15).
- §4.7 Discussion and threats to validity.

### Chapter 5 — Conclusions and Future Work
- Summary of contributions.
- Limitations.
- Future work (the items currently scoped as "Phase 8 / F13–F14",
  framed honestly as future work, not as missing pieces).

### Appendices
- App. A — Reproducibility, gate scoreboards, MLflow tracking,
  hash-chain protocol, hardware specifications.
- App. B — Full per-class CICIoT2023 → kill-chain stage mapping
  table.
- App. C — Hyperparameters per algorithm, random seeds, library
  versions.

## 5. Aggressive doc cleanup applied in this step

The following changes were made in Step 0c-exec to align the
repository with the framing above:

### Files created
- `docs/mentor_review/README.md`
- `docs/mentor_review/HANDOFF_TEMPLATE.md`
- `docs/mentor_review/00_framing.md` (this file)
- `docs/mentor_review/00_HANDOFF.md`

### Files edited
- `README.md` — softened IoTWarden language; removed
  "qualitatively reproduces IoTWarden Fig. 6" framing from claim P2;
  removed "extends IoTWarden" TL;DR phrasing; "82% of oracle
  ceiling" kept but reframed as in-domain optimal-policy
  ratio with no IoTWarden mention.
- `docs/thesis_results_map.md` — restructured: the columns
  "Tier" and "Aligned with" are dropped. New columns "Thesis
  chapter" and "Thesis section" are added. Tier comment moved to
  prose paragraph at the top.
- `docs/README.md` — fixed stale `src/benchmarking/*` reference to
  the canonical `src/benchmark/` (no `g`).
- `docs/HANDOFF.md` — STATUS banner added at the top redirecting the
  reader to `docs/mentor_review/` for the current state of the
  project; the rest of the file is preserved as historical record of
  the Phase 7 → Phase 10 closeout decision.
- Caption files (per-figure):
  - `docs/results/05_blue_team/F3_caption.md`
  - `docs/results/05_blue_team/F4_caption.md`
  - `docs/results/06_benchmark/F7_caption.md`
  - `docs/results/06_benchmark/F8_caption.md`
  - `docs/results/07_ablation/F10_caption.md`
  In each: "aligned with IoTWarden Fig. X" → "(IoTWarden's
  Fig. X presents a similar plot on a different environment; this
  figure is a direct CICIoT2023 result)" or omitted entirely.
- `docs/results/05_blue_team/RESULTS.md` and
  `docs/results/07_ablation/RESULTS.md` — softened "replicate /
  reproduce IoTWarden" wording. Numerical results untouched.
- `CHANGELOG.md` — appended a new top entry for the framing pass.

### Files NOT edited (frozen by protocol)
- All `docs/results/<NN>_<name>/PLAN.md` files — audit trail of
  what we planned at the time. Editing them would be revisionism.
- All `G<N>_scoreboard.json` files — numerical truth, immutable.
- `docs/results/00_phase0_diagnosis.md` — historical pre-restart
  audit; references to "IoTWarden head-to-head" are correctly
  marked as retired in Phase 6 already.
- `docs/decisions.md` — already free of IoTWarden-faithfulness
  claims.

## 6. Entry criteria for Step 1 (Phase 0–1 dataset review)

Step 1 begins when:

- The framing in this memo is acknowledged by the candidate (commit
  or out-of-band).
- The doc-cleanup edits in §5 are committed.
- `00_HANDOFF.md` is the highest-numbered handoff on disk.

Step 1 reviews:

- `docs/results/01_dataset/F0_class_distribution.png`
- `docs/results/01_dataset/F0_stage_distribution.png`
- `docs/results/01_dataset/F0_summary.json`
- `docs/results/01_dataset/manifest.json`
- `docs/dataset_card.md`
- `docs/data-pipeline.md`
- `docs/kill-chain-mapping.md`
- The split-building code: `scripts/data/build_split_indices.py`
- The kill-chain mapper: `src/utils/label_mapper.py`
- The realisation engine: `src/utils/realization_engine.py`

Verdict criteria for Step 1: splits are honest (no train/val/test
contamination), the per-stage class assignment is defensible (every
CICIoT2023 class maps to exactly one kill-chain stage with rationale),
the F0 plots are publication-clean and labelled correctly.

## 7. Sign-off

This memo locks the thesis framing. Subsequent mentor-review steps
operate under it. Any later step that wants to *revise* a framing
decision recorded here issues a *correction* in its own memo and
links back; this file is not edited.

— mentor-review agent, 2026-05-05
