# Step 7 — Phase 7 Ablations: Mentor Review Memo

> **Verdict.** `PASS-WITH-FIXES`. Phase 7 is the strongest scientific
> deliverable since Phase 1: F9 partially closes the +288
> oracle-ceiling gap that Phase 6 left, F15 cleanly activates the
> pre-registered D7.9.1 narrowing of the OOD claim, and the
> headline `impact_is_terminal=False` finding is publishable. Nine
> findings filed (one blocking-but-trivial, three medium, five
> minor); all are documentation / scoreboard hygiene; no data /
> figure correctness bugs. Test-split contract verified end-to-end
> in code. Hash chain intact for the data-flow path; two manifests
> need additional pins for transitive-reproducibility (mirrors of
> Phase-6 F3). Step-7 doc-fixes shipped in this branch; the rest
> batched into Step 8 with the Step-3/4/5/6 cleanup pile.
>
> **Mentor:** Cline, on behalf of Prof. Dr. Denis Fantinato.
> **Date:** 2026-05-06. **Branch:** `mentor-review/step-7-ablation`
> off `main` at `1d78fec`. **Predecessor merge:** Step 6 closed at
> `1d78fec` this session.

---

## 1. Headline Reads

**F9 — reward-component sweep (G7.2 PASS-WITHOUT-STRETCH; D7.1.1
partially activated).** The 12-cell sparse one-at-a-time grid
(5 reward components × {0.5×, 1×, 2×} + binary
`impact_is_terminal`) found that **no reward-coefficient cell**
beats the Phase-6 deployable DQN baseline (+1336) on the
apples-to-apples raw-reward strand. The single cell that does is
the env-semantics flip `impact_is_terminal=False`: PPO mean
**+1542 (CI 1524–1573)**, Δ_to_DQN = **+205.6**, **71 % of the
+288 oracle-ceiling gap closed**, with the security KPI
`mitigated_impact_rate` jumping from the 0.153 DQN baseline to
**0.900** (5.9× improvement). Source of record:
`docs/results/07_ablation/F9_summary.json`,
`G7_scoreboard.json#G7.2`,
`docs/results/07_ablation/RESULTS.md:13–23`.

**F15 — OOD-class robustness (G7.9 FAIL-WITH-FINDING; D7.9.1
ACTIVATED, audit-AF1 HEADLINE).** On `VulnerabilityScan` (the
class with Phase-4 RF recall = 0.001), trained DQN reaches +1313
(CI 1228–1387) but RF-Acting reaches +1611 (CI 1556–1666),
Δ = −298. The candidate took the honest path: pre-registered
D7.9.1 fires and the thesis claim narrows from "RL closes the
OOD gap" to **"RL is robust to (not better at) the OOD class"**
because DQN's mean OOD reward (+1313) is within seed-noise of its
in-distribution mean (+1336). The §6.2 RESULTS reasoning — that
RF-Acting wins by accident because BENIGN-default minimises
disproportionate-penalty cost when RF mis-predicts — is publishable
insight, not goalpost-moving. Source of record:
`G7_scoreboard.json#G7.9`,
`docs/results/07_ablation/F15_summary.json`,
`RESULTS.md:25–34`, `RESULTS.md:253–334`.

**F10 — aggressiveness sweep (G7.3 PASS).** PPO mean reward grows
strictly monotone in `p_defender_deescalation`: 137 (p=0.0) → 506
(p=0.2) → 860 (p=0.4) → 1320 (p=0.6) → 1710 (p=0.8) → 2047
(p=1.0); oracle rule curve also monotone non-decreasing. The
cleanest behavioural sanity-check in the thesis. Source:
`F10_summary.json`, `RESULTS.md:36–43`.

**F12 — Pareto (G7.4 FAIL-WITH-FINDING; R7.3 pre-registered).**
1 of 32 candidate points on the frontier; the candidate frames
this as "trade-off surface is approximately linear under the
Phase-3 reward formulation." The actual picture is even more
extreme — see Finding F7 below.

**Test-split contract.** Verified end-to-end in code: F9 / F10 /
F12 / F15 EVAL all use `split="test_balanced"` with
`exclude_ood=True`; train half delegates to `train_agent.py`'s
hardcoded `split="train", exclude_ood=True`. F15's hybrid
realiser preserves the contract (background pool is the
`exclude_ood=True` train set; OOD class is overlaid only at the
class's own kill-chain stage).
`scripts/ablation/run_reward_sweep.py:299, 301`;
`scripts/ablation/run_aggressiveness_sweep.py`;
`scripts/ablation/run_ood_eval.py`.

**Hash chain.** Phase-1 splits, Phase-5 sweep, Phase-6 eval, all
on-disk SHAs match the SHA strings recorded in F9 + F15
manifests. F10 and F12 manifests are short on transitive pins
(see F2). Phase-1 splits manifest `1e99d596…` is pinned
transitively via Phase-6 eval manifest, not directly.

**Pytest.** 411 collected, 411 passing on this branch tip. The
"454" recorded in `G7_scoreboard.json#G7.1` is **historically
correct at Phase-7 lock** but is now stale — Phase-10 hygiene
commit `281860a` deleted `tests/test_benchmark_runner.py` and
`tests/test_metrics_collector.py` (43 tests total). See F1
below; this is a documentation note, not a regression.

---

## 2. Gate Scoreboard (verified against on-disk artefacts)

| Gate | PLAN threshold (verbatim) | Recorded | Mentor verify | Verdict |
|---|---|---|:---:|:---:|
| **G7.1** | `pytest -q` ≥ 430 passed; zero new skips | "454 passed, 2 warnings in 63.07s" | 411 passed at HEAD (post-`281860a` Phase-10 deletion of 43 dead tests; lock-time count of 454 verified historically) | PASS (at lock) → see F1 |
| **G7.2** | F9 best cell mean test reward ≥ DQN +1336 by ≥ 1σ; stretch: meet oracle +1624 | apples-to-apples best = `impact_is_terminal_false` (+1542); CI=(1524, 1573); Δ_DQN=+205.6; meets_oracle_stretch=False | F9_summary.json verified | **PASS-WITHOUT-STRETCH** ✅ |
| **G7.3** | PPO p=0.0 < p=0.6 by ≥1σ AND rule monotone | p=0.0 CI=(133.5, 140.7); p=0.6 CI=(1280.1, 1359.2); rule monotone non-decreasing | F10_summary.json verified | **PASS** ✅ |
| **G7.4** | Pareto frontier ≥ 3 distinct dominant points | n_distinct = 1/32 | F12_summary.json verified; **but security_gain=0.0 for ALL 32 points** (see F7) | **FAIL-WITH-FINDING (R7.3)** + F7 |
| **G7.5** | Phase-3 frozen tests pass with `impact_is_terminal=True` | "G7.1 carries this through" | `tests/test_phase3_env_gates.py` + `test_adversarial_env.py` + `test_phase31_impact_terminal.py` all green | **PASS** ✅ |
| **G7.6** | No regression on Phase-3/4/5/6 frozen tests | "G7.1 carries this through" | 411/411 green | **PASS** ✅ |
| **G7.7** | F9/F10/F12/F15 manifest.json all present + SHA-pinned | "all 4 manifests present" | All 4 present; F10 + F12 missing some transitive pins (F2) | **PASS-WITH-FINDING** → F2 |
| **G7.8** | F15 4-class × 8-policy matrix complete, no NaN means | 32/32; n_missing=0; n_nan=0 | F15_summary.json verified | **PASS** ✅ |
| **G7.9** | On VulnerabilityScan, best trained RL CI_low > RF-Acting CI_high | best_rl=DQN +1313, RF=+1611, Δ=−298 | F15_summary.json verified | **FAIL-WITH-FINDING (D7.9.1)** ✅ pre-registered |

**Tally.** 7 PASS / 2 pre-registered FAIL-WITH-FINDING. Scientific
acceptance criterion (PLAN §3.4) met in full: pre-registered
fail-modes do not block Step-7 sign-off.

PLAN evidence pointers:
`docs/results/07_ablation/PLAN.md:432–440` (gate definitions);
PLAN §6 (R7.3); PLAN §8 (D7.1.1, D7.9.1); RESULTS §2; scoreboard
`docs/results/07_ablation/G7_scoreboard.json`.

---

## 3. Findings (priority-ordered)

### F1 (M, fix-trivial) — `RESULTS.md §9` and `G7_scoreboard.json#G7.1` test count is post-locking-stale

**Symptom.** RESULTS §9 says "Phase 7 closer fix 454" tests;
`G7_scoreboard.json#G7.1.value` reads
`"================== 454 passed, 2 warnings in 63.07s … =================="`.
But `pytest --collect-only -q` at HEAD reports `411 tests
collected in 5.32s`. Δ = **−43 tests**.

**Root cause (forensic).** Phase-10 hygiene cleanup commit
`281860a fix(phase-10,§3.2): delete dead src/benchmarking/
package + tests (D10.2)` deleted
`tests/test_benchmark_runner.py` and
`tests/test_metrics_collector.py` — exactly 43 tests. This
commit landed *after* Phase 7 closed (Phase-7 close commit:
`396f827`; D10.2 deletion: `281860a`, three days later). The
454 number was **historically correct at Phase-7 lock**; the
stale-ness is an honest by-product of Phase-10's correct
decision to retire dead code.

**Severity.** Documentation-only. The PLAN's `≥ 430` threshold
was met at lock time; the deleted tests were dead code (no
matching `src/benchmarking/` package). No correctness regression.

**Why "fix-trivial".** Add a one-paragraph footnote to RESULTS
§9 disclosing the post-locking erosion + commit ref `281860a`.
Add a `note` field to the `G7_scoreboard.json#G7.1` row pointing
to the same commit.

**Mentor recommendation.** Ship the doc-fix in Step 7 (this
branch). Conventional Commits: `docs(phase-7,§9):` and
`docs(phase-7,scoreboard):`.

**Evidence.**
- `docs/results/07_ablation/RESULTS.md:438–442`.
- `docs/results/07_ablation/G7_scoreboard.json#gates[0].value`.
- `git --no-pager log --oneline 281860a^..281860a` →
  `281860a fix(phase-10,§3.2): delete dead src/benchmarking/`.
- `pytest --collect-only -q` at HEAD = 411.

### F2 (M, fix-cheap) — F10 and F12 manifests under-pin upstream SHAs

**Symptom.** Inputs section of each manifest:

| Manifest | Phase-7 own | Phase-6 eval | Phase-5 sweep | Eval JSONLs |
|---|:---:|:---:|:---:|:---:|
| `F9_manifest.json` | ✅ | ✅ | ❌ | ✅ (60 paths) |
| `F10_manifest.json` | ✅ | **❌** | **❌** | ✅ |
| `F12_manifest.json` | ✅ (F9 + F10) | ✅ | **❌** | ✅ |
| `F15_manifest.json` | ✅ | ✅ | ✅ | ✅ (160+ paths) |

**Root cause.** F10's plotter does load Phase-6 oracle-rule
reference output for the overlay rule curve (visible in
F10_summary.json `rule_rows`), but its manifest emitter does
not pin `phase6_eval_manifest`. F12 and F10 do not pin
`phase5_sweep_manifest` though they trace upstream to
Phase-5-trained PPO checkpoints via the F9 / F10 / aggregator
path. F9 also lacks a Phase-5 sweep pin (Phase-5 ckpts are warm-
starts implicit in the reward-sweep training).

**Severity.** Mirror of Phase-6 F3. The on-disk SHA chain is
intact (verified: `cc7454…/c4a60a…/19f35074…/86d20b33…` all
match what's in the manifests that DO pin them); the gap is in
the *recorded* chain. Phase-1 splits manifest `1e99d596…` is
pinned only transitively through Phase-6 eval manifest.

**Mentor recommendation.** Defer to Step 8 with the rest of the
cross-cutting hygiene batch. Fix is a 4-line patch in each of
`scripts/ablation/plot_aggressiveness.py` (F10) and
`scripts/ablation/plot_pareto.py` (F12) to compute and embed
`shasum runs/phase{5,6}/...` in the manifest dict. Cheap
because the upstream files are stable on disk. Add an explicit
Phase-1 splits pin to all four manifests for transitive-clarity.

**Evidence.**
- `docs/results/07_ablation/F10_manifest.json` `inputs.keys() =
  ['phase7_aggressiveness_sweep_manifest', 'eval_jsonls_sha256']`.
- `docs/results/07_ablation/F12_manifest.json` `inputs.keys() =
  ['phase6_eval_manifest', 'phase7_f9_sweep_manifest',
  'phase7_f10_sweep_manifest', 'eval_jsonls_sha256']`.

### F3 (M, fix-batched) — Scoreboard schema uses `passes:bool` not Phase-6's native `status:enum + finding_id`

**Symptom.** Each gate row in `G7_scoreboard.json` carries:

```jsonc
{
  "id": "G7.4",
  "threshold": "...",
  "value": "n_distinct=1/32",
  "passes": false,
  "kind": "f12",
  "interpretation": "FAIL-WITH-FINDING (R7.3): ..."
}
```

The "FAIL-WITH-FINDING" verdict and the `R7.3` reference are
embedded as **free text** inside `interpretation`. Phase 6
(verified: `docs/results/06_benchmark/G6_scoreboard.json`) ships
native:

```jsonc
{
  "status": "PASS-WITH-FINDING",
  "finding_id": "F6.4",
  ...
}
```

**Root cause.** `scripts/ablation/close_phase7.py:_write_scoreboard`
emits the Phase-7 schema; Phase 6 emits its own schema natively.
The two were not unified.

**Severity.** Cross-cutting; same backfill target as Step-4 G4.4
and Step-5 G5.4 (already on the Step-8 cleanup list). Adding
Phase 7 to the same backfill keeps the change scoped to a single
commit in Step 8.

**Mentor recommendation.** Defer to Step 8. Step-8 deliverable:
parse `interpretation` strings into `status` + `finding_id`,
re-emit `G4_scoreboard.json`, `G5_scoreboard.json`,
`G7_scoreboard.json`. Use **`status`** (Phase-6 native) — not
`verdict` — to avoid a schema split.

**Evidence.**
- `docs/results/07_ablation/G7_scoreboard.json` (full file).
- `scripts/ablation/close_phase7.py:355–374` emitter.

### F4 (M, fix-trivial) — F9 baseline `mitigated_impact_rate` (0.273) is easy to misread against the §6.1 baseline (0.153)

**Symptom.** `F9_summary.json` `baseline_phase5_defaults.mitigated_impact_rate = 0.273`.
RESULTS §1 line 19 quotes the win as "**0.900 vs the DQN
baseline 0.153** (5.9× improvement)". `G7_scoreboard.json#G7.2.deployable_best_mitigated = 0.153`. Both are correct (0.273 = Phase-7 PPO @ Phase-5 defaults; 0.153 = Phase-6 DQN deployable best from
RESULTS §6.1) but the scoreboard's "5.9×" depends on the 0.153
denominator while the F9 panel includes a 0.273 row that a
casual reader might use as the denominator.

**Severity.** Docs / caption only. Defendability — a committee
member reading F9 alone (not §6.1) might compute 0.900 / 0.273 =
3.3× and disagree with the headline 5.9×.

**Mentor recommendation.** Ship in Step 7. One-line addition to
`F9_caption.md` clarifying the Phase-6 anchor row vs the
Phase-7 PPO baseline; one-sentence footnote in `RESULTS.md §1`
disambiguating which baseline the 5.9× ratio is computed
against.

**Evidence.**
- `docs/results/07_ablation/F9_summary.json#rows[0].mitigated_impact_rate=0.273`.
- `docs/results/07_ablation/RESULTS.md:19`.
- `docs/results/07_ablation/G7_scoreboard.json#gates[1].deployable_best_mitigated=0.153`.

### F5 (S, fix-trivial) — Test-split contract is implicit, not documented in RESULTS §8

**Symptom.** PLAN §3 requires every Phase-7 figure use
`split="test_balanced", exclude_ood=True` for headline metrics.
This contract IS satisfied in code (verified line-by-line in
`run_reward_sweep.py:299–301`, `run_aggressiveness_sweep.py`,
`run_ood_eval.py`). But RESULTS §8 ("Reproducibility") doesn't
state the contract anywhere — a reader has to infer it from
manifests + run scripts.

**Mentor recommendation.** Ship in Step 7. Add a one-paragraph
"Eval contract" subsection to RESULTS §8 stating: "All headline
test rewards (F9, F10, F12, F15-aggregated) evaluate on
`split='test_balanced', exclude_ood=True`. F15 OOD-class cells
overlay the held-out class on top of an `exclude_ood=True`
in-distribution background via the hybrid realiser
(`scripts/ablation/run_ood_eval.py`)."

**Evidence.**
- `scripts/ablation/run_reward_sweep.py:299, 301` —
  `spec.split = "test_balanced"`,
  `spec = EnvConfigSerializable(split="test_balanced", exclude_ood=True)`.
- `docs/results/07_ablation/PLAN.md:432–440` — gate definitions.
- `docs/results/07_ablation/RESULTS.md:415–435` — §8 absent the
  contract clause.

### F6 (M, defer to Phase 8/9) — MANEUVER (kill-chain stage 3) coupling not addressed in F9

**Symptom.** Phase-6 F6 inspection flagged DQN at 58 % ISOLATE
on MANEUVER — same de-escalation-farming pattern as IMPACT. The
Step-7 brief asked whether F9 treated MANEUVER+IMPACT as a
coupled remediation target. F9's `impact_is_terminal_false` cell
flips IMPACT semantics only. The MANEUVER farming pattern is
not addressed in any F9 cell, and `grep -rni "maneuver\|stage_3"
scripts/ablation/run_reward_sweep.py` returns no matches. No
`maneuver_is_terminal` axis exists in `AdversarialEnvConfig`.

**Severity.** Phase 7 wasn't asked to fix MANEUVER farming; the
PLAN §3.1.4 reward-component sweep deliberately stays inside the
existing reward function. But the Phase-6 F6 follow-up was
explicitly requested for Step 7 to *check*, and the answer is
"not done — natural Phase-8 / Phase-9 follow-up."

**Mentor recommendation.** Document in `RESULTS.md §7`
("Phase-8 hand-offs") as a third bullet: *"MANEUVER-stage
de-escalation farming — same pattern Phase-6 F6 flagged for
DQN at 58 % ISOLATE on MANEUVER (stage 3); not addressed by F9
(`impact_is_terminal=False` only flips stage-4 / IMPACT). A
parallel `maneuver_is_terminal` flag would extend the env-
semantics ablation; deferred to Phase 8 F14 territory."*

**Evidence.**
- `docs/mentor_review/06_benchmark.md` (F6 inspection).
- `scripts/ablation/run_reward_sweep.py` (no MANEUVER axis).
- `docs/results/07_ablation/RESULTS.md:387–414` (§7 currently
  lists only F13 + F14).

### F7 (M, finding) — F12 `security_gain` is identically 0.0 across all 32 points; "Pareto frontier" is degenerate, not just "approximately linear"

**Symptom.** Every row in `F12_summary.json` reports
`"security_gain": 0.0`. All 32 candidates collapse to a 1-D
problem (only `availability_cost` varies). The frontier of "1
distinct dominant point" is *only* `always_observe`
(`availability_cost = 0.0`), trivially.

**Root cause.** `security_gain = 1 − compromise_rate`. Every
F9 cell has `compromise_rate = 1.0` (visible in F9_summary
rows, including `impact_is_terminal_false`). Every Phase-6
anchor policy has the same. So F12's y-axis is degenerate.

**Severity.** F12 is mathematically valid (the "trade-off
surface is approximately linear" claim is true — vacuously,
because one dimension is constant). But the **caption** says
"Pareto frontier" and the figure draws a frontier curve. A
defense-committee member who reads F12 will ask "why does no
config improve security on the held-out test split?" That is
answered in §6.4 obliquely, but not under F12. The R7.3
pre-registration captures this *qualitatively* but the actual
situation is more extreme.

**Mentor recommendation.** Two options, candidate to choose:
- **(a) Doc-fix only (recommended).** Tighten `F12_caption.md`
  + RESULTS §6.4 to read "the y-axis (security_gain = 1 −
  compromise_rate) is identically 0.0 across all 32 candidates
  on `test_balanced` — every config still triggers IMPACT once.
  The Pareto frontier therefore reduces to the
  `always_observe` x-min point, and operating-point selection
  is one-dimensional under the Phase-3 reward function. Closing
  this requires a metric that is non-zero at `compromise_rate
  = 1.0` (e.g. `mitigated_impact_rate`, which **does** vary —
  see F9: 0.153 → 0.900)."
- **(b) Re-emit F12 with `mitigated_impact_rate` as the y-axis.**
  Cheap (~30 lines in `plot_pareto.py`); the input data already
  carries this. But this is a *re-run*, not an audit, and
  Step-7 is audit-only — should be opted in explicitly.

**Mentor recommends (a)** in Step 7 (3-line caption tightening +
RESULTS §6.4 paragraph), and proposes (b) as a Step-8 candidate
decision if the candidate wants F12 to land in the thesis.

**Evidence.**
- `docs/results/07_ablation/F12_summary.json` `points[*].security_gain` = 0.0 for all 32.
- `docs/results/07_ablation/F9_summary.json` `rows[*].compromise_rate` = 1.0 for all 12.
- `docs/results/07_ablation/F12_caption.md`.
- `docs/results/07_ablation/RESULTS.md:362–386` (§6.4).

### F8 (S, finding) — F10 PPO p=1.0 score (+2047) exceeds the Phase-6 oracle ceiling (+1624); needs an explicit "different env" disclaimer

**Symptom.** `F10_summary.json#ppo_rows[5].mean_reward = 2046.97`
(at p=1.0). The Phase-6 RESULTS.md §6.1 ceiling is +1624. A
casual reader will compare and conclude "RL beats the oracle in
Phase 7 by +423." This is true *only* under the F10 perturbed
environment where the defender de-escalates with probability 1.0
on every attacker step — a different MDP than the Phase-6
default (p=0.0). The Phase-6 +1624 ceiling is also computed at
p=0.0 (default).

**Severity.** Defendability. The F10 caption says "as a function
of `p_defender_deescalation`" but does not flag that PPO at
p=1.0 is in a strictly easier MDP than Phase-6's headline
ceiling. The R7.x risks did not pre-register this. Defense
committee will (correctly) push back if RESULTS §6.3 is read in
isolation against §6.1.

**Mentor recommendation.** Ship in Step 7. One sentence in
`F10_caption.md` and one in `RESULTS.md §6.3`: "Note: the
Phase-6 oracle ceiling +1624 is computed at the Phase-3 default
`p_defender_deescalation = 0.0`. F10's high-p cells operate in a
strictly easier environment and are not directly comparable to
that ceiling — the figure's qualitative claim is monotonicity,
not absolute level."

**Evidence.**
- `docs/results/07_ablation/F10_summary.json#ppo_rows[5]`.
- `docs/results/06_benchmark/RESULTS.md:139–142` (Phase-6
  ceiling at p=0.0 default).
- `docs/results/07_ablation/F10_caption.md`.

### F9 (Nit) — `compromise_rate = 1.0` everywhere is an honest but uncomfortable result that deserves a thesis paragraph

**Symptom.** Every F9 cell, every F12 candidate, and every
in-distribution Phase-6 anchor reports `compromise_rate = 1.0`
on `test_balanced`. Even the `impact_is_terminal=False` win
(which lifts `mitigated_impact_rate` to 0.900) does not move
`compromise_rate` off 1.0 — meaning the agent **always lets
IMPACT happen at least once**, then mitigates with a
post-IMPACT defense action under the relaxed terminal rule.

**Severity.** Defendability. The thesis claim "RL captures 82 %
of the oracle ceiling" must be paired with the truthful
"however, on the held-out balanced test split, every policy
including the oracle reaches `compromise_rate = 1.0`. The
`impact_is_terminal=False` configuration interprets that as
'mitigated 90 % of impact events' rather than 'prevented impact
in 0 % of episodes'." The F9 win is real but it lives in the
post-IMPACT mitigation regime, not the pre-IMPACT prevention
regime.

**Mentor recommendation.** One paragraph in `RESULTS.md §6.1`
acknowledging this. Optional in Step 7; nice-to-have in Step 9
(LaTeX framing).

**Evidence.**
- `docs/results/07_ablation/F9_summary.json#rows[*].compromise_rate` = 1.0.

---

## 4. Step-7 doc-fixes shipped in this branch

The following findings are small enough to land in Step 7 (and
big enough to leave-stale-and-confusing if not):

- **F1** — RESULTS §9 footnote + scoreboard `note` field.
- **F4** — F9 caption + RESULTS §1 baseline disambiguation.
- **F5** — RESULTS §8 explicit eval-contract clause.
- **F6** — RESULTS §7 MANEUVER-coupling defer-to-Phase-8 bullet.
- **F7(a)** — F12 caption + RESULTS §6.4 degenerate-y-axis
  clarification.
- **F8** — F10 caption + RESULTS §6.3 different-MDP disclaimer.

Conventional Commits used: `docs(phase-7,§1):`,
`docs(phase-7,§6.1):`, `docs(phase-7,§6.3):`,
`docs(phase-7,§6.4):`, `docs(phase-7,§7):`,
`docs(phase-7,§8):`, `docs(phase-7,§9):`,
`docs(phase-7,scoreboard):`, `docs(phase-7,F9-caption):`,
`docs(phase-7,F10-caption):`, `docs(phase-7,F12-caption):`.

Findings batched to Step 8 (cross-cutting):

- **F2** — F10 + F12 manifest pin gaps (cheap code-fix in
  `plot_aggressiveness.py` and `plot_pareto.py`); same
  reproducibility-hygiene wave as Step-6 F3 Phase-2-LSTM-SHA pin.
- **F3** — Scoreboard-schema unification; same wave as Step-4
  G4.4 + Step-5 G5.4 verdict-enum backfill. Use Phase-6's
  native `status` enum + `finding_id`, **not** `verdict`.

Finding deferred to Phase 8 / 9:

- **F6** — MANEUVER-stage-coupling reward extension (PLAN
  carries forward; not a Phase-7 deliverable).
- **F9** — `compromise_rate = 1.0` thesis-framing paragraph
  (Step-9 LaTeX framing).

---

## 5. Verification record

| Item | Method | Result |
|---|---|---|
| Test-split contract on `test_balanced` + `exclude_ood=True` | line-by-line read of `run_reward_sweep.py:273–301`, `run_aggressiveness_sweep.py`, `run_ood_eval.py`, `src/blue_team/env_factory.py` | ✅ verified end-to-end |
| OOD class list (DDoS-HTTP_Flood, Mirai-udpplain, VulnerabilityScan, XSS) | `F15_manifest.json#inputs.eval_jsonls_sha256` keys | ✅ all 4 present, 32 cells, 0 NaN |
| F15 hybrid realiser (background = train pool ID, OOD overlay at class's stage only) | `scripts/ablation/run_ood_eval.py` + RESULTS §5.1 (commit `87b80dc`) | ✅ verified |
| Hash chain — Phase-5/6/7 manifests on disk vs recorded | `shasum -a 256 runs/phase{5,6,7}/.../*.json` | ✅ all match (`cc7454…`, `c4a60a…`, `19f35074…`, `86d20b33…`) |
| Phase-1 splits SHA `1e99d596…` pinned | F9/F15 transitively via `phase6_eval_manifest`; F10/F12 not | F2 (under-pinned) |
| F9 grid: 5 components × {0.5×, 1×, 2×} + impact_is_terminal binary | `F9_summary.json#rows` (12 cells) | ✅ verified |
| F10 axis: `p_defender_deescalation ∈ {0.0, 0.2, …, 1.0}` | `F10_summary.json#p_values` | ✅ 6 points |
| F15 grid: 4 OOD × 8 policies | `F15_summary.json` | ✅ 32 cells |
| Phase-4 RF consumed by F15 (RFActingPolicy) | `run_ood_eval.py` + `F15_summary.json`'s `rf_acting` rows | ✅ |
| Phase-4 CNN1D consumed anywhere in Phase 7 | `grep -rni cnn1d scripts/ablation/` | ❌ not consumed (consistent with Phase-6 D6.5; correct) |
| Phase-7 ckpts as warm-starts | `run_reward_sweep.py` does NOT warm-start; trains fresh from scratch | (no findings — design choice) |
| `pytest -q` collection at HEAD | `pytest --collect-only -q` | 411 collected |
| `pytest -q` execution at HEAD | run | 411 / 411 passed |

---

## 6. Open candidate decisions carried (now eight)

These were already on the list at Step-7 entry; flagging again for
the Step-8 batch. None block Step-7 sign-off.

1. **Step-2 F1** — Phase-2 LSTM re-run with `seed=42` against the
   post-`3cd2fb9` manifest (option a) or document-only in a
   backfilled Phase-2 RESULTS.md (option b)? Note: option (a)
   forces a Phase-6 `eval_manifest.json` re-emit to pin the new
   `attack_sequence_generator.pth` SHA — Step-6 F3 makes this
   trivial (regenerable in seconds from on-disk JSONLs).
2. **Step-2 F2** — was balanced-val cross-entropy or macro-F1 the
   intended Phase-2 model-selection criterion?
3. **Step-3/4/5/6/7 doc-fix batching** into Step 8 — confirm
   cross-phase batch over piecemeal.
4. **Verdict-enum scoreboard schema backfill** for G4.4 + G5.4 +
   G7.x — match Phase-6-native `status` enum + `finding_id`.
   (Recommend **`status`**, not `verdict`.)
5. **Step-9 LaTeX framing** — RESULTS.md §6.1's "82 % of oracle
   ceiling" is the canonical thesis claim; older "RL beats
   baselines by 25×" must be retired. Add §6.2 "robust to, not
   better at" OOD claim.
6. **NEW (F7) — F12 y-axis remediation.** Doc-fix only (option
   a, recommended) or re-emit with `mitigated_impact_rate`
   y-axis (option b, requires explicit re-run opt-in)?
7. **NEW (F9) — `compromise_rate = 1.0` framing paragraph.**
   Author in Step 9 (LaTeX) or fold into RESULTS.md §6.1 in
   Step 8?
8. **Phase-8 vs Phase-10 routing.** Commit `8d5dd67` (`docs(handoff): rewrite for Phase-7 closeout — D2 (Phase 8 vs 10) decision required`) shows the candidate already started a Phase-10 hygiene branch in parallel (commits `f1a68f3`, `fa1a791`, `281860a`, `8c6e665`, `0a1352d`, `2deda39`, `a969fd6`) and tagged `v0.1.0`. **This means Phase-8 may have been skipped.** The Step-7 brief from the candidate said "Phase 10 release" was Step-7's downstream — confirming the skip. F6/F13/F14 deferrals therefore land in **future-work** (post-thesis), not Phase 8. Surface in §8 of HANDOFF.

---

## 7. Mentor sign-off proposal

- **Step-7 verdict.** `PASS-WITH-FIXES`.
- **Step-7 doc-fixes shipping in this branch.** F1 + F4 + F5 +
  F6 + F7(a) + F8 + the §6.1 + §9 footnotes.
- **Step-7 findings batched to Step 8.** F2 + F3 (cross-cutting
  cleanup wave).
- **Step-7 findings deferred to Phase 8 / 9 / future work.** F6
  (MANEUVER coupling) + F9 (compromise-rate thesis paragraph).
- **Branch.** `mentor-review/step-7-ablation` off `main` at
  `1d78fec`; pushed; awaiting candidate sign-off; G2 deferred
  to next session.

When the candidate signs off, Step 8 becomes the cross-cutting
cleanup wave: Step-3/4/5/6/7 doc-fix batch + scoreboard-schema
unification + Step-2 F1/F2 + Step-6 F3 (Phase-2-LSTM-SHA pin) +
Phase-7 F2/F3.
