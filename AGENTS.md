# AGENTS.md

Adversarial-RL IoT defense research codebase (MSc thesis). Python, CPU-first.
`src/` is importable library code; `scripts/` holds runners; `main.py` is a
thin CLI. Use the **Makefile as the source of truth for commands**.

## Commands

```bash
make help              # canonical target list (always trust this over README)
make test              # pytest -q  (this is what CI runs)
make test-cov          # pytest with coverage
make lint              # ruff check + black --check on src tests scripts main.py
make format            # black + ruff --fix + isort (run before committing)
pytest tests/test_adversarial_env.py -q          # single file
pytest tests/test_adversarial_env.py::TestX::test_y   # single test
```

Order that matters for commits/CI: **format → lint → test**. CI runs `ruff
check`, `black --check`, then `pytest -q --cov` on Python 3.9.
Pre-commit hooks (ruff, ruff-format, black, isort) run on commit; `make
install-dev` installs them into `.venv` via `requirements-dev.txt` (versions
pinned to match `.pre-commit-config.yaml`). The `lint`/`format` Makefile
targets invoke the tools as `$(PYTHON) -m ruff|black|isort` so they always
resolve through `.venv` — no system-level install is needed.

## Architecture

Adversarial loop: a reactive **tug-of-war attacker** (`MarkovAttacker`,
`src/generator/`) walks the kill chain. Escalation pressure is **coupled to the
attacker's proximity to the impact stage** (no fixed intrusion budget). It
reacts to the defender's force via a signed proportionality rule on
`d = action - recommended(stage)`: proportionate (`d==0`) de-escalates one
stage (p_down=0.90; ISOLATE 0.98), under-forced (`d<=-1`) advances (p_up=0.90),
over-forced (`d>=1`) holds. The attacker emits a stage → `RealizationEngine`
(`src/utils/`) samples a **session-coherent, aliasing-aware** CICIoT2023
feature row for that stage → `AdversarialIoTEnv` (`src/environment/`,
Gymnasium API, 290-dim windowed obs, 5 actions) → **Blue Team** SB3 agent
(DQN/PPO/A2C via `src/algorithms/`) acts → kill-chain reward. The defender
**never observes the true stage** (POMDP = central thesis).

Kill-chain stages (5): `0 BENIGN, 1 RECON, 2 ACCESS, 3 MANEUVER, 4 IMPACT`
(canonical map in `tests/conftest.py`).

`src/` modules: `generator/`=attacker, `environment/`=Gymnasium env,
`detector/`=supervised stage detector, `blue_team/`=env factory + run config,
`benchmark/`=baselines + eval runner, `algorithms/`=SB3 wrappers,
`utils/`=dataset + realization engine.

**Canonical blue-team checkpoints (current contract): `runs/redesign_5M_det/alpha_<NN>/<algo>/seed_<n>/best_model.zip`** (`alpha_00/02/04/06/08/10`; headline = `alpha_04`,
matching HeadlineAlpha=0.4). The legacy `runs/blue_team/` path is **pre-redesign
and absent on fresh checkouts** — do NOT point eval at it. The OOD eval target
`make ablation-ood-eval` is wired to `ABLATION_OOD_BLUE_TEAM_RUNS ?=
runs/redesign_5M_det/alpha_04` and passes the partial-observability flags
(`--aliasing-rate 0.4 --session-coherent --no-post-transition-leak
--proximity-coupled --proximity-min-escalation 0.4`) via
`ABLATION_OOD_POMDP_FLAGS`; these flags are **mandatory** — omitting them makes
`run_ood_eval.py` fall back to a fully-observable MDP (aliasing=0, no session
coherence) that mismatches the trained checkpoints and trips the train/eval
parity assertion.

## Conventions / gotchas

- **`config.yml` `dataset:` section is live** (consumed by `main.py
  process-data`); all RL/env/reward sections are **documentation-only** (the
  training pipeline uses `DEFAULT_HPARAMS` in `scripts/blue_team/train_agent.py`
  + Makefile overrides + dataclass defaults in `run_config.py`). There are many
  `.get(key, default)` fallbacks, so a missing key silently uses the code
  default — grep both places when changing a param.
- **Reproducibility = hash chains.** Every thesis figure under
  `docs/results/<area>/` ships a sibling `manifest.json` (git SHA + input/output
  SHA-256). Verify via `python -m scripts.reproducibility_smoke` (`--strict` to
  exit 1 on any hash miss).
- **Gitignored / machine-local:** `data/`, `runs/`, `artifacts/`, `mlruns/`,
  `results/`. Raw CICIoT2023 CSVs are NOT in the repo (CIC license); they go in
  `data/raw/ciciot2023/`.
- Lint tolerates scientific naming (`X_train`, `y_test`) and SB3/Gym
  default-arg patterns; `data/ mlruns/ artifacts/ runs/ results/ notebooks/`
  are excluded from all tooling.
- `notebooks/` is exploratory and NOT on the thesis reproduction path.

## Testing notes

- Default run is **synthetic-only**. Real-data tests are guarded with
  `@pytest.mark.skipif(not (data/processed/ciciot2023/...).exists())` — they
  auto-skip on a fresh checkout. Markers: `slow`, `integration`, `gpu`
  (registered in `pyproject.toml`).
- Pipeline smoke targets are the canary for env/config drift; run them after
  touching the env, reward, or training code: `make blue-team-smoke` (~20s),
  `make benchmark-smoke`, `make ablation-ood-smoke`. If a smoke fails on a
  clean checkout, treat it as a bug.
- Full sweeps are expensive and CPU-bound (hours): `make blue-team`, `make
  benchmark`, `make ablation`. Don't run these to "verify" a code change — use
  smoke targets.

## Thesis (separate toolchain)

LaTeX under `tex/` builds with **Podman** (Docker is a fallback): `make thesis`
(wraps `bash tex/build.sh`). Main file is `tex/main.tex` (abnTeX2/FEEC
template), **NOT** `thesis.tex`. Output is `tex/main.pdf`.

**Numbers in the thesis are macro-driven, never hand-typed.** Canonical
experiment JSONs live under `docs/results/<area>/`. `make render-tables`
(`scripts/thesis/render_tables.py`) regenerates `tex/generated/{numbers,tables}.tex`
from them; `tex/generated/` is **gitignored** (rebuild before any thesis
edit/build). `docs/results/test_count.json` feeds the `\NumTests` macro; bump
it when the suite count changes.

**LaTeX gotcha:** macro names cannot contain digits — a digit ends the
control-sequence name. Use spelled-out forms (`\FnineStructuralReward` not
`\F9...`). `render_tables.py` already maps digits to letters when emitting
macro names; keep it that way.

**Thesis prose rule:** NEVER write "Phase N" in `tex/` prose — use semantic
stage names (Markov Attacker / Adversarial Environment / Stage Detector /
Blue-Team Training / Held-Out Benchmark / Ablation & Robustness).

## Journal paper (`paper/`, separate deliverable)

A condensed journal version of the thesis targets **Elsevier _Internet of
Things_ (ISSN 2542-6605)** and lives under `paper/` (added in `v0.8.1`; submitted
with the snapshot published as release `v0.8.4`). It is a **separate artefact
from the thesis** — the thesis under `tex/` remains the comprehensive record;
`paper/` is the submission package.

- **Template:** Elsevier `elsarticle`, double-column
  (`\documentclass[3p,twocolumn]{elsarticle}`), `\bibliographystyle{elsarticle-num}`.
  Root file is `paper/manuscript.tex`.
- **HARD LIMITS (never violate; verify after every build):**
  - **Compiled `manuscript.pdf` ≤ 10 pages.** This is a hard cap — do not
    exceed it. When edits push over 10 pages, reclaim space by trimming filler
    prose, tightening floats, and/or shrinking the full-width architecture
    figure (`fig:architecture`); as a last resort strip redundant `doi = {...}`
    fields from `refs.bib` (keep every reference and every reported number).
    Never drop cited references to fit. If still >10 pages after reasonable
    trimming, STOP and hand the build command back rather than mangling content.
  - **Abstract < 250 words** (Elsevier guide ceiling; current abstract ≈234).
    Never let it reach 250. Recount after any abstract edit:
    `.venv/bin/python -c "import re,sys; ..."` or a simple word split on the
    abstract block.
  - Page-count check (fitz only works in `.venv`, not system python):
    `.venv/bin/python -c "import fitz; print(fitz.open('paper/manuscript.pdf').page_count)"`
    (must print `<= 10`). `pdfinfo paper/manuscript.pdf` also works.
- **Build:** `make -C paper build` (runs `numbers` → `pdflatex` → `bibtex` →
  `pdflatex`×2, copies `manuscript.pdf` → `paper/build/`). Same **Podman**
  container as the thesis (`localhost/rl-iot-thesis:latest`, which already ships
  `elsarticle.cls` + `elsarticle-num.bst`); the host has no TeX. Other targets:
  `draft`, `numbers`, `wordcount`, `clean`, `verify`.
- **Float placement convention:** single-column in-text figures/tables use
  `[ht]` (or `[h]` for `tab:hparams`) so they sit near their first mention;
  full-width floats (`figure*`/`table*`: `fig:architecture`, `fig:ood`,
  `fig:actions`, `tab:related`) MUST stay `[tbp]`/`[t]` — LaTeX forbids `h` on
  starred floats.
- **Validated model-footprint numbers are macro-driven too.**
  `scripts/benchmark/compute_model_footprint.py` loads the three
  `runs/redesign_5M_det/alpha_04/{ppo,a2c,dqn}/seed_0/best_model.zip` policies +
  the tuned RF joblib and emits `docs/results/benchmark/model_footprint.json`
  (+ sibling `model_footprint_manifest.json` hash chain).
  `scripts/thesis/render_tables.py::_render_footprint_numbers()` turns that into
  `\PolicyFootprintKB` (90), `\PolicyParams` (23K), `\RFDetectorMB` (181),
  `\RFDetectorNodes` (1.7M), `\FootprintRatio` (1956). The paper's edge-footprint
  claim (deployable policy ≈90 KB fp32 / 23.1K params vs ≈181 MB tuned RF) is
  these macros — never hand-type it. **Do NOT resurrect the old, wrong "≈20 KB"
  or "22 MB / 2.6M-param" figures.**
- **Numbers are macro-driven, same as the thesis.** `paper/numbers.tex` is a
  copy of `tex/generated/numbers.tex`; regenerate the source with `make
  render-tables` (or `PYTHONPATH=. .venv/bin/python -m
  scripts.thesis.render_tables`) then re-copy. The digit-free macro rule applies
  (`ATwoC`=A2C, `Ften`=F10, `Fseventeen`=F17, `Ffifteen`=F15).
- **Figures** in `paper/figs/` are vendored vector PDFs copied from `tex/figs/`.
- **Authors:** Felipe Santos (first + corresponding,
  `f233292@dac.unicamp.br`) + Denis Fantinato (`denisf@unicamp.br`); no other
  co-authors.
- **Guide-mandated side files** (Elsevier IoT guide-for-authors):
  `paper/highlights.tex` (3–5 bullets ≤85 chars), `paper/cover-letter.md`, and
  `paper/declarations/` (`credit-statement.md`, `competing-interests.md` +
  `competing-interests.docx` uploaded separately, `funding.md`,
  `genai-declaration.md`, `data-availability.md`). CRediT + competing-interest +
  funding + data-availability + GenAI declaration are also emitted as unnumbered
  sections at the end of `manuscript.tex`, with the GenAI declaration the last
  section directly before the references. (Acknowledgements were intentionally
  removed \u2014 the guide only requires that IF present they sit directly before the
  references; they are not mandatory.)
- **Data availability = Elsevier Research-Data Option C:** raw CICIoT2023 is
  **not** redistributed (CIC license) and is cited as a `[dataset]` reference
  (`CICIoT2023Dataset` in `paper/refs.bib`); code + hash-chain manifests are
  deposited via a **public GitHub release**
  (`github.com/feli-santos/rl-iot-defense-system`). The submission cites release
  **`v0.8.4`** (the paper submission snapshot) in
  `paper/manuscript.tex` (Data availability) + `paper/declarations/data-availability.md`;
  that tag is **published** and is the stable, reproducible reference for the
  manuscript — do NOT bump it when later thesis-only releases are cut (the thesis
  is now at `v0.8.5`). **A Zenodo DOI is NOT required** for Option C \u2014 the
  public GitHub release plus the `[dataset]` citation already satisfy it. (`paper/`
  is Zenodo-free; do not reintroduce any `zenodo.XXXXXXX` placeholders.)
- **Scope decision:** condense-and-submit with **current** results — do NOT add
  a second dataset or a windowed-supervised baseline before first submission
  (both are named follow-ups only if a reviewer demands them).
- **Pre-submission checklist** (Elsevier _Internet of Things_ guide; keep this
  in AGENTS.md, NOT in `paper/README.md` which stays readme-only):
  - Title page with full affiliation + corresponding-author contact — in
    `manuscript.tex` (UNICAMP/FEEC address, American-journal English style with
    the "nº" ordinal dropped because elsarticle `\affiliation{addressline=...}`
    fatally breaks on superscript/fragile macros: `Avenida Albert Einstein 901,
    Cidade Universitária Zeferino Vaz, Barão Geraldo, Campinas, SP, 13083-852,
    Brazil`).
  - Compiled PDF ≤ 10 pages (hard limit above); abstract < 250 words (≈243,
    purely textual — no explicit α/math per the abstract-style decision).
  - Keywords 1–7 short indexing terms (currently 7).
  - Highlights file 3–5 bullets ≤85 chars — `highlights.tex`.
  - CRediT statement — manuscript + `declarations/credit-statement.md`.
  - GenAI-use declaration (section before references) — in manuscript.
  - Competing-interest declaration (separate `.docx`) —
    `declarations/competing-interests.docx`.
  - Funding statement — manuscript + `declarations/funding.md`.
  - Data-availability statement (Option C) — manuscript +
    `declarations/data-availability.md`.
  - Editable `.tex` source + figures as separate files — `figs/`.
  - Public tagged GitHub release **`v0.8.4`** published and cited in Data
    availability (done). Zenodo DOI NOT required for Option C.
  - Author emails filled (`f233292@dac.unicamp.br`, `denisf@unicamp.br`).
  - Optional (not blocking): graphical abstract, SSRN preprint,
    MethodsX/Data-in-Brief co-submission.

## Locked experiment decisions

These are fixed contracts; do not silently change them.

- **Primary reward contract: `reward_mode=outcome`** (sparse, outcome-only) for
  training + benchmark. The `coupled` (proportionality-shaped) reward is used
  only in the reward-coupling ablation. `impact_is_terminal=False` for training
  + benchmark; `True` retained only as a reward-mis-specification case study.
- **10 seeds `{0..9}`** for DRL; baselines/oracle run 1 seed. **n=300 episodes
  for ALL policies.** Tug-of-war probabilities: `p_down=0.90` (ISOLATE 0.98) /
  `p_up=0.90`; BENIGN onset `p_onset=0.35`, `p_onset_access=0.10`.
  Proximity-coupled escalation: `sigma_min=0.4`,
  `p_up_eff = p_up * (sigma_min + (1-sigma_min) * lambda)`, `lambda = stage/4`.
  Prevention bonus `b_prevent = +50`.
- **Canonical headline numbers (deterministic-5M regime)** (from
  `docs/results/ablation/Falpha_summary.json`): PPO flat +138.6→+113.3 vs tuned
  RF +136.5→+64.0 across α=0.0/0.2/0.4/0.6 (RF continues to −29.3 at α=1.0);
  tie at α=0 (overlapping CIs), disjoint CIs from α=0.4 (PPO +121.3 vs RF +94.4,
  sig +26.9). Oracle ceiling +194.8.
  Reward-coupling: coupled best DQN +226.2 (gap −63.1 vs RF +163.1), outcome
  best A2C +146.1 (gap −63.0 vs RF +83.1; DQN −8.6). **On-policy advantage is
  training reliability and defensive doctrine**: at α=0.4 all three clear the
  negative regime on best checkpoint (DQN +72.5, PPO +121.3, A2C +138.7) and
  A2C matches or exceeds PPO at every aliasing rate; across-seed sd PPO≈15 /
  DQN≈52 / A2C≈9; the same DQN's sd inflates ≈17→≈52 switching coupled→outcome.
  OOD (10 held-out zero-day classes): best RL (A2C) prevents 0.71–0.85 vs RF
  0.00–0.15 on every class, with **no detectable dependence** of RL advantage
  on detector recall (Spearman ρ=0.22 p=0.54, Pearson r=−0.02 p=0.95, OLS slope
  CI spans zero).
  F10 (aggressiveness) and F17 (evasion) load the fixed det-5M α=0.4 PPO and
  evaluate across the swept knob (no retraining); F17 uses the **evasive-
  persistence** (post-detection hardening) attacker coupling. Test suite:
  **462 passed**.
