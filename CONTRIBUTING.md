# Contributing to rl-iot-defense-system

Thanks for your interest in contributing! This is an MSc-thesis research
codebase for adversarial-RL IoT defense, so a few conventions keep the
empirical results reproducible.

## Setup

```bash
git clone https://github.com/feli-santos/rl-iot-defense-system.git
cd rl-iot-defense-system
python3.9 -m venv .venv
source .venv/bin/activate
make install-dev      # runtime + dev deps + pre-commit hooks
```

The **Makefile is the source of truth for commands** — run `make help` for
the canonical target list. Trust it over the README if they ever disagree.

## The format → lint → test order

Before pushing, run:

```bash
make format          # black + ruff --fix + isort
make lint            # ruff check + black --check
make test            # pytest -q (452 tests, ~18s, synthetic-only by default)
```

CI runs `make lint` then `pytest -q --cov` on Python 3.9. A PR that fails
either will not merge.

## Where things live

- `src/` — importable library code (attacker, environment, detector,
  algorithms, benchmark, blue_team, utils).
- `scripts/` — runners, plotters, and thesis tooling.
- `tests/` — pytest suite (synthetic by default; real-data tests auto-skip
  unless `data/processed/ciciot2023/` exists).
- `docs/results/` — canonical experiment outputs; each figure ships a sibling
  `manifest.json` (git SHA + I/O SHA-256).
- `tex/` — LaTeX thesis (built with Podman via `make thesis`).
- `config.yml` — dataset processing config (live); RL hparams in `train_agent.py` DEFAULT_HPARAMS + Makefile.

See `AGENTS.md` for the full architecture map, locked experiment contracts,
and gotchas. **Read `AGENTS.md` before touching the environment, reward, or
training code** — several decisions are fixed thesis contracts.

## Smoke targets (fast canaries for env/config drift)

After touching the env, reward, or training code, run:

```bash
make blue-team-smoke      # ~20s
make benchmark-smoke
make ablation-ood-smoke
```

If a smoke fails on a clean checkout, treat it as a bug.

## Reproducibility

Every thesis figure under `docs/results/<area>/` is pinned by a manifest.
Verify nothing drifted:

```bash
python -m scripts.reproducibility_smoke           # verify all manifests
python -m scripts.reproducibility_smoke --strict  # exit 1 on any hash miss
```

## Commit conventions

- Pre-commit hooks (ruff, ruff-format, black, isort) run on commit;
  `make install-dev` installs them.
- Keep commit messages concise and imperative.
- **Never commit** `data/`, `runs/`, `artifacts/`, `mlruns/`, or `results/` —
  they are gitignored and machine-local. Raw CICIoT2023 CSVs are **not** in the
  repo (CIC license); they go in `data/raw/ciciot2023/`.
- Bump `docs/results/test_count.json` when the suite count changes (it feeds
  the thesis `\NumTests` macro).

## Reporting bugs

Open a GitHub issue with: OS, Python version, `make help` output, and the
exact command + full traceback. Run `python -m scripts.reproducibility_smoke`
first and include the verdict.
