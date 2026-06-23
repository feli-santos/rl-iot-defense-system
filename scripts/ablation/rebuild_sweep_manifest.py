"""Rebuild F10/F17 ``sweep_manifest.json`` comprehensively from on-disk cells.

The single-cell repair re-runs (``--p-values 0.0 0.2`` / ``--evasion-values
0.25``) overwrite ``sweep_manifest.json`` with ONLY the cells they processed,
truncating the manifest's value list and ``runs`` array. The figure plotters
read the seed dirs directly (not the manifest) for their data, so figure values
are unaffected — but the reproducibility hash chain hashes this manifest, so it
must list every cell/seed actually present on disk.

This script scans the canonical run roots and rebuilds the full manifest from
whatever seed dirs exist, recomputing ``model_sha256`` and
``test_eval_jsonl_sha256`` so the hash chain is honest.

Usage::

    python -m scripts.ablation.rebuild_sweep_manifest --figure f10
    python -m scripts.ablation.rebuild_sweep_manifest --figure f17
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]

_F10_P_VALUES = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
_F17_EVASION_VALUES = [0.0, 0.25, 0.5, 0.75]
_SEEDS = list(range(10))

# F15 OOD layout. Canonical class order matches run_ood_eval._OOD_STAGE_BY_CLASS.
_F15_OOD_CLASSES = [
    "VulnerabilityScan",
    "Recon-OSScan",
    "XSS",
    "SqlInjection",
    "Mirai-udpplain",
    "DNS_Spoofing",
    "DDoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DDoS-SlowLoris",
    "DDoS-ACK_Fragmentation",
]
_F15_POLICIES = [
    "recommended_action",
    "rf_acting",
    "dqn",
    "ppo",
    "a2c",
    "random",
    "always_observe",
    "always_block",
]
_F15_RL_ALGOS = {"dqn", "ppo", "a2c"}
_F15_DETERMINISTIC = {
    "recommended_action",
    "rf_acting",
    "always_observe",
    "always_block",
}
_F15_RL_SEEDS = [0, 1, 2, 3, 4]  # RL + random use seeds 0-4; deterministic use seed 0


def _sha256(path: Path) -> str | None:
    p = Path(path)
    if not p.exists():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=_ROOT,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        return "unknown"


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r") as fh:
        return sum(1 for _ in fh)


def _slug(v: float) -> str:
    return f"{v:.2f}".replace(".", "p")


def _f17_slug(e: float) -> str:
    return f"{e:.2f}".replace(".", "p")


def _f10_slug(p: float) -> str:
    # F10 cells use one-decimal slug (ppo_p0p0, ppo_p0p2, ... ppo_p1p0).
    return f"{p:.1f}".replace(".", "p")


def _ppo_run_entry_f17(root: Path, e: float, seed: int) -> dict[str, Any] | None:
    out_dir = root / f"ppo_e{_f17_slug(e)}" / f"seed_{seed}"
    if not out_dir.exists():
        return None
    eval_jsonl = out_dir / "eval_test.jsonl"
    model = out_dir / "model.zip"
    return {
        "kind": "ppo",
        "evasion_prob": e,
        "seed": seed,
        "ok_train": model.exists(),
        "ok_test_eval": _count_lines(eval_jsonl) > 2,
        "out_dir": str(out_dir),
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "test_eval_jsonl": str(eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl),
        "test_eval_n_episodes": _count_lines(eval_jsonl),
    }


def _ppo_run_entry_f10(root: Path, p: float, seed: int) -> dict[str, Any] | None:
    out_dir = root / f"ppo_p{_f10_slug(p)}" / f"seed_{seed}"
    if not out_dir.exists():
        return None
    eval_jsonl = out_dir / "eval_test.jsonl"
    model = out_dir / "model.zip"
    return {
        "kind": "ppo",
        "p_down": p,
        "seed": seed,
        "ok_train": model.exists(),
        "ok_test_eval": _count_lines(eval_jsonl) > 2,
        "out_dir": str(out_dir),
        "model_path": str(model),
        "model_sha256": _sha256(model),
        "test_eval_jsonl": str(eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl),
        "test_eval_n_episodes": _count_lines(eval_jsonl),
    }


def _rule_run_entry_f10(root: Path, p: float) -> dict[str, Any] | None:
    out_dir = root / f"rule_p{_f10_slug(p)}" / "seed_0"
    eval_jsonl = out_dir / "eval_test.jsonl"
    if not eval_jsonl.exists():
        return None
    return {
        "kind": "rule",
        "p_down": p,
        "seed": 0,
        "ok": _count_lines(eval_jsonl) > 2,
        "test_eval_jsonl": str(eval_jsonl),
        "test_eval_jsonl_sha256": _sha256(eval_jsonl),
        "test_eval_n_episodes": _count_lines(eval_jsonl),
    }


def rebuild_f10(root: Path) -> dict[str, Any]:
    ppo_runs: list[dict[str, Any]] = []
    rule_runs: list[dict[str, Any]] = []
    for p in _F10_P_VALUES:
        for seed in _SEEDS:
            entry = _ppo_run_entry_f10(root, p, seed)
            if entry is not None:
                ppo_runs.append(entry)
        rule = _rule_run_entry_f10(root, p)
        if rule is not None:
            rule_runs.append(rule)
    return {
        "schema_version": "1.0",
        "stage": "ablation",
        "kind": "f10_aggressiveness_sweep_manifest",
        "git_sha": _git_sha(),
        "rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_from_disk": True,
        "p_values": list(_F10_P_VALUES),
        "seeds": list(_SEEDS),
        "ppo_runs": ppo_runs,
        "rule_runs": rule_runs,
        "n_ppo_ok": sum(1 for r in ppo_runs if r["ok_train"] and r["ok_test_eval"]),
        "n_ppo_failed": sum(1 for r in ppo_runs if not (r["ok_train"] and r["ok_test_eval"])),
    }


def rebuild_f17(root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for e in _F17_EVASION_VALUES:
        for seed in _SEEDS:
            entry = _ppo_run_entry_f17(root, e, seed)
            if entry is not None:
                runs.append(entry)
    n_ok = sum(1 for r in runs if r["ok_train"] and r["ok_test_eval"])
    return {
        "schema_version": "1.0",
        "figure": "F17",
        "git_sha": _git_sha(),
        "rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_from_disk": True,
        "evasion_values": list(_F17_EVASION_VALUES),
        "seeds": list(_SEEDS),
        "total_timesteps": 1_500_000,
        "n_episodes": 300,
        "n_ok": n_ok,
        "n_fail": len(runs) - n_ok,
        "runs": runs,
    }


def _steps_total(path: Path) -> int:
    """Sum ``episode_length`` over an eval_test.jsonl (best-effort, 0 if absent)."""
    if not path.exists():
        return 0
    total = 0
    with path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                total += int(json.loads(line).get("episode_length", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                continue
    return total


def _f15_cell(root: Path, ood_class: str, policy: str, seed: int) -> dict[str, Any] | None:
    """Rebuild one F15 manifest ``runs[]`` entry from a seed dir on disk.

    Mirrors the schema emitted by ``run_ood_eval`` (RL cells ``kind=trained``;
    everything else ``kind=baseline``). The per-cell prevention/compromise stats
    are computed by the plotter from the jsonl, NOT stored here; ``run_policy``
    only contributes the three episode counters, which we recompute from disk.
    """
    out_dir = root / ood_class / policy / f"seed_{seed}"
    eval_jsonl = out_dir / "eval_test.jsonl"
    if not eval_jsonl.exists():
        return None
    n_ep = _count_lines(eval_jsonl)
    run_id = f"f15_{ood_class}_{policy}_seed_{seed}"
    base: dict[str, Any] = {
        "ood_class": ood_class,
        "seed": seed,
        "run_id": run_id,
        "ok": n_ep > 2,
        "eval_jsonl": str(eval_jsonl),
        "eval_jsonl_sha256": _sha256(eval_jsonl),
        "n_episodes_written": n_ep,
        "n_steps_total": _steps_total(eval_jsonl),
        "n_latency_rows": 0,
        "rebuilt_from_disk": True,
    }
    if policy in _F15_RL_ALGOS:
        base["kind"] = "trained"
        base["algo"] = policy
    else:
        base["kind"] = "baseline"
        base["policy"] = policy
    return base


def rebuild_f15(root: Path) -> dict[str, Any]:
    runs: list[dict[str, Any]] = []
    for ood_class in _F15_OOD_CLASSES:
        for policy in _F15_POLICIES:
            if policy in _F15_DETERMINISTIC:
                entry = _f15_cell(root, ood_class, policy, 0)
                if entry is not None:
                    runs.append(entry)
            else:  # RL algos + random: seeds 0-4
                for seed in _F15_RL_SEEDS:
                    entry = _f15_cell(root, ood_class, policy, seed)
                    if entry is not None:
                        runs.append(entry)
    n_ok = sum(1 for r in runs if r["ok"])
    return {
        "schema_version": "1.0",
        "stage": "ablation",
        "kind": "f15_ood_eval_manifest",
        "audit_finding": "AF1 — promote OOD-class robustness to Tier-1 "
        "deliverable (2026-04-30 mentor audit).",
        "git_sha": _git_sha(),
        "rebuilt_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rebuilt_from_disk": True,
        "ood_classes": list(_F15_OOD_CLASSES),
        "policies": list(_F15_POLICIES),
        "runs": runs,
        "n_ok": n_ok,
        "n_failed": len(runs) - n_ok,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--figure", choices=["f10", "f17", "f15"], required=True)
    ap.add_argument("--root", default=None, help="override run root")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if args.figure == "f10":
        root = Path(args.root) if args.root else _ROOT / "runs/ablation/aggressiveness"
        manifest = rebuild_f10(root)
        n_total = len(manifest["ppo_runs"])
        summary = (
            f"F10: {n_total} ppo runs ({manifest['n_ppo_ok']} ok), "
            f"{len(manifest['rule_runs'])} rule cells, "
            f"p_values={manifest['p_values']}"
        )
    elif args.figure == "f17":
        root = Path(args.root) if args.root else _ROOT / "runs/ablation/evasion"
        manifest = rebuild_f17(root)
        summary = (
            f"F17: {len(manifest['runs'])} runs ({manifest['n_ok']} ok), "
            f"evasion_values={manifest['evasion_values']}"
        )
    else:  # f15
        root = Path(args.root) if args.root else _ROOT / "runs/ablation/ood"
        manifest = rebuild_f15(root)
        summary = (
            f"F15: {len(manifest['runs'])} runs ({manifest['n_ok']} ok), "
            f"{len(manifest['ood_classes'])} ood_classes x {len(manifest['policies'])} policies"
        )

    out_name = "eval_manifest.json" if args.figure == "f15" else "sweep_manifest.json"
    out_path = root / out_name
    print(summary)
    if args.dry_run:
        print(f"[dry-run] would write {out_path}")
        return 0
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
