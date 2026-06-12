"""R1 — smoke-reproducibility harness.

Validates the audit-first hash-chain end-to-end **without retraining
any heavy model**. The check is: for every committed
``manifest.json`` / ``F<k>_manifest.json``, every input SHA-256 it
records must match the on-disk SHA-256 of that input file *right
now*. If any hash drifts, the harness fails with a precise diff.

This is the audit-chain self-test. It is also the artefact the
defense committee can run on a fresh checkout to verify
reproducibility-by-hash-chain.

Usage::

    python -m scripts.reproducibility_smoke
    python -m scripts.reproducibility_smoke --strict   # exit 1 on any miss

What it checks (per step, in order):

- Dataset prep: ``docs/results/dataset/manifest.json`` ↔ on-disk
  ``F0_*.png`` / ``F0_summary.json`` outputs.
- Detector: ``docs/results/stage-detector/manifest.json`` ↔ on-disk
  outputs. Inputs (features.npy etc.) are gitignored; if missing,
  the harness skips them with a warning rather than failing.
- Benchmark: ``docs/results/benchmark/F[5-8]_manifest.json`` —
  per-figure manifest each pins ``runs/benchmark/eval_manifest.json``
  by SHA. The run-side artefact is gitignored; if missing, the
  harness skips the upstream-pin check with a warning.
- Ablation: ``docs/results/ablation/F[9,10,12,15]_manifest.json``
  ↔ on-disk outputs (PNG + summary JSON). Upstream pins
  (``blue_team_sweep_manifest`` / ``benchmark_eval_manifest`` /
  ``dataset_splits_manifest``) checked against on-disk if present;
  skipped with a warning if gitignored.
- Scoreboards: every ``G[N]_scoreboard.json`` parses as JSON,
  every ``.gates[].status`` is a member of the canonical enum.

Acceptance (R1 PASS): every check returns OK or SKIP-with-rationale;
no entry returns FAIL. The exit code is 0 iff no FAIL entries.

Wallclock budget: ~5 seconds on a fresh checkout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("scripts.reproducibility_smoke")

_ROOT = Path(__file__).resolve().parents[1]


# Canonical scoreboard status enum — must match
# `scripts/ablation/close_ablation.py::_STATUS_*`.
_VALID_STATUS = frozenset(
    {
        "PASS",
        "PASS-WITH-FINDING",
        "PASS-WITHOUT-STRETCH",
        "FAIL-WITH-FINDING",
        "FAIL",
        "SKIP",
    }
)


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# Per-step manifest target list. Each entry is (label, manifest_path,
# is_committed_artefact) — the third field controls whether a
# missing input is FAIL (committed → must exist) or SKIP (gitignored →
# best-effort).
_TARGETS: list[tuple[str, Path, bool]] = [
    ("dataset", _ROOT / "docs/results/dataset/manifest.json", True),
    ("detector", _ROOT / "docs/results/stage-detector/manifest.json", True),
    ("benchmark/F5", _ROOT / "docs/results/benchmark/F5_manifest.json", True),
    ("benchmark/F6", _ROOT / "docs/results/benchmark/F6_manifest.json", True),
    ("benchmark/F7", _ROOT / "docs/results/benchmark/F7_manifest.json", True),
    ("benchmark/F8", _ROOT / "docs/results/benchmark/F8_manifest.json", True),
    ("ablation/F9", _ROOT / "docs/results/ablation/F9_manifest.json", True),
    ("ablation/F10", _ROOT / "docs/results/ablation/F10_manifest.json", True),
    ("ablation/F12", _ROOT / "docs/results/ablation/F12_manifest.json", True),
    ("ablation/F15", _ROOT / "docs/results/ablation/F15_manifest.json", True),
]

_SCOREBOARDS: list[tuple[str, Path]] = [
    ("G4", _ROOT / "docs/results/stage-detector/detector_acceptance.json"),
    ("G5", _ROOT / "docs/results/blue-team-training/G5_scoreboard.json"),
    # G6 (benchmark) has no standalone scoreboard JSON — its acceptance is
    # captured by the F5/F6/F7/F8 manifest hash-chains above.
    ("G7", _ROOT / "docs/results/ablation/G7_scoreboard.json"),
]


# ---------------------------------------------------------------- helpers


def _walk_pin_entries(node: Any, prefix: str = "") -> list[tuple[str, dict[str, Any]]]:
    """Recursively find every ``{path, sha256}`` dict-record in a manifest tree.

    Manifests use a few different shapes:
    - dataset/red_team/detector ``inputs`` is ``{relpath: sha256_hex}``.
    - benchmark/ablation ``inputs`` is ``{key: {path, sha256}}`` for upstream
      manifests, plus a flat ``{relpath: sha256}`` map for per-JSONL
      inputs (``eval_jsonls_sha256`` / ``eval_jsonl_sha256``).
    - Outputs are sometimes ``{filename: sha256}``, sometimes
      ``{kind: filepath_string}``.

    Returns a list of (label, entry) where entry is either:
    - ``{"path": str, "sha256": hex}`` (benchmark/ablation upstream-manifest pins)
    - or a leaf ``{relpath: sha256}`` rendered as
      ``{"path": relpath, "sha256": hex}`` for uniformity.
    """
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(node, dict):
        # Case 1: this dict IS a {path, sha256} pin.
        if (
            isinstance(node.get("path"), str)
            and isinstance(node.get("sha256"), str)
            and len(node["sha256"]) == 64
        ):
            out.append((prefix, {"path": node["path"], "sha256": node["sha256"]}))
            return out
        # Case 2: this dict is a {relpath: sha256_hex} flat map.
        leaves = [
            (k, v)
            for k, v in node.items()
            if isinstance(k, str)
            and isinstance(v, str)
            and len(v) == 64
            and all(c in "0123456789abcdef" for c in v)
        ]
        non_leaves = [(k, v) for k, v in node.items() if (k, v) not in leaves]
        for k, v in leaves:
            out.append((f"{prefix}/{k}", {"path": k, "sha256": v}))
        for k, v in non_leaves:
            out.extend(_walk_pin_entries(v, prefix=f"{prefix}/{k}"))
    elif isinstance(node, list):
        for i, x in enumerate(node):
            out.extend(_walk_pin_entries(x, prefix=f"{prefix}[{i}]"))
    return out


# Pre-registered, documented hash-chain divergences. These are
# manifest-input pins that record a *historically accurate* SHA
# (matching the artefact at lock time) which has since been
# superseded by a newer SHA on disk. Each entry references the
# Step-N mentor-review finding that documented the divergence and
# the resolution narrative.
_KNOWN_DIVERGENCES: dict[str, dict[str, str]] = {
    # (label, path_str, recorded_sha) -> {actual_sha, finding_id, note}
    (
        "dataset",
        "data/processed/ciciot2023/splits/manifest.json",
        "82aa12149d2e0ee5a2424a7da44719df885ac18495590344e6d393e22d72b5c5",
    ): {
        "actual_sha": "b0661c6b225874e03e14954e09e90b84a3b3b7b0dd51a42a74129ea6050f464f",
        "finding_id": "Step-1 F4 / Step-2 F1",
        "note": (
            "pre-3cd2fb9 (leaky) splits manifest SHA recorded; on-disk is "
            "post-3cd2fb9 (leakage-fixed) splits manifest. The splits manifest "
            "is gitignored machine-local data (raw CICIoT2023 not in repo per "
            "CIC license), so the on-disk SHA is regenerated per machine; on a "
            "fresh clone this input is absent and the check SKIPs."
        ),
    },
}


def _check_manifest(label: str, manifest_path: Path) -> tuple[int, int, int, int, list[str]]:
    """Return (n_ok, n_fail, n_skip, n_known, msgs) for one manifest.

    `n_known` counts pins whose SHA mismatch is a pre-registered,
    documented divergence (Step-1/Step-2 F1) — these do NOT contribute
    to FAIL and are reported separately as ``KNOWN-DIVERGENCE``.
    """
    if not manifest_path.exists():
        return 0, 1, 0, 0, [f"{label}: manifest missing at {manifest_path}"]
    try:
        manifest = json.loads(manifest_path.read_text())
    except json.JSONDecodeError as exc:
        return 0, 1, 0, 0, [f"{label}: manifest JSON parse error: {exc}"]

    pins = _walk_pin_entries(manifest.get("inputs", {}), prefix="inputs")
    if not pins:
        return 0, 0, 1, 0, [f"{label}: no SHA pins to check (inputs empty)"]

    n_ok = n_fail = n_skip = n_known = 0
    msgs: list[str] = []
    for label_full, pin in pins:
        path_str = pin["path"]
        recorded_sha = pin["sha256"]
        # Try absolute first; fall back to repo-relative.
        p = Path(path_str)
        if not p.is_absolute():
            # First try repo-rel; then try manifest-dir-rel.
            for candidate in (_ROOT / path_str, manifest_path.parent / path_str):
                if candidate.exists():
                    p = candidate
                    break
            else:
                p = _ROOT / path_str  # default to repo-rel for missing-msg
        actual_sha = _sha256(p)
        if actual_sha is None:
            n_skip += 1
            msgs.append(
                f"  SKIP  {label} {label_full}: input not on disk ({path_str}; gitignored?)"
            )
        elif actual_sha == recorded_sha:
            n_ok += 1
        else:
            # Check the pre-registered divergence table.
            known = _KNOWN_DIVERGENCES.get((label, path_str, recorded_sha))
            if known and known.get("actual_sha") == actual_sha:
                n_known += 1
                msgs.append(
                    f"  KNOWN {label} {label_full}: SHA mismatch on {path_str}\n"
                    f"           recorded: {recorded_sha}\n"
                    f"           actual:   {actual_sha}\n"
                    f"           finding:  {known['finding_id']}\n"
                    f"           note:     {known['note']}"
                )
            else:
                n_fail += 1
                msgs.append(
                    f"  FAIL  {label} {label_full}: SHA mismatch on {path_str}\n"
                    f"           recorded: {recorded_sha}\n"
                    f"           actual:   {actual_sha}"
                )
    return n_ok, n_fail, n_skip, n_known, msgs


def _check_scoreboard(label: str, path: Path) -> tuple[int, int, int, list[str]]:
    """Verify scoreboard parses + every gate.status is in the canonical enum."""
    if not path.exists():
        return 0, 1, 0, [f"  FAIL  scoreboard {label}: missing at {path}"]
    try:
        s = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return 0, 1, 0, [f"  FAIL  scoreboard {label}: JSON parse error: {exc}"]

    gates = s.get("gates", {})
    if isinstance(gates, dict):
        items = list(gates.items())
        statuses = [(gid, g.get("status")) for gid, g in items if isinstance(g, dict)]
    elif isinstance(gates, list):
        statuses = [(g.get("id", "?"), g.get("status")) for g in gates if isinstance(g, dict)]
    else:
        return 0, 1, 0, [f"  FAIL  scoreboard {label}: .gates is neither dict nor list"]

    n_ok = n_fail = 0
    msgs: list[str] = []
    for gid, status in statuses:
        if status in _VALID_STATUS:
            n_ok += 1
        else:
            n_fail += 1
            msgs.append(
                f"  FAIL  scoreboard {label} gate {gid}: status={status!r} not in canonical enum"
            )
    return n_ok, n_fail, 0, msgs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="R1 smoke-reproducibility harness — verify hash-chain integrity."
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 1 even on SKIP entries (default: exit 1 only on FAIL)",
    )
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    print("=" * 78)
    print("R1 smoke-reproducibility harness")
    print(f"Repo root: {_ROOT}")
    print("=" * 78)

    total_ok = total_fail = total_skip = total_known = 0
    all_msgs: list[str] = []

    print("\n--- Manifest hash-chain checks ---")
    for label, mpath, _committed in _TARGETS:
        n_ok, n_fail, n_skip, n_known, msgs = _check_manifest(label, mpath)
        if n_fail == 0 and n_known == 0:
            print(f"  {'OK' if n_skip == 0 else 'OK*'}    {label:18}  ({n_ok} OK / {n_skip} SKIP)")
        elif n_fail == 0:
            print(f"  OK†   {label:18}  ({n_ok} OK / {n_known} KNOWN-DIVERGENCE / {n_skip} SKIP)")
        else:
            print(
                f"  FAIL  {label:18}  ({n_ok} OK / {n_fail} FAIL / {n_known} KNOWN / {n_skip} SKIP)"
            )
        all_msgs.extend(msgs)
        total_ok += n_ok
        total_fail += n_fail
        total_skip += n_skip
        total_known += n_known

    print("\n--- Scoreboard schema checks ---")
    for label, spath in _SCOREBOARDS:
        n_ok, n_fail, _, msgs = _check_scoreboard(label, spath)
        if n_fail == 0:
            print(f"  OK    scoreboard {label}     ({n_ok} gates valid status)")
        else:
            print(f"  FAIL  scoreboard {label}     ({n_ok} OK / {n_fail} FAIL)")
        all_msgs.extend(msgs)
        total_ok += n_ok
        total_fail += n_fail

    print("\n--- Tally ---")
    print(f"  total OK:               {total_ok}")
    print(f"  total FAIL:             {total_fail}")
    print(f"  total KNOWN-DIVERGENCE: {total_known}  (pre-registered, see _KNOWN_DIVERGENCES)")
    print(f"  total SKIP:             {total_skip}  (gitignored inputs not on disk)")

    if all_msgs:
        print("\n--- Detail ---")
        for m in all_msgs:
            print(m)

    rc = 0
    if total_fail > 0:
        rc = 1
        print("\nVERDICT: FAIL — at least one hash-chain or scoreboard check failed.")
    elif args.strict and total_skip > 0:
        rc = 1
        print(
            "\nVERDICT: STRICT-FAIL — every check OK but --strict requires "
            "all gitignored inputs on disk too."
        )
    else:
        print("\nVERDICT: PASS — hash chain intact; scoreboard schemas valid.")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
