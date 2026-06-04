"""Detector closer: assemble ``G4_scoreboard.json`` from locked detector artefacts.

This is a derived-only emitter. It reads the locked detector artefacts —
``F11_summary.json`` (per-gate evaluation produced by
``scripts/detector/train_detector.py``) and ``manifest.json`` (input/output
hash chain) — and writes ``G4_scoreboard.json`` in the benchmark-native
schema (``status`` enum + ``finding_id``), mirroring
``docs/results/benchmark/benchmark_acceptance.json``.

This script does NOT retrain anything. It does NOT touch
``F11_summary.json`` (the producer of the gate verdicts is
``scripts/detector/train_detector.py``; the locked artefacts on disk
are the canonical numerical record).

Usage::

    python -m scripts.detector.close_detector [--out-dir docs/results/stage-detector]
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("scripts.detector.close_detector")

_ROOT = Path(__file__).resolve().parents[2]


# Canonical scoreboard-status enum (benchmark-native; see
# docs/results/benchmark/benchmark_acceptance.json + 07_HANDOFF.md L196).
_STATUS_PASS = "PASS"
_STATUS_PASS_WITH_FINDING = "PASS-WITH-FINDING"
_STATUS_PASS_WITHOUT_STRETCH = "PASS-WITHOUT-STRETCH"
_STATUS_FAIL_WITH_FINDING = "FAIL-WITH-FINDING"
_STATUS_FAIL = "FAIL"
_STATUS_SKIP = "SKIP"


# Per-gate finding-id table. The OOD-recall G4.4 result is the canonical
# detector thesis-finding entry (revised D2 / step 4.5 PLAN locking),
# carried into the ablation evaluation as the F15 / D7.9.1 headline.
_GATE_FINDING_ID: dict[str, str] = {
    "G4.4": "D2.1",  # OOD-recall blind spot: VulnerabilityScan recall = 0.001
}


# Canonical legacy-status normaliser. F11_summary.json::gates.G4.4 currently
# ships "PASS-with-finding" (lowercase suffix); the unified schema spells
# the enum members all-uppercase to match G6_scoreboard.json. The
# normaliser is permissive on input casing.
def _canon_status(raw: str | None) -> str:
    if raw is None:
        return _STATUS_SKIP
    s = raw.upper().strip()
    aliases = {
        "PASS-WITH-FINDING": _STATUS_PASS_WITH_FINDING,
        "PASS_WITH_FINDING": _STATUS_PASS_WITH_FINDING,
        "PASSWITHFINDING": _STATUS_PASS_WITH_FINDING,
        "PASS-WITH-FINDINGS": _STATUS_PASS_WITH_FINDING,
        "PASS-WITHOUT-STRETCH": _STATUS_PASS_WITHOUT_STRETCH,
        "FAIL-WITH-FINDING": _STATUS_FAIL_WITH_FINDING,
        "PASS": _STATUS_PASS,
        "FAIL": _STATUS_FAIL,
        "SKIP": _STATUS_SKIP,
    }
    return aliases.get(s, s)


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


def _build_g4_1(out_dir: Path) -> dict[str, Any]:
    """G4.1 — full pytest suite green. docs/results/stage-detector/RESULTS.md
    L20 records ``329 / 329 PASS`` at the detector lock. A later dead-code
    cleanup (commit 281860a) reduced the count to 411 by deleting tests for
    a retired src/benchmarking/ package; the detector frozen tests remain
    green. We mark this gate SKIP at the closer level (the producer is
    ``pytest -q``, not this script) and reference the locked RESULTS.md
    value in the note.
    """
    return {
        "id": "G4.1",
        "description": "full pytest suite green",
        "threshold": "all tests green",
        "value": "329 / 329 (detector lock; post-cleanup: 411/411 at HEAD; see RESULTS.md §9 footnote)",
        "status": _STATUS_SKIP,
        "evaluated": False,
        "note": (
            "evaluated separately by `pytest -q`; locked detector value "
            "preserved verbatim from RESULTS.md §2"
        ),
    }


def _g4_2_to_g4_5(summary: dict[str, Any]) -> list[dict[str, Any]]:
    """Translate ``F11_summary.json::gates`` (G4.2..G4.5) to the unified shape."""
    src = summary.get("gates", {}) if summary else {}
    rows: list[dict[str, Any]] = []
    for gid in ("G4.2", "G4.3", "G4.4", "G4.5"):
        s = src.get(gid)
        if s is None:
            rows.append(
                {
                    "id": gid,
                    "description": "missing in F11_summary.json (detector not yet sealed?)",
                    "status": _STATUS_FAIL,
                    "value": "F11_summary.json missing this gate",
                }
            )
            continue
        status = _canon_status(s.get("status"))
        row: dict[str, Any] = {
            "id": gid,
            "description": s.get("name") or "(no description)",
        }
        # threshold + observed: keep verbatim under the original keys
        # but also expose a compact `value` summary string for jq.
        for k in (
            "threshold",
            "threshold_min_recall",
            "threshold_ms",
            "observed",
            "observed_worst",
            "observed_worst_at",
            "observed_min",
            "observed_max",
            "observed_gap",
            "observed_ms",
            "diagnostic_cross_baseline_worst",
            "diagnostic_cross_baseline_worst_at",
            "per_class",
            "note",
        ):
            if k in s:
                row[k] = s[k]
        row["status"] = status
        finding_id = _GATE_FINDING_ID.get(gid)
        if (
            status
            in {
                _STATUS_PASS_WITH_FINDING,
                _STATUS_PASS_WITHOUT_STRETCH,
                _STATUS_FAIL_WITH_FINDING,
            }
            and finding_id is not None
        ):
            row["finding_id"] = finding_id
        rows.append(row)
    return rows


def build_scoreboard(out_dir: Path) -> dict[str, Any]:
    summary_path = out_dir / "F11_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing {summary_path} — detector step has not been sealed yet")
    summary = json.loads(summary_path.read_text())

    manifest_path = out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    gates: list[dict[str, Any]] = [_build_g4_1(out_dir)]
    gates.extend(_g4_2_to_g4_5(summary))

    statuses = [g["status"] for g in gates]
    payload = {
        "schema_version": "2.0",
        "phase": 4,
        "git_sha": _git_sha(),
        "completed_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "evaluated_against": {
            "split": "test_balanced",
            "manifest_git_sha": manifest.get("git_sha"),
            "manifest_generated_at": manifest.get("generated_at"),
            "input_hashes_in": "manifest.json",
        },
        "summary": {
            "total_gates": len(gates),
            "pass": statuses.count(_STATUS_PASS),
            "pass_with_finding": statuses.count(_STATUS_PASS_WITH_FINDING),
            "pass_without_stretch": statuses.count(_STATUS_PASS_WITHOUT_STRETCH),
            "fail_with_finding": statuses.count(_STATUS_FAIL_WITH_FINDING),
            "fail": statuses.count(_STATUS_FAIL),
            "skip": statuses.count(_STATUS_SKIP),
        },
        "gates": gates,
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Detector closer: G4_scoreboard.json")
    p.add_argument("--out-dir", default="docs/results/stage-detector")
    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _ROOT / out_dir
    payload = build_scoreboard(out_dir)
    score_path = out_dir / "G4_scoreboard.json"
    score_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote %s", score_path)
    s = payload["summary"]
    print(
        f"=== Detector gate scoreboard ===\n"
        f"  total={s['total_gates']}  "
        f"pass={s['pass']}  pass_with_finding={s['pass_with_finding']}  "
        f"fail_with_finding={s['fail_with_finding']}  fail={s['fail']}  "
        f"skip={s['skip']}"
    )
    for g in payload["gates"]:
        fid = f"  [{g['finding_id']}]" if g.get("finding_id") else ""
        print(f"  {g['id']:5} [{g['status']:18}] {g['description']}{fid}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
