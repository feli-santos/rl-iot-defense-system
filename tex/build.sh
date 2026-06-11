#!/usr/bin/env bash
# tex/build.sh — Compile the dissertation PDF using the FEEC CCPG 001-2015
# (abnTeX2-based) template via a local container image (Podman; Docker fallback).
#
# Main TeX file: tex/main.tex
# Output PDF:    tex/main.pdf
#
# Usage (from anywhere in the repo):
#   bash tex/build.sh                 # default: build image (first time) + full compile
#   bash tex/build.sh --rebuild       # force rebuild the container image before compiling
#   bash tex/build.sh --draft         # single fast pdflatex pass (no bibtex/makeindex)
#   bash tex/build.sh --no-docker     # use the host TeX Live (must have abntex2 installed)
#   bash tex/build.sh --timeout=600   # per-pass timeout in seconds (default 480)
#
# The template uses BibTeX via abntex2cite (NOT biber). Full sequence:
#   pdflatex -> bibtex -> makeindex -> pdflatex -> pdflatex
#
# Container engine:
#   Podman is auto-detected and preferred if available; otherwise Docker is used.
#
# Notes:
#   * `epstopdf` is invoked automatically via `\usepackage{epstopdf}`; needs
#     `-shell-escape` if .eps figures are converted on the fly. The image
#     pre-converts .eps to .pdf at pull time, so shell-escape is not needed.

set -euo pipefail

# ── Auto-detect container engine (Podman preferred, Docker fallback) ─────────
if command -v podman >/dev/null 2>&1; then
  ENGINE="podman"
elif command -v docker >/dev/null 2>&1; then
  ENGINE="docker"
else
  echo "==> ERROR: neither 'podman' nor 'docker' found on PATH." >&2
  echo "    Install one of them, or use --no-docker if you have a local TeX Live." >&2
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_DIR="$REPO_ROOT/tex"
IMAGE_NAME="rl-iot-thesis"
DOCKERFILE="$TEX_DIR/Dockerfile"
MAIN_TEX="main"      # without .tex extension

# ── Defaults ──────────────────────────────────────────────────────────────────
REBUILD=0
DRAFT=0
USE_DOCKER=1
PASS_TIMEOUT=480   # seconds per pass; full build is up to 5 passes

# ── Parse args ────────────────────────────────────────────────────────────────
for arg in "$@"; do
  case "$arg" in
    --rebuild)         REBUILD=1 ;;
    --draft)           DRAFT=1 ;;
    --no-docker)       USE_DOCKER=0 ;;  # legacy alias; still works
    --timeout=*)       PASS_TIMEOUT="${arg#--timeout=}" ;;
    -h|--help)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *)
      echo "Unknown argument: $arg" >&2
      echo "Try: bash tex/build.sh --help" >&2
      exit 1
      ;;
  esac
done

# ── Locate a `timeout` binary (macOS ships gtimeout via coreutils) ────────────
if command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
elif command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
else
  echo "==> WARNING: neither 'timeout' nor 'gtimeout' found on PATH."
  echo "    On macOS install coreutils: 'brew install coreutils'."
  echo "    Continuing without per-pass timeout."
  TIMEOUT_BIN=""
fi

run_with_timeout () {
  local label="$1"; shift
  [ "$1" = "--" ] && shift
  local rc=0
  if [ -n "$TIMEOUT_BIN" ]; then
    "$TIMEOUT_BIN" "$PASS_TIMEOUT" "$@" || rc=$?
    if [ "$rc" -eq 124 ]; then
      echo ""
      echo "==> ❌ TIMEOUT: $label exceeded ${PASS_TIMEOUT}s — likely a compile loop."
      tail_log_on_failure
      exit 124
    elif [ "$rc" -ne 0 ]; then
      echo ""
      echo "==> ❌ FAILED: $label  (exit $rc)"
      tail_log_on_failure
      exit "$rc"
    fi
  else
    "$@" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo ""
      echo "==> ❌ FAILED: $label  (exit $rc)"
      tail_log_on_failure
      exit "$rc"
    fi
  fi
}

tail_log_on_failure () {
  if [ -f "$TEX_DIR/${MAIN_TEX}.log" ]; then
    echo ""
    echo "==> Last 60 lines of tex/${MAIN_TEX}.log:"
    echo "------------------------------------------------------------"
    tail -n 60 "$TEX_DIR/${MAIN_TEX}.log"
    echo "------------------------------------------------------------"
  fi
}

# ── Purge stale aux files ─────────────────────────────────────────────────────
purge_aux () {
  echo "==> Purging stale auxiliary files..."
  (
    cd "$TEX_DIR"
    rm -f \
      "${MAIN_TEX}.aux" "${MAIN_TEX}.bbl" "${MAIN_TEX}.blg" "${MAIN_TEX}.log" \
      "${MAIN_TEX}.out" "${MAIN_TEX}.toc" "${MAIN_TEX}.lof" "${MAIN_TEX}.lot" \
      "${MAIN_TEX}.idx" "${MAIN_TEX}.ind" "${MAIN_TEX}.ilg" \
      "${MAIN_TEX}.glo" "${MAIN_TEX}.gls" "${MAIN_TEX}.glg" \
      "${MAIN_TEX}.nlo" "${MAIN_TEX}.nls" \
      "${MAIN_TEX}.fdb_latexmk" "${MAIN_TEX}.fls" "${MAIN_TEX}.synctex.gz" \
      "${MAIN_TEX}.tex.bak" "${MAIN_TEX}.brf" \
      missfont.log texput.log
    # Per-chapter aux files left behind by \include
    find . -maxdepth 2 -name "*.aux" -not -path "./_legacy/*" -delete 2>/dev/null || true
  )
}

# ── Build container image if needed ─────────────────────────────────────────
ensure_image () {
  local need_build=0
  if [ "$REBUILD" -eq 1 ]; then
    need_build=1
    echo "==> --rebuild requested: rebuilding $ENGINE image '$IMAGE_NAME'..."
  elif ! "$ENGINE" image inspect "$IMAGE_NAME" &>/dev/null; then
    need_build=1
    echo "==> $ENGINE image '$IMAGE_NAME' not found — building (one-off, ~3–6 min)..."
  fi
  if [ "$need_build" -eq 1 ]; then
    "$ENGINE" build --tag "$IMAGE_NAME" --file "$DOCKERFILE" "$TEX_DIR"
    echo "==> Image '$IMAGE_NAME' ready."
  fi
}

# ── Compile passes ────────────────────────────────────────────────────────────
engine_run () {
  "$ENGINE" run --rm -v "$TEX_DIR":/work -w /work --entrypoint "" "$IMAGE_NAME" "$@"
}

pdflatex_pass () {
  local pass_name="$1"
  echo "==> Running $pass_name..."
  if [ "$USE_DOCKER" -eq 1 ]; then
    run_with_timeout "$pass_name" -- engine_run \
      pdflatex -interaction=nonstopmode -file-line-error "${MAIN_TEX}.tex"
  else
    run_with_timeout "$pass_name" -- bash -c \
      "cd '$TEX_DIR' && pdflatex -interaction=nonstopmode -file-line-error ${MAIN_TEX}.tex"
  fi
}

bibtex_pass () {
  echo "==> Running bibtex..."
  if [ "$USE_DOCKER" -eq 1 ]; then
    run_with_timeout "bibtex" -- engine_run bibtex "${MAIN_TEX}"
  else
    run_with_timeout "bibtex" -- bash -c "cd '$TEX_DIR' && bibtex ${MAIN_TEX}"
  fi
}

makeindex_pass () {
  # makeindex is OK to fail (e.g., no \makeindex used) — we don't fail the build.
  echo "==> Running makeindex (best-effort)..."
  if [ "$USE_DOCKER" -eq 1 ]; then
    engine_run makeindex "${MAIN_TEX}.idx" 2>/dev/null || true
  else
    (cd "$TEX_DIR" && makeindex "${MAIN_TEX}.idx" 2>/dev/null || true)
  fi
}

# ── Main ──────────────────────────────────────────────────────────────────────
echo "==> FEEC/UNICAMP dissertation build"
echo "    Main file: tex/${MAIN_TEX}.tex"
echo "    Mode: $([ "$DRAFT" -eq 1 ] && echo 'DRAFT (single pass)' || echo 'FULL (pdflatex × 3 + bibtex)')"
echo "    Engine: $([ "$USE_DOCKER" -eq 1 ] && echo "$ENGINE" || echo 'host')"
echo "    Per-pass timeout: ${PASS_TIMEOUT}s"
echo ""

purge_aux

if [ "$USE_DOCKER" -eq 1 ]; then
  ensure_image
fi

if [ "$DRAFT" -eq 1 ]; then
  pdflatex_pass "pdflatex pass 1/1 (draft)"
else
  pdflatex_pass "pdflatex pass 1/3"
  bibtex_pass
  makeindex_pass
  pdflatex_pass "pdflatex pass 2/3"
  pdflatex_pass "pdflatex pass 3/3"
fi

echo ""
if [ -f "$TEX_DIR/${MAIN_TEX}.pdf" ]; then
  echo "==> ✅ Done!  Output:"
  ls -lh "$TEX_DIR/${MAIN_TEX}.pdf"
else
  echo "==> ❌ ${MAIN_TEX}.pdf was NOT produced. Check tex/${MAIN_TEX}.log."
  tail_log_on_failure
  exit 1
fi
