#!/usr/bin/env bash
# tex/build.sh — Compile thesis.pdf using the minimal Docker image.
#
# Usage (from repo root):
#   bash tex/build.sh           # build image (first time only) + compile
#   bash tex/build.sh --rebuild  # force rebuild the Docker image before compiling
#
# The produced PDF is written to tex/thesis.pdf on your host.
# Requires: Docker running locally.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEX_DIR="$REPO_ROOT/tex"
IMAGE_NAME="rl-iot-thesis"
DOCKERFILE="$TEX_DIR/Dockerfile"

# ── Parse args ────────────────────────────────────────────────────────────────
REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    *) echo "Unknown argument: $arg" >&2; exit 1 ;;
  esac
done

# ── Build image if needed ─────────────────────────────────────────────────────
need_build=0
if [ "$REBUILD" -eq 1 ]; then
  need_build=1
  echo "==> --rebuild requested: rebuilding Docker image '$IMAGE_NAME'..."
elif ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
  need_build=1
  echo "==> Docker image '$IMAGE_NAME' not found — building (one-off, ~2–4 min)..."
fi

if [ "$need_build" -eq 1 ]; then
  docker build \
    --tag "$IMAGE_NAME" \
    --file "$DOCKERFILE" \
    "$TEX_DIR"
  echo "==> Image '$IMAGE_NAME' ready."
fi

# ── Compile ───────────────────────────────────────────────────────────────────
echo "==> Compiling thesis (pdflatex × 3 + biber)..."
docker run \
  --rm \
  --volume "$TEX_DIR":/work \
  "$IMAGE_NAME"

echo ""
echo "==> Done!  Output: tex/thesis.pdf"
ls -lh "$TEX_DIR/thesis.pdf"
