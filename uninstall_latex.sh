#!/usr/bin/env bash
# uninstall_latex.sh — completely removes BasicTeX / TeX Live from macOS.
#
# Run ONCE in your Terminal (requires your admin password):
#   bash uninstall_latex.sh
#
# After this script:
#   which pdflatex   → (empty)
#   brew list | grep basictex  → (empty)
# Use `make thesis` (Docker) to compile the thesis instead.

set -euo pipefail

echo "==> Removing TeX Live distribution (/usr/local/texlive)..."
if [ -d /usr/local/texlive ]; then
  sudo rm -rf /usr/local/texlive
  echo "    Done."
else
  echo "    Not found — skipping."
fi

echo "==> Removing /Library/TeX umbrella (symlinks + Distributions)..."
if [ -d /Library/TeX ]; then
  sudo rm -rf /Library/TeX
  echo "    Done."
else
  echo "    Not found — skipping."
fi

echo "==> Removing user-level TeX Live prefs (~/Library/texlive)..."
if [ -d ~/Library/texlive ]; then
  rm -rf ~/Library/texlive
  echo "    Done."
else
  echo "    Not found — skipping."
fi

echo "==> Removing /etc/paths.d/TeX and /etc/manpaths.d/TeX ..."
sudo rm -f /etc/paths.d/TeX /etc/manpaths.d/TeX
echo "    Done."

echo "==> Removing Homebrew cask record for basictex..."
if brew list --cask 2>/dev/null | grep -q basictex; then
  # --zap removes the pkg receipts too; skip if already gone
  brew uninstall --cask --zap basictex 2>/dev/null || brew uninstall --cask basictex 2>/dev/null || true
  echo "    Done."
else
  echo "    basictex cask not tracked by brew — skipping."
fi

echo ""
echo "==> LaTeX fully uninstalled."
echo "    'which pdflatex' should now return nothing."
echo "    Use 'make thesis' to compile the thesis via Docker."
