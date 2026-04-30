#!/bin/bash
# Watcher: when phase7.end appears, run close_phase7.py and exit.
set -uo pipefail
cd /Users/felipe.santos/Projects/rl-iot-defense-system
LOG=logs/phase7/finalize.log
echo "[$(date '+%H:%M:%S')] finalizer waiting for phase7.end" > "$LOG"
while [ ! -f logs/phase7/phase7.end ]; do
  sleep 60
done
echo "[$(date '+%H:%M:%S')] phase7.end seen — running close_phase7.py" >> "$LOG"
source .venv/bin/activate
python -m scripts.ablation.close_phase7 >> "$LOG" 2>&1
echo "[$(date '+%H:%M:%S')] close_phase7 done (exit=$?)" >> "$LOG"
