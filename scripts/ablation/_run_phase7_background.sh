#!/bin/bash
set -uo pipefail
cd /Users/felipe.santos/Projects/rl-iot-defense-system
source .venv/bin/activate

LOG_DIR=logs/phase7
date +%s > "$LOG_DIR/phase7.start"

echo "[$(date '+%H:%M:%S')] phase-7 background runner starting (~7.5 h walk-away)" \
  | tee -a "$LOG_DIR/phase7.log"

# Run the 3 sweeps + figure step. Each is foregrounded WITHIN this
# background-shell so we can chain them; the outer & at invocation
# time is what makes the whole thing background.
{
  echo "=== [$(date '+%H:%M:%S')] starting phase-7-ood ===" >&2
  make phase-7-ood 2>&1
  echo "=== [$(date '+%H:%M:%S')] phase-7-ood DONE (exit=$?) ===" >&2
} > "$LOG_DIR/ood.log" 2>&1 &
PID_OOD=$!

{
  echo "=== [$(date '+%H:%M:%S')] starting phase-7-aggressiveness ===" >&2
  make phase-7-aggressiveness 2>&1
  echo "=== [$(date '+%H:%M:%S')] phase-7-aggressiveness DONE (exit=$?) ===" >&2
} > "$LOG_DIR/aggressiveness.log" 2>&1 &
PID_AGGR=$!

{
  echo "=== [$(date '+%H:%M:%S')] starting phase-7-reward ===" >&2
  make phase-7-reward 2>&1
  echo "=== [$(date '+%H:%M:%S')] phase-7-reward DONE (exit=$?) ===" >&2
} > "$LOG_DIR/reward.log" 2>&1 &
PID_REWARD=$!

echo "[$(date '+%H:%M:%S')] launched: ood=$PID_OOD aggr=$PID_AGGR reward=$PID_REWARD" \
  | tee -a "$LOG_DIR/phase7.log"
echo "$PID_OOD $PID_AGGR $PID_REWARD" > "$LOG_DIR/phase7.pids"

# Wait for all three to finish.
wait $PID_OOD; OOD_RC=$?
wait $PID_AGGR; AGGR_RC=$?
wait $PID_REWARD; REWARD_RC=$?

echo "[$(date '+%H:%M:%S')] all 3 sweeps finished: ood_rc=$OOD_RC aggr_rc=$AGGR_RC reward_rc=$REWARD_RC" \
  | tee -a "$LOG_DIR/phase7.log"

# Run F12 Pareto last (depends on F9 + F10 outputs).
echo "[$(date '+%H:%M:%S')] running phase-7-pareto" | tee -a "$LOG_DIR/phase7.log"
make phase-7-pareto > "$LOG_DIR/pareto.log" 2>&1
PARETO_RC=$?

date +%s > "$LOG_DIR/phase7.end"
echo "[$(date '+%H:%M:%S')] PHASE-7 RUNNER COMPLETE (ood=$OOD_RC aggr=$AGGR_RC reward=$REWARD_RC pareto=$PARETO_RC)" \
  | tee -a "$LOG_DIR/phase7.log"
