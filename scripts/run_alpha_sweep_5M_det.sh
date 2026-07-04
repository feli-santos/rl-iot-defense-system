#!/usr/bin/env bash
# Chained deterministic 5M alpha-sweep driver.
#
# Waits for the coupling ablation (PID passed as $1, optional) to finish, then
# for each aliasing rate alpha in {0.0, 0.2, 0.6, 0.8, 1.0}:
#   1. trains dqn/ppo/a2c x 10 seeds x 5M steps, no early stop, n_eval=50
#      -> runs/redesign_5M_det/alpha_<NN>/
#   2. seeded benchmark (n=300) with the mandatory POMDP flags
#      -> runs/redesign_5M_det/benchmark_alpha_<NN>/
#
# alpha=0.4 is already done (runs/redesign_5M_det/alpha_04 + benchmark_alpha_04).
# Deterministic single-threaded BLAS so 10 workers don't oversubscribe 11 cores.
set -uo pipefail

cd "$(dirname "$0")/.."
PY=.venv/bin/python
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1

WAIT_PID="${1:-}"
LOGDIR="runs/redesign_5M_det/_chain_logs"
mkdir -p "$LOGDIR"

if [[ -n "$WAIT_PID" ]]; then
  echo "$(date -u +%FT%TZ) waiting for coupling PID $WAIT_PID to finish..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
  echo "$(date -u +%FT%TZ) coupling PID $WAIT_PID finished; starting alpha sweep."
fi

REWARD_OVERRIDES_BASE='{"reward_mode":"outcome","session_coherent":true,"no_post_transition_leak":true,"proximity_coupled":true}'

# alpha value -> two-digit NN tag (0.0->00, 0.2->02, 0.6->06, 0.8->08, 1.0->10)
declare -a ALPHAS=(0.0 0.2 0.6 0.8 1.0)

for ALPHA in "${ALPHAS[@]}"; do
  NN=$(printf '%02d' "$(echo "$ALPHA * 10 / 1" | bc)")
  TRAIN_ROOT="runs/redesign_5M_det/alpha_${NN}"
  BENCH_ROOT="runs/redesign_5M_det/benchmark_alpha_${NN}"
  OVERRIDES=$(echo "$REWARD_OVERRIDES_BASE" | sed "s/\"proximity_coupled\":true/\"proximity_coupled\":true,\"aliasing_rate\":${ALPHA}/")

  echo "$(date -u +%FT%TZ) ===== alpha=${ALPHA} (NN=${NN}) TRAIN ====="
  $PY -m scripts.blue_team.run_sweep \
    --algos dqn ppo a2c \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --total-timesteps 5000000 \
    --eval-freq 25000 \
    --n-eval-episodes 50 \
    --out-root "$TRAIN_ROOT" \
    --parallel 10 \
    --no-early-stop \
    --impact-is-terminal false \
    --reward-overrides "$OVERRIDES" \
    --continue-on-failure \
    > "$LOGDIR/train_alpha_${NN}.log" 2>&1
  echo "$(date -u +%FT%TZ) alpha=${ALPHA} TRAIN done (rc=$?)"

  echo "$(date -u +%FT%TZ) ===== alpha=${ALPHA} (NN=${NN}) BENCHMARK ====="
  $PY -m scripts.benchmark.run_test_eval \
    --algos dqn ppo a2c \
    --seeds 0 1 2 3 4 5 6 7 8 9 \
    --n-episodes 30 \
    --n-deterministic-episodes 300 \
    --blue-team-runs-root "$TRAIN_ROOT" \
    --out-root "$BENCH_ROOT" \
    --rf-path artifacts/detector/random_forest.joblib \
    --reward-mode outcome \
    --aliasing-rate "$ALPHA" \
    --session-coherent \
    --no-post-transition-leak \
    --proximity-coupled \
    --proximity-min-escalation 0.4 \
    > "$LOGDIR/bench_alpha_${NN}.log" 2>&1
  echo "$(date -u +%FT%TZ) alpha=${ALPHA} BENCHMARK done (rc=$?)"
done

echo "$(date -u +%FT%TZ) ===== ALL ALPHAS COMPLETE ====="
