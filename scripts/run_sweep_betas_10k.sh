#!/usr/bin/env bash
# Group 1: beta1 in {-0.5, ..., 0.5}, beta2=0.99 (11 runs).
# Group 2: beta1=0, beta2 in {0.9, ..., 0.3} (7 runs).
# Each run uses 10,000 training steps, FID/IS evaluation every 1,000 steps,
# batch size 64, and at most two parallel jobs by default.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
MAX_JOBS="${MAX_JOBS:-2}"

wait_slot() {
  while true; do
    n=$(jobs -r -p 2>/dev/null | wc -l)
    if [ "${n:-0}" -lt "$MAX_JOBS" ]; then
      break
    fi
    wait -n 2>/dev/null || sleep 2
  done
}

COMMON=(
  --dataset celeba --arch sagan64 --optimizer adam_nm
  --lr_G 1e-4 --lr_D 1e-4
  --batch_size 64
  --total_steps 10000 --final_total_steps 100000
  --eval_step 1000
)

LOGDIR="${LOGDIR:-./logs/sagan64_sweep_betas_10k}"
mkdir -p "$LOGDIR"
RUN_ID="$(date +%Y%m%d_%H%M%S)"
MASTER_LOG="$LOGDIR/run_${RUN_ID}.log"
echo "Logging per-job stderr to $LOGDIR/*.log ; tail: $MASTER_LOG"

{
  echo "=== sweep start RUN_ID=$RUN_ID ==="
  date

  for B1 in -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5; do
    wait_slot
    tag="g1_b1_${B1}_b2_0.99"
    echo "[START] $tag"
    "$PY" main.py "${COMMON[@]}" --betas "${B1},0.99" \
      >>"$LOGDIR/${tag}.log" 2>&1 &
  done

  for B2 in 0.9 0.8 0.7 0.6 0.5 0.4 0.3; do
    wait_slot
    tag="g2_b1_0_b2_${B2}"
    echo "[START] $tag"
    "$PY" main.py "${COMMON[@]}" --betas "0,${B2}" \
      >>"$LOGDIR/${tag}.log" 2>&1 &
  done

  wait
  echo "=== sweep finished ==="
  date
} | tee -a "$MASTER_LOG"
