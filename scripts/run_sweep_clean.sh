#!/usr/bin/env bash
# Clean Transformer sweep: no DiffAugment and no EMA.
# Identical training pipeline to ResNet experiments, only architecture differs.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-python}"
MAX_JOBS="${MAX_JOBS:-2}"

wait_slot() {
  while true; do
    n=$(jobs -r -p 2>/dev/null | wc -l)
    if [ "${n:-0}" -lt "$MAX_JOBS" ]; then break; fi
    wait -n 2>/dev/null || sleep 2
  done
}

LOGDIR="./logs/sweep_clean_10k"
mkdir -p "$LOGDIR"
echo "=== Clean sweep start $(date) ===" | tee "$LOGDIR/master.log"

run_one() {
  local tag="$1" b="$2"
  wait_slot
  echo "[START] $tag  betas=$b" | tee -a "$LOGDIR/master.log"
  "$PY" main.py \
    --dataset celeba --arch trans64 --optimizer adam_nm \
    --lr_G 1e-4 --lr_D 1e-4 \
    --batch_size 64 \
    --total_steps 10000 --final_total_steps 100000 \
    --eval_step 1000 \
    --diff_aug "" --ema_decay 0 \
    --betas "$b" \
    >>"$LOGDIR/${tag}.log" 2>&1 &
}

for B1 in -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5; do
  run_one "b1_${B1}_b2_0.99" "${B1},0.99"
done

for B2 in 0.9 0.8 0.7 0.6 0.5 0.4 0.3; do
  run_one "b1_0_b2_${B2}" "0,${B2}"
done

wait
echo "=== All done $(date) ===" | tee -a "$LOGDIR/master.log"
