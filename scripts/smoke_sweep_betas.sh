#!/usr/bin/env bash
# 每组 beta 只跑 10 步，验证命令与双进程并行是否正常（关闭 record/FID 以省时间）
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
    wait -n 2>/dev/null || sleep 1
  done
}

COMMON=(
  --dataset celeba --arch trans64 --optimizer adam_nm
  --lr_G 1e-4 --lr_D 1e-4
  --batch_size 64
  --total_steps 10 --final_total_steps 100000 --eval_step 1000
  --norecord --noenable_fid
)

echo "=== Smoke: 11 runs group1 (beta1 sweep, beta2=0.99) + 7 runs group2 (beta1=0, beta2 sweep), 10 steps each, max_parallel=$MAX_JOBS ==="

for B1 in -0.5 -0.4 -0.3 -0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5; do
  wait_slot
  echo "[START] g1 betas=${B1},0.99"
  "$PY" main.py "${COMMON[@]}" --betas "${B1},0.99" &
done

for B2 in 0.9 0.8 0.7 0.6 0.5 0.4 0.3; do
  wait_slot
  echo "[START] g2 betas=0,${B2}"
  "$PY" main.py "${COMMON[@]}" --betas "0,${B2}" &
done

wait
echo "=== Smoke sweep finished OK ==="
