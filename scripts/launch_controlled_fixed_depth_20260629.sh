#!/usr/bin/env bash
set -euo pipefail

cd "${HOME}/custom_simulation"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate palpation

OUT_DIR="${OUT_DIR:-data/palpation_fixed_depth_36mm_40step_ecoflex0010_linear_2000train_800val_800test_20260629}"
WORKER_COUNT="${WORKER_COUNT:-1}"
WORKER_INDEX="${WORKER_INDEX:-0}"
MAX_RESTARTS="${MAX_RESTARTS:-24}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${LOG_DIR}" "${OUT_DIR}/status"

STATUS_FILE="${OUT_DIR}/status/launcher_worker_${WORKER_INDEX}.json"
LAUNCH_LOG="${LOG_DIR}/launcher_worker_${WORKER_INDEX}.log"

write_status() {
  local state="$1"
  local restart="$2"
  local extra="${3:-}"
  printf '{"state":"%s","worker_index":%s,"worker_count":%s,"restart":%s,"out_dir":"%s","updated_at":"%s"%s}\n' \
    "$state" "$WORKER_INDEX" "$WORKER_COUNT" "$restart" "$OUT_DIR" "$(date -Is)" "$extra" > "$STATUS_FILE"
}

restart=0
write_status "starting" "$restart"
while true; do
  echo "[$(date -Is)] starting worker ${WORKER_INDEX}/${WORKER_COUNT}, restart=${restart}" | tee -a "$LAUNCH_LOG"
  set +e
  python scripts/generate_controlled_fixed_depth_dataset.py \
    --out-dir "$OUT_DIR" \
    --resume \
    --worker-count "$WORKER_COUNT" \
    --worker-index "$WORKER_INDEX" \
    --backend newton \
    --num-train 2000 \
    --num-val 800 \
    --num-test 800 \
    --seed 20260629 \
    --grid-h 20 \
    --grid-w 20 \
    --edge-margin 0.015 \
    --probe-radius 0.012 \
    --max-indentation 0.036 \
    --press-steps 40 \
    --substeps-per-depth 3 \
    --vbd-iterations 10 \
    --cells-x 48 \
    --cells-y 48 \
    --cells-z 16 \
    --particle-radius 0.004 \
    --normal-k-mu-min 8000 \
    --normal-k-mu-max 12000 \
    --normal-k-lambda-min 8000 \
    --normal-k-lambda-max 12000 \
    --lump-stiffness-min 5 \
    --lump-stiffness-max 30 \
    --no-save-phantom-3d \
    --no-save-press-records \
    --no-save-scan-animation \
    >> "$LAUNCH_LOG" 2>&1
  exit_code=$?
  set -e
  if [[ "$exit_code" -eq 0 ]]; then
    echo "[$(date -Is)] worker complete" | tee -a "$LAUNCH_LOG"
    write_status "complete" "$restart"
    exit 0
  fi
  restart=$((restart + 1))
  write_status "restarting" "$restart" ",\"last_exit_code\":${exit_code}"
  echo "[$(date -Is)] worker failed with ${exit_code}; restart ${restart}/${MAX_RESTARTS}" | tee -a "$LAUNCH_LOG"
  if [[ "$restart" -ge "$MAX_RESTARTS" ]]; then
    write_status "failed" "$restart" ",\"last_exit_code\":${exit_code}"
    exit "$exit_code"
  fi
  sleep 30
done
