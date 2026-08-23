#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/guoheng/custom_simulation}"
PYTHON_BIN="${PYTHON_BIN:-/home/guoheng/.conda/envs/palpation/bin/python}"
RUN_DIR="${RUN_DIR:?RUN_DIR is required}"
MAX_REPAIR_PARALLEL="${MAX_REPAIR_PARALLEL:-1}"
POLL_SECONDS="${POLL_SECONDS:-300}"
SCRIPT="scripts/run_real_phantom_sample_newton.py"

cd "$PROJECT_ROOT"
mkdir -p "$RUN_DIR/logs"

COMMON_ARGS=(
  --source-metadata data/palpation_synthetic_depth_modes_1000train_400val_400test_20260622/random_trajectory/data/test/sample_0147_gt.json
  --out-dir "$RUN_DIR"
  --cells-x 96
  --cells-y 96
  --cells-z 32
  --probe-diameter-mm 8
  --max-indentation-mm 10
  --press-steps 160
  --substeps 16
  --vbd-iterations 32
  --soft-contact-margin-mm 1
  --normal-k-mu 10000
  --normal-k-lambda 10000
  --normal-k-damp 0.0001
  --soft-contact-ke 2000000
  --rigid-stiffness-multiplier 10000
  --device cuda:0
)

worker_count() {
  local matches
  matches="$(pgrep -af "run_real_phantom_sample_newton.py.*--out-dir ${RUN_DIR}" 2>/dev/null || true)"
  printf '%s\n' "$matches" | awk 'NF && $0 !~ /supervise/ {count += 1} END {print count + 0}'
}

chunk_count() {
  find "$RUN_DIR/chunks" -maxdepth 1 -name 'rows_*.npz' -print 2>/dev/null | wc -l
}

missing_rows() {
  for row in $(seq 0 19); do
    next=$((row + 1))
    chunk=$(printf "%s/chunks/rows_%03d_%03d.npz" "$RUN_DIR" "$row" "$next")
    if [ ! -f "$chunk" ]; then
      echo "$row"
    fi
  done
}

launch_repairs() {
  local active=0
  local pids=()
  for row in "$@"; do
    local next=$((row + 1))
    local log_path
    log_path=$(printf "%s/logs/supervisor_repair_%02d_rows_%03d_%03d.log" "$RUN_DIR" "$row" "$row" "$next")
    "$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --resume --no-assemble --row-start "$row" --row-end "$next" > "$log_path" 2>&1 &
    pids+=("$!")
    active=$((active + 1))
    if [ "$active" -ge "$MAX_REPAIR_PARALLEL" ]; then
      for pid in "${pids[@]}"; do
        wait "$pid" || true
      done
      pids=()
      active=0
    fi
  done
  for pid in "${pids[@]}"; do
    wait "$pid" || true
  done
}

echo "supervisor started at $(date -Is) for $RUN_DIR" >> "$RUN_DIR/logs/supervisor.log"
attempt=0
while true; do
  count=$(chunk_count)
  workers=$(worker_count)
  echo "$(date -Is) chunks=$count workers=$workers" >> "$RUN_DIR/logs/supervisor.log"
  if [ "$count" -eq 20 ] && [ "$workers" -eq 0 ]; then
    break
  fi
  if [ "$workers" -eq 0 ] && [ "$count" -lt 20 ]; then
    attempt=$((attempt + 1))
    if [ "$attempt" -gt 4 ]; then
      printf '{"status":"failed","run_dir":"%s","updated_at":"%s","reason":"missing_chunks_after_repair_attempts"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
      exit 1
    fi
    mapfile -t missing < <(missing_rows)
    echo "$(date -Is) repairing missing rows: ${missing[*]}" >> "$RUN_DIR/logs/supervisor.log"
    launch_repairs "${missing[@]}"
    continue
  fi
  sleep "$POLL_SECONDS"
done

printf '{"status":"supervisor_assembling","run_dir":"%s","updated_at":"%s","force_estimator":"bottom_support_reaction"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
"$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --resume --assemble-only > "$RUN_DIR/logs/supervisor_assemble.log" 2>&1
printf '{"status":"complete","run_dir":"%s","updated_at":"%s","force_estimator":"bottom_support_reaction"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
echo "supervisor completed at $(date -Is)" >> "$RUN_DIR/logs/supervisor.log"
