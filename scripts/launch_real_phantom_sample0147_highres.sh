#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/home/guoheng/custom_simulation}"
PYTHON_BIN="${PYTHON_BIN:-/home/guoheng/.conda/envs/palpation/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d-%H%M%S)}"
RUN_DIR="${RUN_DIR:-runs/${RUN_STAMP}-real_phantom_sample0147_rigid_ecoflex0010_probe8_depth10_reaction_v2_seq_t160_s16_i32}"
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

printf '{"status":"initializing","run_dir":"%s","started_at":"%s"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
"$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --init-only > "$RUN_DIR/logs/init.log" 2>&1

monitor_loop() {
  while true; do
    {
      echo "=== $(date -Is) ==="
      find "$RUN_DIR/chunks" -maxdepth 1 -name 'rows_*.npz' -print 2>/dev/null | sort | wc -l
      nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader || true
      ps -o pid,ppid,etime,pcpu,pmem,cmd -p "$WORKER_PID_CSV" 2>/dev/null || true
    } >> "$RUN_DIR/logs/monitor.log"
    sleep 60
  done
}

log_path="$RUN_DIR/logs/worker_rows_000_020.log"
"$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --resume --no-assemble --row-start 0 --row-end 20 > "$log_path" 2>&1 &
WORKER_PID=$!
printf '%s\n' "$WORKER_PID" > "$RUN_DIR/worker_pids.txt"
WORKER_PID_CSV="$WORKER_PID"
printf '{"status":"running","run_dir":"%s","worker_count":1,"worker_pids":[%s],"updated_at":"%s","force_estimator":"bottom_support_reaction"}\n' \
  "$RUN_DIR" "$WORKER_PID_CSV" "$(date -Is)" > "$RUN_DIR/job_status.json"

monitor_loop &
MONITOR_PID=$!

set +e
failed=0
wait "$WORKER_PID"
rc=$?
if [ "$rc" -ne 0 ]; then
  failed=1
  echo "worker pid $WORKER_PID failed with rc=$rc" >> "$RUN_DIR/logs/worker_failures.log"
fi
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

if [ "$failed" -ne 0 ]; then
  printf '{"status":"failed","run_dir":"%s","updated_at":"%s","reason":"one_or_more_workers_failed"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
  exit 1
fi

printf '{"status":"assembling","run_dir":"%s","updated_at":"%s","force_estimator":"bottom_support_reaction"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
"$PYTHON_BIN" "$SCRIPT" "${COMMON_ARGS[@]}" --resume --assemble-only > "$RUN_DIR/logs/assemble.log" 2>&1
printf '{"status":"complete","run_dir":"%s","updated_at":"%s","force_estimator":"bottom_support_reaction"}\n' "$RUN_DIR" "$(date -Is)" > "$RUN_DIR/job_status.json"
