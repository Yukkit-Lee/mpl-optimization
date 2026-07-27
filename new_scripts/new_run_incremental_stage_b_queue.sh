#!/bin/bash
# Phase 4: Run Full Data Secure Average Stage B formal queue (9 tasks)
# Sequential execution with checkpoint, skip-completed, and restart-safe.
# Can be killed and restarted — will skip already-COMPLETE tasks.

set -euo pipefail

source /root/.config/new-secure-feature-pipeline.env

TSV="$NEW_PROJECT_ROOT/new_configs/new_incremental_comparison/new_tfe_full_feature_stage_b.tsv"
LOG_DIR="$NEW_PROJECT_ROOT/new_log/new_incremental_comparison/txt"
STATUS_FILE="$NEW_PROJECT_ROOT/new_log/new_incremental_comparison/json/new_stage_b_queue_status.json"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_FILE")"

TOTAL=0
COMPLETED=0
FAILED_TASKS=()
TASK_IDS=()

# Parse TSV
while IFS=$'\t' read -r task_id checkpoint_dir command; do
    [[ "$task_id" =~ ^#.*$ ]] && continue
    [[ -z "$task_id" ]] && continue
    TASK_IDS+=("$task_id")
    TOTAL=$((TOTAL + 1))
done < "$TSV"

echo "Phase 4: Full Data Stage B formal queue"
echo "Total tasks: $TOTAL"
echo "Started at: $(date +'%Y-%m-%d %H:%M:%S')"
echo "Log dir: $LOG_DIR"
echo ""

for task_id in "${TASK_IDS[@]}"; do
    TASK_LINE=$(grep -F "$task_id" "$TSV" | head -1)
    CHECKPOINT_DIR=$(echo "$TASK_LINE" | cut -f2)
    COMMAND=$(echo "$TASK_LINE" | cut -f3-)

    LOG_FILE="$LOG_DIR/${task_id}.log"
    STATUS_JSON="$CHECKPOINT_DIR/status.json"

    # Check if already COMPLETE (two checks: status JSON + output report)
    ALREADY_COMPLETE=false
    if [ -f "$STATUS_JSON" ]; then
        STATUS=$(python3 -c "import json; d=json.load(open('$STATUS_JSON')); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        if [ "$STATUS" = "COMPLETE" ]; then
            ALREADY_COMPLETE=true
        fi
    fi
    # Also check if the output model exists and its report says COMPLETE
    if ! $ALREADY_COMPLETE; then
        # Extract dataset and seed from task_id
        DATASET=$(echo "$task_id" | sed 's/incremental_full_feature_stage_b_\([a-z0-9]*\)_seed.*/\1/')
        SEED=$(echo "$task_id" | sed 's/.*_seed\([0-9]*\)$/\1/')
        REPORT_PATH="$NEW_PROJECT_ROOT/new_output/new_incremental_comparison/stage_b/${DATASET}/seed${SEED}/new_secure_stage_b_report_${DATASET}_iid_seed${SEED}.json"
        if [ -f "$REPORT_PATH" ]; then
            REPORT_STATUS=$(python3 -c "import json; d=json.load(open('$REPORT_PATH')); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
            if [ "$REPORT_STATUS" = "COMPLETE" ]; then
                ALREADY_COMPLETE=true
            fi
        fi
    fi

    if $ALREADY_COMPLETE; then
        echo "[SKIP] $task_id (already COMPLETE)"
        COMPLETED=$((COMPLETED + 1))
        continue
    fi

    echo ""
    echo "=== [$((COMPLETED + 1))/$TOTAL] Running: $task_id ==="
    echo "Started at: $(date +'%Y-%m-%d %H:%M:%S')"
    echo "Log: $LOG_FILE"
    echo "Checkpoint: $CHECKPOINT_DIR"

    mkdir -p "$CHECKPOINT_DIR"

    START_TIME=$(date +%s)
    set +e
    eval "$COMMAND" > "$LOG_FILE" 2>&1
    EXIT_CODE=$?
    set -e
    END_TIME=$(date +%s)
    ELAPSED=$((END_TIME - START_TIME))

    if [ "$EXIT_CODE" -ne 0 ]; then
        echo "[FAILED] $task_id (exit code=$EXIT_CODE, elapsed=${ELAPSED}s)"
        echo "See log: $LOG_FILE"

        python3 -c "
import json, time
json.dump({'task_id':'$task_id','status':'FAILED','exit_code':$EXIT_CODE,'elapsed_seconds':$ELAPSED,'timestamp':time.time()}, open('$STATUS_JSON','w'), indent=2)
" 2>/dev/null || true

        python3 -c "
import json, time
s={'total':$TOTAL,'completed':$COMPLETED,'failed':1,'failed_task':'$task_id','last_update':time.time()}
json.dump(s, open('$STATUS_FILE','w'), indent=2)
" 2>/dev/null || true

        echo ""
        echo "QUEUE STOPPED due to failure. Remove status.json and fix before resume."
        exit 1
    fi

    echo "[COMPLETE] $task_id (elapsed=${ELAPSED}s)"
    COMPLETED=$((COMPLETED + 1))

    # Write success status
    python3 -c "
import json, time
json.dump({'task_id':'$task_id','status':'COMPLETE','elapsed_seconds':$ELAPSED,'timestamp':time.time()}, open('$STATUS_JSON','w'), indent=2)
" 2>/dev/null || true

    python3 -c "
import json, time
s={'total':$TOTAL,'completed':$COMPLETED,'failed':0,'current_task':'$task_id','last_update':time.time()}
json.dump(s, open('$STATUS_FILE','w'), indent=2)
" 2>/dev/null || true

    echo ""
done

echo ""
echo "=== Phase 4 Stage B queue COMPLETE ==="
echo "Total: $TOTAL, Completed: $COMPLETED"
echo "Finished at: $(date +'%Y-%m-%d %H:%M:%S')"
