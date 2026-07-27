#!/bin/bash
# Phase 4: Run Full Data Secure Average Stage B formal queue (9 tasks)
# Sequential execution with checkpoint, logging, and status tracking.

set -euo pipefail

source /root/.config/new-secure-feature-pipeline.env

TSV="$NEW_PROJECT_ROOT/new_configs/new_incremental_comparison/new_tfe_full_feature_stage_b.tsv"
LOG_DIR="$NEW_PROJECT_ROOT/new_log/new_incremental_comparison/txt"
STATUS_FILE="$NEW_PROJECT_ROOT/new_log/new_incremental_comparison/json/new_stage_b_queue_status.json"

mkdir -p "$LOG_DIR" "$(dirname "$STATUS_FILE")"

TOTAL=0
COMPLETED=0
TASK_IDS=()

# Parse TSV (skip header comment)
while IFS=$'\t' read -r task_id checkpoint_dir command; do
    [[ "$task_id" =~ ^#.*$ ]] && continue
    [[ -z "$task_id" ]] && continue
    TASK_IDS+=("$task_id")
    TOTAL=$((TOTAL + 1))
done < "$TSV"

echo "Phase 4: Full Data Stage B formal queue"
echo "Total tasks: $TOTAL"
echo "Started at: $(date)"
echo "Log dir: $LOG_DIR"
echo ""

for task_id in "${TASK_IDS[@]}"; do
    # Find the task line
    TASK_LINE=$(grep -F "$task_id" "$TSV" | head -1)
    CHECKPOINT_DIR=$(echo "$TASK_LINE" | cut -f2)
    COMMAND=$(echo "$TASK_LINE" | cut -f3-)

    LOG_FILE="$LOG_DIR/${task_id}.log"
    STATUS_JSON="$CHECKPOINT_DIR/status.json"

    # Check if already complete
    if [ -f "$STATUS_JSON" ]; then
        STATUS=$(python3 -c "import json; d=json.load(open('$STATUS_JSON')); print(d.get('status','UNKNOWN'))" 2>/dev/null || echo "UNKNOWN")
        if [ "$STATUS" = "COMPLETE" ]; then
            echo "[SKIP] $task_id (already COMPLETE)"
            COMPLETED=$((COMPLETED + 1))
            continue
        fi
    fi

    echo ""
    echo "=== [$((COMPLETED + 1))/$TOTAL] Running: $task_id ==="
    echo "Started at: $(date)"
    echo "Log: $LOG_FILE"

    mkdir -p "$CHECKPOINT_DIR" "$(dirname "$CHECKPOINT_DIR")/output"

    START_TIME=$(date +%s)
    if eval "$COMMAND" > "$LOG_FILE" 2>&1; then
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
    else
        EXIT_CODE=$?
        END_TIME=$(date +%s)
        ELAPSED=$((END_TIME - START_TIME))
        echo "[FAILED] $task_id (exit code=$EXIT_CODE, elapsed=${ELAPSED}s)"
        echo "See log: $LOG_FILE"

        # Write status
        python3 -c "
import json, time
json.dump({'task_id':'$task_id','status':'FAILED','exit_code':$EXIT_CODE,'elapsed_seconds':$ELAPSED,'timestamp':time.time()}, open('$STATUS_JSON','w'), indent=2)
" 2>/dev/null || true

        # Update queue status
        python3 -c "
import json
s={'total':$TOTAL,'completed':$COMPLETED,'failed':1,'task_ids':$(python3 -c "import json; print(json.dumps(['$task_id']))"),'last_update':__import__('time').time()}
json.dump(s, open('$STATUS_FILE','w'), indent=2)
" 2>/dev/null || true

        echo ""
        echo "QUEUE STOPPED due to failure. Fix and resume."
        exit 1
    fi

    echo "[COMPLETE] $task_id (elapsed=${ELAPSED}s)"
    COMPLETED=$((COMPLETED + 1))

    # Write status
    python3 -c "
import json, time
json.dump({'task_id':'$task_id','status':'COMPLETE','elapsed_seconds':$ELAPSED,'timestamp':time.time()}, open('$STATUS_JSON','w'), indent=2)
" 2>/dev/null || true

    # Update queue status
    python3 -c "
import json
s={'total':$TOTAL,'completed':$COMPLETED,'failed':0,'current_task':'$task_id','last_update':__import__('time').time()}
json.dump(s, open('$STATUS_FILE','w'), indent=2)
" 2>/dev/null || true

    echo ""
done

echo ""
echo "=== Phase 4 Stage B queue COMPLETE ==="
echo "Total: $TOTAL, Completed: $COMPLETED"
echo "Finished at: $(date)"
