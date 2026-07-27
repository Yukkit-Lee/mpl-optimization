#!/bin/bash
# Auto-chain: Phase 5 Stage C -> Phase 6 target freeze -> Phase 7 final test -> Phase 8 figures
# Run AFTER Phase 4 Stage B queue is 9/9 COMPLETE.
# This script is restart-safe: each phase checks its own completion.

set -euo pipefail

source /root/.config/new-secure-feature-pipeline.env

PROJECT="$NEW_PROJECT_ROOT"
LOG_DIR="$PROJECT/new_log/new_incremental_comparison/txt"
mkdir -p "$LOG_DIR"

echo "============================================"
echo "Phase 5-8 Auto-Chain"
echo "Started: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================"

# ---- Phase 5: Stage C Adaptive (9 tasks) ----
echo ""
echo "=== PHASE 5: Full Data Adaptive Stage C ==="
STAGE_C_STATUS="$PROJECT/new_log/new_incremental_comparison/json/new_stage_c_queue_status.json"
STAGE_C_COMPLETE=$(python3 -c "
import json, os
if os.path.exists('$STAGE_C_STATUS'):
    d=json.load(open('$STAGE_C_STATUS'))
    print(d.get('completed',0))
else:
    print(0)
" 2>/dev/null || echo "0")

if [ "$STAGE_C_COMPLETE" -ge 9 ]; then
    echo "Phase 5: 9/9 already COMPLETE, skipping"
else
    echo "Running Stage C queue..."
    bash "$PROJECT/new_scripts/new_run_incremental_stage_c_queue.sh" 2>&1 | tee "$LOG_DIR/phase5_stage_c_queue.log"
    PHASE5_EXIT=${PIPESTATUS[0]}
    if [ "$PHASE5_EXIT" -ne 0 ]; then
        echo "Phase 5 FAILED. See log: $LOG_DIR/phase5_stage_c_queue.log"
        exit 1
    fi
fi

# ---- Phase 6: Reference aggregation + target freeze ----
echo ""
echo "=== PHASE 6: Reference Aggregation + Target Freeze ==="
TARGET_JSON="$PROJECT/new_log/new_incremental_comparison/json/new_target_freeze.json"
if [ -f "$TARGET_JSON" ]; then
    echo "Phase 6: target already frozen, skipping"
else
    echo "Running Phase 6 aggregation..."
    /root/autodl-tmp/new_envs/distillation/bin/python \
        "$PROJECT/new_report_incremental_full_feature.py" \
        --phase6-only 2>&1 | tee "$LOG_DIR/phase6_target_freeze.log"
    PHASE6_EXIT=${PIPESTATUS[0]}
    if [ "$PHASE6_EXIT" -ne 0 ]; then
        echo "Phase 6 FAILED. See log: $LOG_DIR/phase6_target_freeze.log"
        exit 1
    fi
fi

# ---- Phase 7: Final test evaluation ----
echo ""
echo "=== PHASE 7: Final Test Evaluation ==="
FINAL_TEST_JSON="$PROJECT/new_log/new_incremental_comparison/json/new_final_test_incremental.json"
if [ -f "$FINAL_TEST_JSON" ]; then
    echo "Phase 7: final test already done, skipping"
else
    echo "Running Phase 7 final test..."
    /root/autodl-tmp/new_envs/distillation/bin/python \
        "$PROJECT/new_report_incremental_full_feature.py" \
        --final-test 2>&1 | tee "$LOG_DIR/phase7_final_test.log"
    PHASE7_EXIT=${PIPESTATUS[0]}
    if [ "$PHASE7_EXIT" -ne 0 ]; then
        echo "Phase 7 FAILED. See log: $LOG_DIR/phase7_final_test.log"
        exit 1
    fi
fi

# ---- Phase 8: Figures + Report ----
echo ""
echo "=== PHASE 8: Figure Generation + Final Report ==="
FIG_PNG="$PROJECT/new_expFig/new_incremental_comparison/new_figure2_full_feature_vs_ours.png"
if [ -f "$FIG_PNG" ]; then
    echo "Phase 8: figures already exist, skipping"
else
    echo "Running Phase 8 figure generation..."
    /root/autodl-tmp/new_envs/distillation/bin/python \
        "$PROJECT/new_report_incremental_full_feature.py" \
        --phase8 2>&1 | tee "$LOG_DIR/phase8_figures.log"
    PHASE8_EXIT=${PIPESTATUS[0]}
    if [ "$PHASE8_EXIT" -ne 0 ]; then
        echo "Phase 8 FAILED. See log: $LOG_DIR/phase8_figures.log"
        exit 1
    fi
fi

echo ""
echo "============================================"
echo "Phase 5-8 Auto-Chain COMPLETE"
echo "Finished: $(date +'%Y-%m-%d %H:%M:%S')"
echo "============================================"
