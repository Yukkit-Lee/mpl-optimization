#!/usr/bin/env bash
# 用途：幂等串联 Module C → End-to-End baseline → End-to-End → TFE system。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/new_common_env.sh"

ACTIVE="${NEW_PROJECT_ROOT}/new_configs/new_secure_feature/new_active_single_gpu_queue.txt"
STATUS="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_public_mtt_mainline_chain.json"
MODULE_C_QUEUE="${NEW_PROJECT_ROOT}/new_configs/new_public_mtt_mainline/new_tfe_module_c_formal.tsv"
BASELINE_QUEUE="${NEW_PROJECT_ROOT}/new_configs/new_public_mtt_mainline/new_tfe_end_to_end_baseline_stage_b.tsv"
END_TO_END_QUEUE="${NEW_PROJECT_ROOT}/new_configs/new_public_mtt_mainline/new_tfe_end_to_end_formal.tsv"
SYSTEM_QUEUE="${NEW_PROJECT_ROOT}/new_configs/new_public_mtt_mainline/new_tfe_system_formal.tsv"
MODULE_C_REPORT="${NEW_PROJECT_ROOT}/new_log/new_public_mtt_mainline/json/new_module_c_formal_report.json"
END_TO_END_REPORT="${NEW_PROJECT_ROOT}/new_log/new_public_mtt_mainline/json/new_end_to_end_formal_report.json"
FINAL_TEST_REPORT="${NEW_PROJECT_ROOT}/new_log/new_public_mtt_mainline/json/new_final_test_evaluation.json"
SYSTEM_REPORT="${NEW_PROJECT_ROOT}/new_log/new_public_mtt_mainline/json/new_tfe_system_formal_report.json"
mkdir -p "$(dirname "${STATUS}")"

atomic_status() {
  local stage="$1"
  local state="$2"
  local detail="${3:-}"
  local temporary="${STATUS}.new_tmp_$$"
  printf '{"status":"%s","stage":"%s","detail":"%s","checked_at":"%s"}\n' \
    "${state}" "${stage}" "${detail}" "$(date --iso-8601=seconds)" > "${temporary}"
  mv -f "${temporary}" "${STATUS}"
}

queue_state() {
  local queue="$1"
  local task_id checkpoint command status seen=0 complete=0
  while IFS=$'\t' read -r task_id checkpoint command; do
    [[ -z "${task_id}" || "${task_id}" == \#* ]] && continue
    seen=$((seen + 1))
    status="${NEW_OUTPUT_ROOT}/new_log/new_secure_feature/status/new_${task_id}.json"
    if [[ -f "${status}" ]] && grep -q '"status":"FAILED"' "${status}"; then
      printf 'FAILED:%s' "${task_id}"
      return 0
    fi
    if [[ -f "${status}" ]] && grep -q '"status":"COMPLETE"' "${status}"; then
      complete=$((complete + 1))
    fi
  done < "${queue}"
  if (( seen > 0 && complete == seen )); then
    printf 'COMPLETE:%d/%d' "${complete}" "${seen}"
  else
    printf 'RUNNING:%d/%d' "${complete}" "${seen}"
  fi
}

wait_queue() {
  local stage="$1"
  local queue="$2"
  local state
  while true; do
    state="$(queue_state "${queue}")"
    case "${state}" in
      COMPLETE:*)
        atomic_status "${stage}" "STAGE_COMPLETE" "${state#*:}"
        return 0
        ;;
      FAILED:*)
        atomic_status "${stage}" "NEEDS_ATTENTION" "${state#*:}"
        return 1
        ;;
      RUNNING:*)
        atomic_status "${stage}" "WAITING" "${state#*:}"
        sleep 30
        ;;
      *)
        atomic_status "${stage}" "NEEDS_ATTENTION" "invalid_queue_state"
        return 1
        ;;
    esac
  done
}

active_queue_is() {
  local queue="$1"
  [[ -s "${ACTIVE}" ]] && [[ "$(head -n 1 "${ACTIVE}")" == $'tfe\t'"${queue}" ]]
}

activate_if_needed() {
  local stage="$1"
  local queue="$2"
  shift 2
  if active_queue_is "${queue}" || pgrep -af "new_run_task_queue.sh tfe ${queue}" >/dev/null; then
    atomic_status "${stage}" "ALREADY_ACTIVE" "${queue}"
    return 0
  fi
  atomic_status "${stage}" "ACTIVATING" "${queue}"
  "$@"
}

if [[ ! -f "${MODULE_C_REPORT}" ]]; then
  wait_queue "module_c_formal" "${MODULE_C_QUEUE}"
  atomic_status "module_c_formal" "REPORTING"
  "${NEW_MODERN_PYTHON}" "${NEW_PROJECT_ROOT}/new_report_module_c_formal.py"
fi

if [[ "$(queue_state "${BASELINE_QUEUE}")" != COMPLETE:* ]]; then
  activate_if_needed \
    "end_to_end_baseline_stage_b" \
    "${BASELINE_QUEUE}" \
    "${NEW_MODERN_PYTHON}" \
    "${NEW_PROJECT_ROOT}/new_prepare_end_to_end_baseline_stage_b.py" \
    --activate
  wait_queue "end_to_end_baseline_stage_b" "${BASELINE_QUEUE}"
fi

if [[ ! -f "${END_TO_END_REPORT}" ]]; then
  if [[ ! -f "${END_TO_END_QUEUE}" ]] || [[ "$(queue_state "${END_TO_END_QUEUE}")" != COMPLETE:* ]]; then
    activate_if_needed \
      "end_to_end_formal" \
      "${END_TO_END_QUEUE}" \
      "${NEW_MODERN_PYTHON}" \
      "${NEW_PROJECT_ROOT}/new_prepare_end_to_end_formal.py" \
      --activate
    wait_queue "end_to_end_formal" "${END_TO_END_QUEUE}"
  fi
  atomic_status "end_to_end_formal" "REPORTING"
  "${NEW_MODERN_PYTHON}" "${NEW_PROJECT_ROOT}/new_report_end_to_end_formal.py"
fi

if [[ ! -f "${FINAL_TEST_REPORT}" ]]; then
  atomic_status "final_test_evaluation" "REPORTING" \
    "single_readonly_after_all_config_and_models_frozen"
  "${NEW_MODERN_PYTHON}" "${NEW_PROJECT_ROOT}/new_final_test_evaluation.py"
fi

if [[ ! -f "${SYSTEM_REPORT}" ]]; then
  if [[ ! -f "${SYSTEM_QUEUE}" ]] || [[ "$(queue_state "${SYSTEM_QUEUE}")" != COMPLETE:* ]]; then
    activate_if_needed \
      "tfe_system_formal" \
      "${SYSTEM_QUEUE}" \
      "${NEW_MODERN_PYTHON}" \
      "${NEW_PROJECT_ROOT}/new_prepare_tfe_system_formal.py" \
      --activate
    wait_queue "tfe_system_formal" "${SYSTEM_QUEUE}"
  fi
  atomic_status "tfe_system_formal" "REPORTING"
  "${NEW_MODERN_PYTHON}" "${NEW_PROJECT_ROOT}/new_report_tfe_system_formal.py"
fi

atomic_status "mainline_remaining" "COMPLETE" \
  "module_c,end_to_end,final_test,tfe_system"
