#!/usr/bin/env bash
set -euo pipefail

# Evaluate all OCRBench tasks for one model result directory.
#
# Usage:
#   bash scripts/eval_all_tasks.sh <model_name_or_sanitized> [jobs]
#
# Example:
#   bash scripts/eval_all_tasks.sh gemini-3_1-flash-lite-preview
#   bash scripts/eval_all_tasks.sh gemini-3_1-flash-lite-preview 8
#
# Optional:
#   SKIP_DOC_PARSING_EVAL=1  — 仍打印「### 5) Doc Parsing ###」，但不跑 doc_parsing 评测（省时间）。

if [[ $# -lt 1 ]]; then
  echo "Usage: bash scripts/eval_all_tasks.sh <model_name_or_sanitized> [jobs]"
  exit 1
fi

MODEL_RAW="$1"
JOBS="${2:-1}"
# Keep consistent with evaluate_results.py sanitize_model_name()
MODEL_SANITIZED="$(echo "$MODEL_RAW" | sed 's/[^[:alnum:]_-]/_/g')"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_EVAL="${REPO_ROOT}/src/evaluate_results.py"
RESULT_ROOT="${REPO_ROOT}/results/${MODEL_SANITIZED}"
DATA_ROOT="${REPO_ROOT}/ocr_datasets"
PRED_SUBDIR="pred_${MODEL_SANITIZED}"

if [[ ! -f "${SRC_EVAL}" ]]; then
  echo "Error: evaluate entry not found: ${SRC_EVAL}"
  exit 1
fi

if [[ ! -d "${RESULT_ROOT}" ]]; then
  echo "Error: result root not found: ${RESULT_ROOT}"
  exit 1
fi

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "Error: dataset root not found: ${DATA_ROOT}"
  exit 1
fi

ts="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${RESULT_ROOT}/eval_logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/eval_all_tasks_${ts}.log"
declare -A METRIC_SUM
declare -A METRIC_CNT

echo "============================================================" | tee -a "${LOG_FILE}"
echo "Model: ${MODEL_RAW} (sanitized: ${MODEL_SANITIZED})" | tee -a "${LOG_FILE}"
echo "Result root: ${RESULT_ROOT}" | tee -a "${LOG_FILE}"
echo "Dataset root: ${DATA_ROOT}" | tee -a "${LOG_FILE}"
echo "Internal jobs per dataset: ${JOBS}" | tee -a "${LOG_FILE}"
echo "Log file: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"

run_cmd() {
  local _tmp_out
  _tmp_out="$(mktemp)"
  echo "" | tee -a "${LOG_FILE}"
  echo ">>> $*" | tee -a "${LOG_FILE}"
  if "$@" >"${_tmp_out}" 2>&1; then
    cat "${_tmp_out}" >> "${LOG_FILE}"
    rm -f "${_tmp_out}"
    return 0
  fi
  cat "${_tmp_out}" >> "${LOG_FILE}"
  rm -f "${_tmp_out}"
  echo "[WARN] command failed, continue. (details in log)" | tee -a "${LOG_FILE}"
  return 1
}

extract_last_value() {
  local pattern="$1"
  local file="$2"
  awk -v pat="${pattern}" '
    $0 ~ pat {
      v=$0
    }
    END {
      if (v != "") print v
    }
  ' "${file}"
}

run_and_print_mean() {
  local task_name="$1"
  local task_group="$2"
  local ds_name="$3"
  shift 3

  local _tmp_out
  _tmp_out="$(mktemp)"

  echo "" >> "${LOG_FILE}"
  echo ">>> [$task_name][$task_group][$ds_name] $*" >> "${LOG_FILE}"

  if ! "$@" >"${_tmp_out}" 2>&1; then
    cat "${_tmp_out}" >> "${LOG_FILE}"
    rm -f "${_tmp_out}"
    echo "[WARN][$task_name][$task_group][$ds_name] failed (details in log)" | tee -a "${LOG_FILE}"
    return 1
  fi

  cat "${_tmp_out}" >> "${LOG_FILE}"

  # Parse mean-like metrics per task and record for final summary only.
  record_metric() {
    local key="$1"
    local val="$2"
    if [[ -z "${val}" ]]; then
      return 0
    fi
    if [[ ! "${val}" =~ ^-?[0-9]+([.][0-9]+)?$ ]]; then
      return 0
    fi
    local prev="${METRIC_SUM[$key]:-0}"
    METRIC_SUM["$key"]="$(awk -v a="${prev}" -v b="${val}" 'BEGIN{printf "%.12f", a+b}')"
    METRIC_CNT["$key"]=$(( ${METRIC_CNT[$key]:-0} + 1 ))
    
    # 实时打印每个三级子数据集的得分 (仅打印一次，避免重复打印)
    if [[ "${key}" == *"${task_group}"* && "${ds_name}" != "all" ]]; then
        echo "  -> [${ds_name}] ${key} = ${val}" | tee -a "${LOG_FILE}"
    fi
  }

  if [[ "${task_name}" == "kie" ]]; then
    local f1
    f1="$(awk '/F1 Score:/ {v=$NF} END{print v}' "${_tmp_out}")"
    record_metric "kie.mean_f1" "${f1}"
    record_metric "kie.${task_group}.mean_f1" "${f1}"
  elif [[ "${task_name}" == "recognition" ]]; then
    local macro_f1 micro_f1
    macro_f1="$(awk -F': ' '/"macro_f1_score":/ {gsub(/,/, "", $2); v=$2} END{print v}' "${_tmp_out}")"
    micro_f1="$(awk -F': ' '/"mirco_f1_score":/ {gsub(/,/, "", $2); v=$2} END{print v}' "${_tmp_out}")"
    record_metric "recognition.mean_macro_f1" "${macro_f1}"
    record_metric "recognition.mean_micro_f1" "${micro_f1}"
    record_metric "recognition.${task_group}.mean_macro_f1" "${macro_f1}"
    record_metric "recognition.${task_group}.mean_micro_f1" "${micro_f1}"
  elif [[ "${task_name}" == "vqa" ]]; then
    local ms
    ms="$(awk '/overall mean_score:/ {v=$NF} END{print v}' "${_tmp_out}")"
    record_metric "vqa.mean_score" "${ms}"
    record_metric "vqa.${task_group}.mean_score" "${ms}"
  elif [[ "${task_name}" == "grounding" ]]; then
    local tmiou omiou
    tmiou="$(awk -F': ' '/"text_grounding"/{in_t=1;in_o=0} /"object_grounding"/{in_t=0;in_o=1} in_t && /"mean_iou":/{gsub(/,/, "", $2); v=$2; print v; exit}' "${_tmp_out}")"
    omiou="$(awk -F': ' '/"text_grounding"/{in_t=1;in_o=0} /"object_grounding"/{in_t=0;in_o=1} in_o && /"mean_iou":/{gsub(/,/, "", $2); v=$2; print v; exit}' "${_tmp_out}")"
    
    # 从 JSON 输出中提取并打印每个三级子数据集的得分
    awk -F': ' '
      /"text_grounding"/{in_t=1;in_o=0} 
      /"object_grounding"/{in_t=0;in_o=1} 
      /"subset_means": \{/ {in_sub=1; next}
      /\}/ {if(in_sub) {in_sub=0}}
      in_sub && /"[^"]+": [0-9.]+/ {
        gsub(/[" ,]/, "", $1); gsub(/,/, "", $2);
        if(in_t) print "  -> ["$1"] grounding.text.mean_iou = "$2;
        if(in_o) print "  -> ["$1"] grounding.object.mean_iou = "$2;
      }
    ' "${_tmp_out}" | tee -a "${LOG_FILE}"

    record_metric "grounding.text.mean_iou" "${tmiou}"
    record_metric "grounding.object.mean_iou" "${omiou}"
    record_metric "grounding.mean_iou" "${tmiou}"
    record_metric "grounding.mean_iou" "${omiou}"
  elif [[ "${task_name}" == "parsing" ]]; then
    local mean_edit mean_teds mean_combined dp_score
    mean_edit="$(awk '/mean edit similarity:/ {v=$NF} END{print v}' "${_tmp_out}")"
    mean_teds="$(awk '/mean TEDS \(all paired tables\):/ {v=$NF} END{print v} /mean TEDS:/ {v=$NF} END{if (v!="") print v}' "${_tmp_out}" | awk 'END{print $0}')"
    mean_combined="$(awk '/mean combined \(table w=[0-9.]+\):/ {v=$NF} END{print v}' "${_tmp_out}")"
    
    # Determine the primary score for overall mean
    if [[ -n "${mean_combined}" ]]; then
      dp_score="${mean_combined}"
    elif [[ -n "${mean_teds}" ]]; then
      dp_score="${mean_teds}"
    else
      dp_score="${mean_edit}"
    fi
    record_metric "parsing.mean_score" "${dp_score}"
    record_metric "parsing.${task_group}.mean_score" "${dp_score}"
  fi

  rm -f "${_tmp_out}"
  return 0
}

eval_dir_task() {
  local task_name="$1"     # kie|recognition|vqa|parsing
  local task_group="$2"    # business_transactions|public_services|...
  local result_task_root="$3"
  local gt_root="$4"

  if [[ ! -d "${result_task_root}" ]]; then
    echo "[SKIP] result task dir missing: ${result_task_root}" | tee -a "${LOG_FILE}"
    return 0
  fi
  if [[ ! -d "${gt_root}" ]]; then
    echo "[SKIP] gt root missing: ${gt_root}" | tee -a "${LOG_FILE}"
    return 0
  fi

  # Only iterate dataset directories; ignore files such as run_meta_*.jsonl.
  local ds_list=()
  local ds_path
  for ds_path in "${result_task_root}"/*; do
    [[ -d "${ds_path}" ]] || continue
    ds_list+=("$(basename "${ds_path}")")
  done
  if [[ ${#ds_list[@]} -gt 0 ]]; then
    mapfile -t ds_list < <(printf '%s\n' "${ds_list[@]}" | sort)
  fi
  for ds in "${ds_list[@]}"; do
    local pred_dir="${result_task_root}/${ds}/${PRED_SUBDIR}"
    local gt_dir="${gt_root}/${ds}"
    if [[ ! -d "${pred_dir}" ]]; then
      echo "[SKIP][${task_name}] pred dir missing: ${pred_dir}" | tee -a "${LOG_FILE}"
      continue
    fi
    if [[ ! -d "${gt_dir}" ]]; then
      echo "[SKIP][${task_name}] gt dir missing: ${gt_dir}" | tee -a "${LOG_FILE}"
      continue
    fi

    run_and_print_mean "${task_name}" "${task_group}" "${ds}" python "${SRC_EVAL}" \
      --task "${task_name}" \
      --pred_dir "${pred_dir}" \
      --gt_dir "${gt_dir}" \
      --jobs "${JOBS}"
  done
}

echo "" | tee -a "${LOG_FILE}"
echo "### 1) KIE (Extraction) ###" | tee -a "${LOG_FILE}"
for task_group in business_transactions public_services regulated_records; do
  eval_dir_task "kie" "${task_group}" \
    "${RESULT_ROOT}/extraction/${task_group}" \
    "${DATA_ROOT}/extraction/${task_group}/answer"
done

echo "" | tee -a "${LOG_FILE}"
echo "### 2) Recognition ###" | tee -a "${LOG_FILE}"
for task_group in multi_lingual_recognition natural_scene_recognition; do
  eval_dir_task "recognition" "${task_group}" \
    "${RESULT_ROOT}/recognition/${task_group}" \
    "${DATA_ROOT}/recognition/${task_group}/answer"
done

echo "" | tee -a "${LOG_FILE}"
echo "### 3) VQA (QA) ###" | tee -a "${LOG_FILE}"
for task_group in dashboard_qa financial_documents_qa scientific_documents_qa user_interface_qa; do
  eval_dir_task "vqa" "${task_group}" \
    "${RESULT_ROOT}/qa/${task_group}" \
    "${DATA_ROOT}/qa/${task_group}/answer"
done

echo "" | tee -a "${LOG_FILE}"
echo "### 4) Grounding ###" | tee -a "${LOG_FILE}"
run_and_print_mean "grounding" "all" "all" python "${SRC_EVAL}" \
  --task grounding \
  --gt_grounding_root "${DATA_ROOT}/grounding" \
  --pred_grounding_root "${RESULT_ROOT}/grounding" \
  --pred_folder "${PRED_SUBDIR}" \
  --jobs "${JOBS}"

echo "" | tee -a "${LOG_FILE}"
echo "### 5) Doc Parsing (Parsing) ###" | tee -a "${LOG_FILE}"
if [[ "${SKIP_DOC_PARSING_EVAL:-0}" == "1" || "${SKIP_DOC_PARSING_EVAL:-}" == "true" ]]; then
  echo "[SKIP] Doc Parsing: SKIP_DOC_PARSING_EVAL=1 — 仅展示本节标题，不执行评测。" | tee -a "${LOG_FILE}"
elif [[ -d "${RESULT_ROOT}/parsing" && -d "${DATA_ROOT}/parsing" ]]; then
  for task_group in complex_table_parsing formula_parsing general_documents_parsing info_board_parsing molecular_parsing; do
    result_task_root="${RESULT_ROOT}/parsing/${task_group}"
    
    if [[ ! -d "${result_task_root}" ]]; then
      continue
    fi
    
    dp_ds_list=()
    for ds_path in "${result_task_root}"/*; do
      [[ -d "${ds_path}" ]] || continue
      dp_ds_list+=("$(basename "${ds_path}")")
    done
    if [[ ${#dp_ds_list[@]} -gt 0 ]]; then
      mapfile -t dp_ds_list < <(printf '%s\n' "${dp_ds_list[@]}" | sort)
    fi
    for ds in "${dp_ds_list[@]}"; do
      pred_dir="${result_task_root}/${ds}/${PRED_SUBDIR}"
      if [[ ! -d "${pred_dir}" ]]; then
        echo "[SKIP][parsing] pred dir missing: ${pred_dir}" | tee -a "${LOG_FILE}"
        continue
      fi
      # parsing supports --pred + --dataset, and auto-resolves GT.
      run_and_print_mean "parsing" "${task_group}" "${ds}" python "${SRC_EVAL}" \
        --task parsing \
        --pred "${pred_dir}" \
        --dataset "${ds}" \
        --jobs "${JOBS}"
    done
  done
else
  echo "[SKIP] parsing roots missing." | tee -a "${LOG_FILE}"
fi

echo "" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
echo "All tasks evaluation finished." | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Final grouped mean scores:" | tee -a "${LOG_FILE}"

# Helper function to print metric if exists
print_metric() {
  local key="$1"
  local cnt="${METRIC_CNT[$key]:-0}"
  if [[ "${cnt}" -gt 0 ]]; then
    local sum="${METRIC_SUM[$key]}"
    local avg="$(awk -v s="${sum}" -v c="${cnt}" 'BEGIN{printf "%.6f", s/c}')"
    echo "  ${key} = ${avg}" | tee -a "${LOG_FILE}"
  fi
}

# 1. KIE (Extraction)
echo "" | tee -a "${LOG_FILE}"
echo "[KIE (Extraction)]" | tee -a "${LOG_FILE}"
print_metric "kie.mean_f1"
echo "  --- By task (Level 2) ---" | tee -a "${LOG_FILE}"
print_metric "kie.business_transactions.mean_f1"
print_metric "kie.public_services.mean_f1"
print_metric "kie.regulated_records.mean_f1"

# 2. Recognition
echo "" | tee -a "${LOG_FILE}"
echo "[Recognition]" | tee -a "${LOG_FILE}"
print_metric "recognition.mean_macro_f1"
print_metric "recognition.mean_micro_f1"
echo "  --- By task (Level 2) ---" | tee -a "${LOG_FILE}"
print_metric "recognition.multi_lingual_recognition.mean_macro_f1"
print_metric "recognition.multi_lingual_recognition.mean_micro_f1"
print_metric "recognition.natural_scene_recognition.mean_macro_f1"
print_metric "recognition.natural_scene_recognition.mean_micro_f1"

# 3. VQA (QA)
echo "" | tee -a "${LOG_FILE}"
echo "[VQA (QA)]" | tee -a "${LOG_FILE}"
print_metric "vqa.mean_score"
echo "  --- By task (Level 2) ---" | tee -a "${LOG_FILE}"
print_metric "vqa.dashboard_qa.mean_score"
print_metric "vqa.financial_documents_qa.mean_score"
print_metric "vqa.scientific_documents_qa.mean_score"
print_metric "vqa.user_interface_qa.mean_score"

# 4. Grounding
echo "" | tee -a "${LOG_FILE}"
echo "[Grounding]" | tee -a "${LOG_FILE}"
print_metric "grounding.mean_iou"
echo "  --- By task (Level 2) ---" | tee -a "${LOG_FILE}"
print_metric "grounding.text.mean_iou"
print_metric "grounding.object.mean_iou"

# 5. Doc Parsing (Parsing)
echo "" | tee -a "${LOG_FILE}"
echo "[Doc Parsing (Parsing)]" | tee -a "${LOG_FILE}"
print_metric "parsing.mean_score"
echo "  --- By task (Level 2) ---" | tee -a "${LOG_FILE}"
print_metric "parsing.complex_table_parsing.mean_score"
print_metric "parsing.formula_parsing.mean_score"
print_metric "parsing.general_documents_parsing.mean_score"
print_metric "parsing.info_board_parsing.mean_score"
print_metric "parsing.molecular_parsing.mean_score"

echo "" | tee -a "${LOG_FILE}"
echo "Log saved: ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "============================================================" | tee -a "${LOG_FILE}"
