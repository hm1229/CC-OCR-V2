#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_ROOT}/results"

if [[ ! -d "${RESULTS_DIR}" ]]; then
  echo "Error: results directory not found at ${RESULTS_DIR}"
  exit 1
fi

echo "============================================================"
echo "🚀 开始一键评测所有模型..."
echo "============================================================"

# 遍历 results 目录下的所有模型文件夹
for model_path in "${RESULTS_DIR}"/*; do
  if [[ -d "${model_path}" ]]; then
    model_name="$(basename "${model_path}")"
    echo ""
    echo "▶️ 正在评测模型: ${model_name}"
    
    # 调用单模型评测脚本，默认开启 8 进程加速（可根据服务器配置调整）
    bash "${REPO_ROOT}/scripts/eval_all_tasks_one_model.sh" "${model_name}" 8
  fi
done

echo ""
echo "============================================================"
echo "🎉 所有模型评测完成！"
echo "============================================================"
