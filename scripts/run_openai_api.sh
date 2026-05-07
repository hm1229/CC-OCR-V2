#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "USE: $0 <MODEL_NAME>"
  echo "Set OCR_ROOTS in this script to ocr_datasets task dirs (each must have question/ and images/)."
  exit 1
fi

MODEL="$1"
# 所有任务根目录（每个都包含 question/ 和 images/），按需注释掉不需要的
OCR_ROOTS=(
  ocr_datasets/grounding/text_grounding
  ocr_datasets/grounding/object_grounding
  ocr_datasets/vqa
  ocr_datasets/recognition/multi_scene_recognition
  ocr_datasets/recognition/multi_lan_recognition
  ocr_datasets/kie
  ocr_datasets/doc_parsing
)

for root in "${OCR_ROOTS[@]}"; do
  echo "==> Running: --ocr-root ${root} with model ${MODEL}"
  python src/request_openai.py --skip-existing --ocr-root "${root}" --model "${MODEL}" --api-key "" --api-base ""
done
