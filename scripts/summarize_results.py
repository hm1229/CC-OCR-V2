#!/usr/bin/env python3
import os
import glob
import re
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "results"

def get_all_logs_sorted(model_dir):
    log_dir = model_dir / "eval_logs"
    if not log_dir.exists():
        return []
    logs = list(log_dir.glob("eval_all_tasks_*.log"))
    # 按修改时间排序，取最新的在前面
    logs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return logs

def parse_log(log_path):
    metrics = {}
    with open(log_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 找到 "Final grouped mean scores:" 之后的内容
    start_idx = -1
    for i, line in enumerate(lines):
        if "Final grouped mean scores:" in line:
            start_idx = i
            break
            
    if start_idx == -1:
        return metrics
        
    for line in lines[start_idx:]:
        line = line.strip()
        # 匹配 key = value 格式，例如：kie.mean_f1 = 0.850000
        m = re.match(r"^([a-zA-Z0-9_.]+)\s*=\s*([0-9.]+)$", line)
        if m:
            key, val = m.groups()
            metrics[key] = float(val) * 100 # 转换为百分制
            
    return metrics

def extract_json_blocks(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = []
    brace_level = 0
    in_string = False
    escape = False
    start_idx = -1
    
    for i, char in enumerate(content):
        if char == '"' and not escape:
            in_string = not in_string
        elif char == '\\' and in_string:
            escape = not escape
        else:
            escape = False
            
        if not in_string:
            if char == '{':
                if brace_level == 0:
                    start_idx = i
                brace_level += 1
            elif char == '}':
                brace_level -= 1
                if brace_level == 0 and start_idx != -1:
                    json_str = content[start_idx:i+1]
                    try:
                        obj = json.loads(json_str)
                        if "per_file" in obj or "text_grounding" in obj or "object_grounding" in obj or ("results" in obj and "per_file" in obj["results"]):
                            blocks.append(obj)
                    except json.JSONDecodeError:
                        pass
                    start_idx = -1
                    
    return blocks

def get_model_category(model_name):
    model_lower = model_name.lower()
    # 根据模型名称中的关键词判断是否为端侧模型 (Device)
    if "10b" in model_lower or "8b" in model_lower or "9b" in model_lower or "minicpm" in model_lower:
        return "On-Device LMMs"
    return "On-Server LMMs"

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist.")
        return

    all_data = []
    all_sample_rows = []
    
    for model_dir in RESULTS_DIR.iterdir():
        if not model_dir.is_dir():
            continue
            
        model_name = model_dir.name
        logs = get_all_logs_sorted(model_dir)
        
        if not logs:
            continue
            
        # 1. 提取大任务汇总得分 (从新到旧合并，保留最新的有效分数)
        combined_metrics = {}
        for log in logs:
            metrics = parse_log(log)
            for k, v in metrics.items():
                if k not in combined_metrics:
                    combined_metrics[k] = v
                    
        if combined_metrics:
            rec_score = combined_metrics.get('recognition.mean_micro_f1', combined_metrics.get('recognition.mean_macro_f1', None))
            parse_score = combined_metrics.get('parsing.mean_score', None)
            grd_score = combined_metrics.get('grounding.mean_iou', None)
            kie_score = combined_metrics.get('kie.mean_f1', None)
            qa_score = combined_metrics.get('vqa.mean_score', None)
            
            valid_scores = [s for s in [rec_score, parse_score, grd_score, kie_score, qa_score] if s is not None]
            avg_score = sum(valid_scores) / len(valid_scores) if len(valid_scores) == 5 else None
            
            category = get_model_category(model_name)
            
            def fmt(val):
                return f"{val:.2f}" if val is not None else "-"
                
            all_data.append({
                'Category': category,
                'Model': model_name,
                'Recognition': fmt(rec_score),
                'Parsing': fmt(parse_score),
                'Grounding': fmt(grd_score),
                'Extraction': fmt(kie_score),
                'QA': fmt(qa_score),
                'Average': fmt(avg_score),
                '_avg_sort': avg_score if avg_score is not None else -1
            })

        # 2. 提取单样本明细得分 (从新到旧合并，避免重复提取相同的 dataset)
        seen_datasets = set()
        for log in logs:
            blocks = extract_json_blocks(log)
            for block in blocks:
                # Handle Grounding which has nested tasks
                if "text_grounding" in block or "object_grounding" in block:
                    if "grounding" in seen_datasets:
                        continue
                    seen_datasets.add("grounding")
                    
                    for t in ["text_grounding", "object_grounding"]:
                        if t in block and "per_file" in block[t]:
                            for item in block[t]["per_file"]:
                                all_sample_rows.append({
                                    "Model": model_name,
                                    "Task": "grounding",
                                    "Dataset": item.get("subset", "unknown"),
                                    "File": item.get("file", "unknown"),
                                    "Score_Type": "iou",
                                    "Score": item.get("iou", None)
                                })
                    continue

                # Handle other tasks
                gt_dir = Path(block.get("gt_dir", ""))
                dataset_name = gt_dir.name if gt_dir.name else "unknown"
                
                if dataset_name in seen_datasets:
                    continue
                seen_datasets.add(dataset_name)
                
                task = "unknown"
                if "parsing" in str(gt_dir): task = "parsing"
                elif "kie" in str(gt_dir): task = "kie"
                elif "vqa" in str(gt_dir): task = "vqa"
                elif "recognition" in str(gt_dir): task = "recognition"
                
                per_file_list = block.get("per_file", [])
                if not per_file_list and "results" in block:
                    per_file_list = block["results"].get("per_file", [])
                
                for item in per_file_list:
                    filename = item.get("file", "unknown")
                    
                    score = None
                    score_type = None
                    if "combined" in item:
                        score = item["combined"]
                        score_type = "combined"
                    elif "text_f1" in item:
                        score = item["text_f1"]
                        score_type = "text_f1"
                    elif "teds" in item:
                        score = item["teds"]
                        score_type = "teds"
                    elif "edit_similarity" in item:
                        score = item["edit_similarity"]
                        score_type = "edit_similarity"
                    elif "score" in item:  # VQA
                        score = item["score"]
                        score_type = "score"
                    elif "macro_f1" in item:  # Recognition
                        score = item["macro_f1"]
                        score_type = "macro_f1"
                    elif "f1_score" in item:  # KIE
                        score = item["f1_score"]
                        score_type = "f1_score"
                        
                    all_sample_rows.append({
                        "Model": model_name,
                        "Task": task,
                        "Dataset": dataset_name,
                        "File": filename,
                        "Score_Type": score_type,
                        "Score": score
                    })

    if not all_data:
        print("没有可用的评测结果。")
        return

    # --- 1. 输出大任务宏平均汇总表 ---
    all_data.sort(key=lambda x: (0 if 'Device' in x['Category'] else 1, x['_avg_sort']))
    for d in all_data:
        del d['_avg_sort']

    df = pd.DataFrame(all_data)
    
    print("\n" + "="*100)
    print("🏆 所有模型评测结果汇总 (百分制)")
    print("="*100 + "\n")
    
    df_device = df[df['Category'] == 'On-Device LMMs'].drop(columns=['Category'])
    if not df_device.empty:
        print("### On-Device LMMs")
        print(df_device.to_markdown(index=False, colalign=("left", "center", "center", "center", "center", "center", "center")))
        print("\n")
        
    df_server = df[df['Category'] == 'On-Server LMMs'].drop(columns=['Category'])
    if not df_server.empty:
        print("### On-Server LMMs")
        print(df_server.to_markdown(index=False, colalign=("left", "center", "center", "center", "center", "center", "center")))
        print("\n")
    
    out_csv = REPO_ROOT / "all_models_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"✅ 宏平均结果已保存至: {out_csv}")

    # --- 2. 输出单样本得分表 ---
    if all_sample_rows:
        df_samples = pd.DataFrame(all_sample_rows)
        out_csv_samples = REPO_ROOT / "all_models_all_tasks_sample_scores.csv"
        df_samples.to_csv(out_csv_samples, index=False)
        print(f"✅ 单样本得分已保存至: {out_csv_samples} (共 {len(df_samples)} 条记录)")

if __name__ == "__main__":
    main()
