import json
import argparse
from pathlib import Path
import sys
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure kie_evaluator can be imported from the same directory
sys.path.append(str(Path(__file__).parent))
import kie_evaluator

def normalize_func(text, **kwargs):
    """Text normalization function - strictly follows evaluate_kie.py lines 12-16"""
    halfwidth_text = kie_evaluator.fullwidth_to_halfwidth(str(text))
    cleaned_text = kie_evaluator.remove_unnecessary_spaces(halfwidth_text)
    return cleaned_text

def load_json_files(directory: str, jobs: int = 1) -> dict:
    """Load all .txt files in a directory and parse as JSON"""
    data_dict = {}
    path = Path(directory)
    if not path.exists():
        print(f"Warning: Directory {directory} does not exist")
        return data_dict
    
    file_paths = list(path.glob("*.txt"))

    def parse_one(file_path: Path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Simple Markdown code block handling to ensure json.loads success
                if "```json" in content:
                    m = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL)
                    if m:
                        content = m.group(1)
                elif "```" in content:
                    m = re.search(r"```\s*(.*?)\s*```", content, re.DOTALL)
                    if m:
                        content = m.group(1)

                data = json.loads(content)
                return file_path.name, data
        except Exception:
            return None

    jobs = max(1, int(jobs))
    if jobs == 1:
        for file_path in file_paths:
            parsed = parse_one(file_path)
            if parsed:
                data_dict[parsed[0]] = parsed[1]
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(parse_one, file_path) for file_path in file_paths]
            for fut in as_completed(futures):
                parsed = fut.result()
                if parsed:
                    data_dict[parsed[0]] = parsed[1]

    return data_dict

def main():
    parser = argparse.ArgumentParser(description="Quickly evaluate KIE accuracy for directory-based predictions")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .txt files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth .txt files")
    parser.add_argument("--jobs", type=int, default=1, help="dataset internal parallel workers")
    
    args = parser.parse_args()
    
    predictions = load_json_files(args.pred_dir, jobs=args.jobs)
    ground_truth = load_json_files(args.gt_dir, jobs=args.jobs)
    
    if not predictions or not ground_truth:
        print("Error: No predictions or ground truth loaded.")
        return

    normalized_preds = kie_evaluator.normalize_values_of_nested_dict(predictions, normalize_func)
    normalized_gts = kie_evaluator.normalize_values_of_nested_dict(ground_truth, normalize_func)
    
    f1_score, class_f1_info, f1_error_info = kie_evaluator.cal_f1_all(normalized_preds, normalized_gts)
    
    # Calculate per-file F1
    per_file = []
    for file_name, answer in normalized_gts.items():
        pred = normalized_preds.get(file_name, {})
        pred_flat, answer_flat = kie_evaluator.flatten(kie_evaluator.normalize_dict(pred)), kie_evaluator.flatten(kie_evaluator.normalize_dict(answer))
        sample_tp = 0
        sample_fn_or_fp = 0
        for field in pred_flat:
            if field in answer_flat:
                sample_tp += 1
                answer_flat.remove(field)
            else:
                sample_fn_or_fp += 1
        sample_fn_or_fp += len(answer_flat)
        sample_f1 = sample_tp / (sample_tp + sample_fn_or_fp / 2 + 1e-6)
        per_file.append({"file": file_name, "f1_score": sample_f1})
    
    matched = len(set(normalized_preds.keys()) & set(normalized_gts.keys()))
    
    report = {
        "pred_dir": args.pred_dir,
        "gt_dir": args.gt_dir,
        "matched_samples": matched,
        "f1_score": f1_score,
        "per_file": per_file
    }
    
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nEvaluation Results (Strictly following evaluate_kie.py logic):")
    print(f"  Matched samples: {matched}")
    print(f"  F1 Score: {f1_score:.4f}")

if __name__ == "__main__":
    main()
