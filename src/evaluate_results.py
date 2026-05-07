import json
import re
import sys
import argparse
import subprocess
from pathlib import Path
from collections import defaultdict

import kie_evaluator

SRC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SRC_DIR.parent
DATASETS_DIR = REPO_ROOT / "datasets"


def sanitize_model_name(model_name: str) -> str:
    return re.sub(r"[^\w\-_]", "_", model_name)


def run_evaluator_script(script: str, argv: list) -> int:
    cmd = [sys.executable, str(SRC_DIR / script)] + argv
    print("==>", " ".join(cmd))
    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


def normalize_func(text, **kwargs):
    """Text normalization function"""
    halfwidth_text = kie_evaluator.fullwidth_to_halfwidth(str(text))
    cleaned_text = kie_evaluator.remove_unnecessary_spaces(halfwidth_text)
    return cleaned_text


def extract_image_name(url: str) -> str:
    if url.startswith("images/"):
        return url[7:]
    return url


def load_predictions(pred_jsonl_path: str) -> dict:
    predictions = defaultdict(dict)
    
    print(f"Loading predictions: {pred_jsonl_path}")
    
    with open(pred_jsonl_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue
            
            try:
                data = json.loads(line.strip())
                
                # Get dataset name and URL
                dataset = data.get("dataset", "unknown")
                url = data.get("url", "")
                
                if not url:
                    print(f"Warning: Line {line_num} missing url field, skipping")
                    continue
                
                image_name = extract_image_name(url)
                model_result = data.get("model_result")
            
                if "error" in data:
                    print(f"Warning: {dataset}/{image_name} has error: {data['error']}")
                
                if model_result is None:
                    print(f"Warning: {dataset}/{image_name} has no prediction result")
                    continue
    
                if "_parse_error" in model_result:
                    print(f"Warning: {dataset}/{image_name} JSON parsing failed: {model_result.get('_parse_error')}")
                    continue
                
                predictions[dataset][image_name] = model_result
                
            except json.JSONDecodeError as e:
                print(f"Error: Line {line_num} JSON parsing failed: {e}")
                continue
            except Exception as e:
                print(f"Error: Line {line_num} processing failed: {e}")
                continue
    
    # Statistics
    total_samples = sum(len(preds) for preds in predictions.values())
    print(f"Successfully loaded {total_samples} predictions")
    for dataset, preds in predictions.items():
        print(f"  {dataset}: {len(preds)} samples")
    
    return dict(predictions)


def load_ground_truth(dataset_name: str) -> dict:
    """Load ground truth labels
    
    Args:
        dataset_name: Dataset name
    
    Returns:
        dict: {image_name: label_dict}
    """
    label_path = DATASETS_DIR / dataset_name / "label.json"
    
    if not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")
    
    print(f"Loading ground truth labels: {label_path}")
    
    with open(label_path, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    
    print(f"Loaded {len(labels)} ground truth labels")
    return labels


def evaluate_dataset(predictions: dict, ground_truth: dict, dataset_name: str) -> dict:
    """Evaluate a single dataset
    
    Args:
        predictions: {image_name: prediction_dict}
        ground_truth: {image_name: label_dict}
        dataset_name: Dataset name
    
    Returns:
        Evaluation result dictionary
    """
    print(f"\n{'='*60}")
    print(f"Evaluating dataset: {dataset_name}")
    print(f"{'='*60}")
    
    # Normalize predictions and ground truth
    normalized_preds = kie_evaluator.normalize_values_of_nested_dict(predictions, normalize_func)
    normalized_gts = kie_evaluator.normalize_values_of_nested_dict(ground_truth, normalize_func)
    
    # Calculate F1 score
    f1_score, class_f1_info, f1_error_info = kie_evaluator.cal_f1_all(normalized_preds, normalized_gts)
    
    # Statistics
    total_pred = len(normalized_preds)
    total_gt = len(normalized_gts)
    matched = len(set(normalized_preds.keys()) & set(normalized_gts.keys()))
    
    print(f"\nDataset statistics:")
    print(f"  Prediction samples: {total_pred}")
    print(f"  Ground truth samples: {total_gt}")
    print(f"  Matched samples: {matched}")
    
    print(f"\nEvaluation results:")
    print(f"  F1 Score: {f1_score:.4f}")
    
    # Return evaluation results
    eval_result = {
        "dataset": dataset_name,
        "summary": {
            "f1_score": f1_score,
            "total_predictions": total_pred,
            "total_ground_truth": total_gt,
            "matched_samples": matched
        },
        "class_f1_score": class_f1_info,
        "f1_error_info": f1_error_info
    }
    
    return eval_result


def run_jsonl_kie_eval(args) -> int:
    """UNIKIE JSONL + datasets/<name>/label.json（内置 kie_evaluator）。"""
    all_predictions = load_predictions(args.pred)
    if not all_predictions:
        print("Error: No predictions loaded")
        return 1

    if args.dataset:
        if args.dataset not in all_predictions:
            print(f"Error: Dataset {args.dataset} not found in predictions")
            print(f"Available datasets: {list(all_predictions.keys())}")
            return 1
        datasets_to_eval = [args.dataset]
    else:
        datasets_to_eval = list(all_predictions.keys())

    all_results = {}
    for dataset_name in datasets_to_eval:
        try:
            ground_truth = load_ground_truth(dataset_name)
            predictions = all_predictions[dataset_name]
            eval_result = evaluate_dataset(predictions, ground_truth, dataset_name)
            all_results[dataset_name] = eval_result
        except FileNotFoundError as e:
            print(f"Error: {e}")
            continue
        except Exception as e:
            print(f"Error: Failed to evaluate dataset {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Summary Results")
        print(f"{'='*60}")
        total_f1 = sum(r["summary"]["f1_score"] for r in all_results.values())
        total_pred = sum(r["summary"]["total_predictions"] for r in all_results.values())
        total_gt = sum(r["summary"]["total_ground_truth"] for r in all_results.values())
        total_matched = sum(r["summary"]["matched_samples"] for r in all_results.values())
        avg_f1 = total_f1 / len(all_results)
        print(f"\nOverall statistics:")
        print(f"  Number of datasets: {len(all_results)}")
        print(f"  Total prediction samples: {total_pred}")
        print(f"  Total ground truth samples: {total_gt}")
        print(f"  Total matched samples: {total_matched}")
        print(f"\nAverage evaluation results:")
        print(f"  Average F1 Score: {avg_f1:.4f}")
        all_results["_summary"] = {
            "num_datasets": len(all_results),
            "average_f1_score": avg_f1,
            "total_predictions": total_pred,
            "total_ground_truth": total_gt,
            "total_matched_samples": total_matched,
        }

    if args.output:
        output_path = Path(args.output)
    else:
        pred_file = Path(args.pred)
        output_path = pred_file.parent / f"{pred_file.stem}_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"\nEvaluation results saved to: {output_path}")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="按任务调用 src 下对应评估脚本；jsonl 模式沿用内置 KIE 评估。"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=("jsonl", "kie", "recognition", "grounding", "vqa", "doc_parsing", "parsing"),
        help="jsonl: UNIKIE JSONL；doc_parsing|parsing: 文档解析（auto 按子目录含 custom/doc/table/formula/molecular）；其余：转调 evaluate_<task>.py",
    )
    parser.add_argument(
        "--pred",
        type=str,
        default=None,
        help="[jsonl] 预测 JSONL；[doc_parsing|parsing] 可与 --dataset 联用：作 pred_dir（预测 txt 目录）",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="[jsonl] 只评某一数据集；[doc_parsing|parsing] 可填 answer 下子目录名以自动 gt_dir",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="[jsonl] 评估结果 JSON 输出路径",
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        default=None,
        help="[kie|recognition|vqa|doc_parsing] 预测目录（与 request_openai 输出 layout 一致）",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        default=None,
        help="[kie|recognition|vqa|doc_parsing] 标注目录（ocr_datasets/.../answer/<subset>/）",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=("auto", "en", "cn"),
        help="[recognition] 传给 evaluate_recognition.py",
    )
    parser.add_argument(
        "--recognition-group",
        type=str,
        default="auto",
        choices=("auto", "multi_scene", "multi_lan", "other"),
        help="[recognition] 传给 evaluate_recognition.py；auto 根据路径识别 multi_scene（alphanum_only）",
    )
    parser.add_argument(
        "--evaluator",
        type=str,
        default="auto",
        choices=("auto", "en", "cn", "case_sensitive"),
        help="[vqa] 传给 evaluate_vqa.py",
    )
    parser.add_argument(
        "--gt_grounding_root",
        type=str,
        default=None,
        help="[grounding] GT 根目录，含 text_grounding/answer、object_grounding/answer；默认 <repo>/ocr_datasets/grounding",
    )
    parser.add_argument(
        "--pred_grounding_root",
        type=str,
        default=None,
        help="[grounding] 预测根目录，与 GT 镜像；默认 <repo>/results/<sanitized --model>/grounding",
    )
    parser.add_argument(
        "--pred_folder",
        type=str,
        default=None,
        help="[grounding] 预测子目录名，如 pred_gemini-3_1-flash；默认 pred_<sanitized --model>",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="[grounding] 模型名（与 results 目录名一致）；用于默认 pred_grounding_root / pred_folder",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="数据集内部并行 worker 数（外部数据集循环仍串行）",
    )

    args = parser.parse_args()

    if args.task == "jsonl":
        if not args.pred:
            parser.error("--task jsonl 需要 --pred")
        return run_jsonl_kie_eval(args)

    if args.task == "kie":
        if not args.pred_dir or not args.gt_dir:
            parser.error("--task kie 需要 --pred_dir 与 --gt_dir")
        return run_evaluator_script(
            "evaluate_kie.py",
            ["--pred_dir", args.pred_dir, "--gt_dir", args.gt_dir, "--jobs", str(args.jobs)],
        )

    if args.task == "recognition":
        if not args.pred_dir or not args.gt_dir:
            parser.error("--task recognition 需要 --pred_dir 与 --gt_dir")
        return run_evaluator_script(
            "evaluate_recognition.py",
            [
                "--pred_dir",
                args.pred_dir,
                "--gt_dir",
                args.gt_dir,
                "--mode",
                args.mode,
                "--recognition-group",
                args.recognition_group,
                "--jobs",
                str(args.jobs),
            ],
        )

    if args.task == "vqa":
        if not args.pred_dir or not args.gt_dir:
            parser.error("--task vqa 需要 --pred_dir 与 --gt_dir")
        return run_evaluator_script(
            "evaluate_vqa.py",
            ["--pred_dir", args.pred_dir, "--gt_dir", args.gt_dir, "--evaluator", args.evaluator, "--jobs", str(args.jobs)],
        )

    if args.task in ("doc_parsing", "parsing"):
        pred_dir = args.pred_dir or args.pred
        gt_dir = args.gt_dir
        if not gt_dir and args.dataset:
            # 尝试在新的 parsing 目录下寻找对应的 dataset
            parsing_root = REPO_ROOT / "ocr_datasets" / "parsing"
            found_gt = None
            if parsing_root.exists():
                for task_dir in parsing_root.iterdir():
                    if task_dir.is_dir():
                        potential_gt = task_dir / "answer" / args.dataset
                        if potential_gt.exists():
                            found_gt = str(potential_gt)
                            break
            if found_gt:
                gt_dir = found_gt
            else:
                # 兼容旧路径
                gt_dir = str(REPO_ROOT / "ocr_datasets" / "doc_parsing" / "answer" / args.dataset)
        if not pred_dir or not gt_dir:
            parser.error(
                "--task doc_parsing|parsing 需要 pred_dir+gt_dir，或 --pred（作为预测目录）+ --dataset（子集名，自动 "
                "寻找 gt_dir）"
            )
        return run_evaluator_script(
            "evaluate_doc_parsing.py",
            ["--pred_dir", pred_dir, "--gt_dir", gt_dir, "--jobs", str(args.jobs)],
        )

    if args.task == "grounding":
        gt_root = args.gt_grounding_root
        pred_root = args.pred_grounding_root
        pred_folder = args.pred_folder
        if args.model:
            sm = sanitize_model_name(args.model)
            if gt_root is None:
                gt_root = str(REPO_ROOT / "ocr_datasets" / "grounding")
            if pred_root is None:
                pred_root = str(REPO_ROOT / "results" / sm / "grounding")
            if pred_folder is None:
                pred_folder = f"pred_{sm}"
        if not gt_root or not pred_root:
            parser.error(
                "--task grounding 需要 --gt_grounding_root 与 --pred_grounding_root，或提供 --model 使用默认路径"
            )
        if not pred_folder:
            parser.error("--task grounding 需要 --pred_folder，或提供 --model")
        return run_evaluator_script(
            "evaluate_grounding.py",
            [gt_root, pred_root, pred_folder, str(args.jobs)],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main() or 0)
