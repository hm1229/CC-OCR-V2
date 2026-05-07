import argparse
import ast
import json
import math
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed


def levenshtein_distance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1

    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def vqa_evaluation(predict, answers):
    score = 0
    if type(answers) == list:
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ")
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
        else:
            dist = levenshtein_distance(predict, answers)
            length = max(len(predict), len(answers))
            ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
            ANLS_value = 1 - ANLS_value

            if ANLS_value >= 0.5 and ANLS_value > score:
                score = ANLS_value

    return score


def cn_vqa_evaluation(predict, answers):
    score = 0
    if type(answers) == list:
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ").replace(" ", "")
            if isinstance(predict, (int, float)):
                predict = str(predict)
            predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
            if len(answer.split(",")) < 4:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.lower().strip().replace("\n", " ").replace(" ", "")
        predict = predict.lower().strip().replace("\n", " ").replace(" ", "")
        if len(answers.split(",")) < 4:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def vqa_evaluation_case_sensitive(predict, answers):
    score = 0
    if type(answers) == list:
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].strip().replace("\n", " ")
            predict = predict.strip().replace("\n", " ")
            if len(answer.split()) < 5:
                if answer in predict:
                    score = 1
            else:
                dist = levenshtein_distance(predict, answer)
                length = max(len(predict), len(answer))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    else:
        answers = answers.strip().replace("\n", " ")
        predict = predict.strip().replace("\n", " ")
        if len(answers.split()) < 5:
            if answers in predict:
                score = 1
            else:
                dist = levenshtein_distance(predict, answers)
                length = max(len(predict), len(answers))
                ANLS_value = 0.0 if length == 0 else float(dist) / float(length)
                ANLS_value = 1 - ANLS_value

                if ANLS_value >= 0.5 and ANLS_value > score:
                    score = ANLS_value

    return score


def extract_first_number(string):
    match = re.search(r"\d+", string)
    if match:
        return int(match.group())
    return None


def counting_evaluation(predict, answers, eval_method):
    score = 0

    if isinstance(predict, str):
        predict_processed = predict.lower().strip().replace("\n", " ")
    elif isinstance(predict, float) and math.isnan(predict):
        return 0
    else:
        predict_processed = int(predict)
    if type(answers) == list:
        temp_score = 0
        for j in range(len(answers)):
            if isinstance(answers[j], (int, float)):
                answers[j] = str(answers[j])
            answer = answers[j].lower().strip().replace("\n", " ")
            if eval_method == "exact match":
                pred_str = (
                    predict.lower().strip().replace("\n", " ")
                    if isinstance(predict, str)
                    else str(predict).lower().strip().replace("\n", " ")
                )
                if answer in pred_str:
                    score = 1
                else:
                    score = 0
            elif eval_method == "regression":
                predict_number = extract_first_number(str(predict_processed))
                if predict_number:

                    answer = int(answer)

                    if predict_number <= 0 or predict_number >= 2 * answer:
                        score = 0
                    else:
                        iou = 1 - abs(predict_number - answer) / answer
                        if iou > 0.5:
                            score = iou
                        else:
                            score = 0
                else:
                    score = 0
            if score > temp_score:
                temp_score = score
        score = temp_score

    else:
        answer = answers.lower().strip().replace("\n", " ")
        predict = predict.lower().strip().replace("\n", " ")
        if eval_method == "exact match":
            if answer in predict:
                score = 1
            else:
                score = 0
        elif eval_method == "regression":
            predict = extract_first_number(predict)
            if predict:
                answer = int(answer)
                if predict <= 0 or predict >= 2 * answer:
                    score = 0
                else:
                    iou = 1 - abs(predict - answer) / answer

                    if iou > 0.5:
                        score = iou
                    else:
                        score = 0
            else:
                score = 0
    return score


def math_expression_evaluation(predict, answers):
    score = 0
    if type(answers) == list:
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            predict = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        predict = predict.strip().replace("\n", " ").replace(" ", "")
        if answers in predict:
            score = 1
    return score


def remove_text_tags(latex_str):
    """
    Removes LaTeX \\text{...} tags while keeping their content.

    :param latex_str: A string containing LaTeX expressions
    :return: The processed string with \\text{...} tags removed
    """

    pattern = r"\\text\{([^{}]*)\}"

    processed_str = re.sub(pattern, r"\1", latex_str)

    return processed_str


def cn_math_expression_evaluation(predict, answers):
    score = 0

    assert len(answers) == 1
    answers = [remove_text_tags(answers[0])]
    predict = remove_text_tags(predict)

    if type(answers) == list:
        for j in range(len(answers)):
            answer = answers[j].strip().replace("\n", " ").replace(" ", "")
            predict = predict.strip().replace("\n", " ").replace(" ", "")
            if answer in predict:
                score = 1
    else:
        answers = answers.strip().replace("\n", " ").replace(" ", "")
        predict = predict.strip().replace("\n", " ").replace(" ", "")
        if answers in predict:
            score = 1
    return score


def strip_code_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def parse_gt_answers(raw: str):
    """返回 str 或 list，供 vqa_evaluation / cn_vqa_evaluation 使用。"""
    raw = raw.strip()
    if not raw:
        return None
    if raw.startswith("["):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                data = None
        if isinstance(data, list) and data:
            return data
    return raw


def pick_evaluator_mode(answers, force: str) -> str:
    """force: auto | en | cn | case_sensitive"""
    if force != "auto":
        return force
    if isinstance(answers, list):
        s = " ".join(str(x) for x in answers)
    else:
        s = str(answers)
    return "cn" if _has_cjk(s) else "en"


def score_vqa_sample(gt_path: Path, pred_path: Path, evaluator_mode: str):
    try:
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "bad_gt"
    answers = parse_gt_answers(gt_raw)
    if answers is None:
        return None, "bad_gt"

    try:
        pred_raw = pred_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None, "bad_pred"

    predict = strip_code_fence(pred_raw).strip()
    if not predict:
        return 0.0, "empty_pred"

    mode = pick_evaluator_mode(answers, evaluator_mode)
    if mode == "cn":
        sc = cn_vqa_evaluation(predict, answers)
    elif mode == "case_sensitive":
        sc = vqa_evaluation_case_sensitive(predict, answers)
    else:
        sc = vqa_evaluation(predict, answers)
    return float(sc), None


def collect_pairs(ccocr_vqa: Path, eval_vqa: Path, pred_folder_name: str):
    """
    遍历 ccocr_vqa/answer 下所有 .txt，在 eval_vqa 下找同名预测：
      pred_path = eval_vqa / {subset} / pred_folder_name / {name}.txt
    """
    gt_root = ccocr_vqa / "answer"
    pairs = []
    if not gt_root.is_dir():
        return pairs
    for gt_file in sorted(gt_root.rglob("*.txt")):
        rel = gt_file.relative_to(gt_root)
        pred_file = eval_vqa / rel.parent / pred_folder_name / gt_file.name
        if pred_file.is_file():
            pairs.append((gt_file, pred_file))
    return pairs


def summarize_vqa_scores(scores: list[float]):
    """mean_score ∈ [0,1]；acc_exact 为满分(1)；acc_half 为 ≥0.5（含 ANLS 部分分）。"""
    if not scores:
        return {
            "n": 0,
            "mean_score": 0.0,
            "acc_exact": 0.0,
            "acc_at_half": 0.0,
        }
    n = len(scores)
    mean = sum(scores) / n
    acc_exact = sum(1 for s in scores if s >= 1.0 - 1e-9) / n
    acc_half = sum(1 for s in scores if s >= 0.5) / n
    return {
        "n": n,
        "mean_score": mean,
        "acc_exact": acc_exact,
        "acc_at_half": acc_half,
    }


def run_eval(ccocr_vqa: Path, eval_vqa: Path, pred_folder: str, evaluator_mode: str):
    pairs = collect_pairs(ccocr_vqa, eval_vqa, pred_folder)
    scores_all = []
    by_subset: dict[str, list[float]] = {}
    skipped: dict[str, int] = {}

    for gt_path, pred_path in pairs:
        sc, err = score_vqa_sample(gt_path, pred_path, evaluator_mode)
        if err == "bad_gt":
            skipped["bad_gt"] = skipped.get("bad_gt", 0) + 1
            continue
        if sc is None:
            skipped[err or "unknown"] = skipped.get(err or "unknown", 0) + 1
            continue
        scores_all.append(sc)
        subset = gt_path.parent.name
        by_subset.setdefault(subset, []).append(sc)

    report = {
        "pred_folder": pred_folder,
        "evaluator_mode_default": evaluator_mode,
        "ccocr_vqa": str(ccocr_vqa),
        "eval_vqa": str(eval_vqa),
        "overall": summarize_vqa_scores(scores_all),
        "by_subset": {k: summarize_vqa_scores(v) for k, v in sorted(by_subset.items())},
        "skipped": skipped,
    }
    return report


def main():
    parser = argparse.ArgumentParser(description="Quickly evaluate VQA accuracy for directory-based predictions")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .txt files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth .txt files")
    parser.add_argument("--evaluator", choices=("auto", "en", "cn", "case_sensitive"), default="auto",
                        help="auto：参考答案含中文用 cn_vqa，否则 vqa_evaluation")
    parser.add_argument("--jobs", type=int, default=1, help="dataset internal parallel workers")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()

    if not pred_dir.is_dir() or not gt_dir.is_dir():
        print(f"Warning: pred_dir or gt_dir is not a directory.\n  pred_dir: {pred_dir}\n  gt_dir: {gt_dir}")
        if not gt_dir.exists():
            print("Error: gt_dir does not exist.")
            return

    pairs = []
    for gt_file in sorted(gt_dir.rglob("*.txt")):
        rel_path = gt_file.relative_to(gt_dir)
        pred_file = pred_dir / rel_path
        
        if not pred_file.is_file():
            matches = list(pred_dir.rglob(gt_file.name))
            if matches:
                pred_file = matches[0]
        
        if pred_file.is_file():
            pairs.append((gt_file, pred_file))

    if not pairs:
        print("Error: No matching prediction and ground truth files found.")
        return

    scores_all = []
    per_file = []
    skipped = {}

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for gt_path, pred_path in pairs:
            sc, err = score_vqa_sample(gt_path, pred_path, args.evaluator)
            if err == "bad_gt":
                skipped["bad_gt"] = skipped.get("bad_gt", 0) + 1
                continue
            if sc is None:
                skipped[err or "unknown"] = skipped.get(err or "unknown", 0) + 1
                continue
            scores_all.append(sc)
            per_file.append({"file": str(gt_path.relative_to(gt_dir)), "score": sc})
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(score_vqa_sample, gt_path, pred_path, args.evaluator): (gt_path, pred_path)
                for gt_path, pred_path in pairs
            }
            for fut in as_completed(futures):
                gt_path, pred_path = futures[fut]
                sc, err = fut.result()
                if err == "bad_gt":
                    skipped["bad_gt"] = skipped.get("bad_gt", 0) + 1
                    continue
                if sc is None:
                    skipped[err or "unknown"] = skipped.get(err or "unknown", 0) + 1
                    continue
                scores_all.append(sc)
                per_file.append({"file": str(gt_path.relative_to(gt_dir)), "score": sc})

    report = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "evaluator_mode": args.evaluator,
        "overall": summarize_vqa_scores(scores_all),
        "skipped": skipped,
        "per_file": per_file,
    }

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    print("\noverall mean_score:", report["overall"]["mean_score"])


if __name__ == "__main__":
    main()
