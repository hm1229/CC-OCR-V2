import os
import json
import re
import argparse
from abc import abstractmethod
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed


def pick_response_text(json_path):
    """
    """
    try:
        with open(json_path, "r") as f:
            json_data = json.load(f)
    except Exception as e:
        print("--> file error: msg: {}, path: {}".format(e, json_path))
        return None

    for required_key in ["model_name", "response"]:
        if required_key not in json_data:
            print("--> required key not exists, name: {}, path: {}".format(required_key, json_path))
            return None

    model_name = json_data["model_name"]
    model_response = json_data["response"]

    response_text = None
    if model_name.startswith("gpt") or model_name.startswith("o1"):
        response_text = model_response.get("data", {}).get("response", {}).get("choices", [{}])[0].get("message", {}).get("content", None)
    elif model_name.startswith("local_"):
        response_text = model_response
    else:
        if model_name.startswith("claude"):
            content_list = model_response.get("content", None)
        elif model_name.startswith("gemini"):
            content_list = model_response.get("candidates", [{}])[0].get("content", {}).get("parts", None)
        elif model_name.startswith("qwen"):
            content_list = model_response.get("output", {}).get("choices", [{}])[0].get("message", {}).get("content", None)
        else:
            raise NotImplementedError("The pick_response_text NOT implemented for model: {}".format(model_name))

        if isinstance(content_list, list) and len(content_list) > 0:
            response_text = content_list[0].get("text", None)

    if response_text is None:
        print("--> [error][{}] text pick error, path: {}".format(model_name, json_path))
    return response_text


def load_response_from_dir(res_dir):
    """
    """
    response_info = {}
    for file_name in os.listdir(res_dir):
        file_path = os.path.abspath(os.path.join(res_dir, file_name))
        if not file_name.endswith(".json"):
            print("--> skip: result file should be a json: but got: {}".format(file_path))
            continue

        response_text = pick_response_text(file_path)
        if response_text is None:
            continue

        file_name_wo_ext, ext = os.path.splitext(file_name)
        response_info[file_name_wo_ext] = response_text
    return response_info


class BaseMetric(object):
    """ BaseMetric """
    def __init__(self, group_name, **kwargs):
        self.group_name = group_name
        self.kwargs = kwargs

    def response_post_func(self, response_text, **kwargs):
        return response_text

    @abstractmethod
    def evaluate(self, response_info, gt_info, normalize_func=None, **kwargs):
        pass

    def __call__(self, pdt_res_dir, gt_info, with_response_ratio=True, **kwargs):
        if isinstance(pdt_res_dir, dict):
            raw_response_info = pdt_res_dir
        elif os.path.exists(pdt_res_dir) and os.path.isdir(pdt_res_dir):
            raw_response_info = load_response_from_dir(pdt_res_dir)
        else:
            return ValueError("invalid input: response dict or folder are required, but got {}".format(pdt_res_dir))

        post_error_list, response_info = [], {}
        response_error_list = list(gt_info.keys() - raw_response_info.keys())
        for file_name, single_pdt_str in raw_response_info.items():
            single_pdt_str = self.response_post_func(single_pdt_str, **kwargs)
            if single_pdt_str is None:
                post_error_list.append(file_name)
                continue
            response_info[file_name] = single_pdt_str

        meta_info = {
            "gt_total_num": len(gt_info), "pdt_total_num": len(response_info),
            "post_error_list": post_error_list, "response_error_list": response_error_list,
        }
        eval_info = self.evaluate(response_info, gt_info, **kwargs)

        # add response_success_ratio
        if "summary" in eval_info and with_response_ratio:
            success_ratio = (len(response_info) + len(post_error_list)) / (len(gt_info) + 1e-9)
            eval_info["summary"].update({"response_success_ratio": success_ratio })
        return meta_info, eval_info


def convert_to_halfwidth(text):
    halfwidth_chars = str.maketrans(
        '！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～',
        "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"
    )

    return text.translate(halfwidth_chars)



def token_normalize(token_text, is_lower=False, is_alphanum_only=False):
    if is_lower:
        token_text = token_text.lower()
    if is_alphanum_only:
        token_text = re.sub('[^A-Za-z0-9]+', '', token_text)
    return token_text


def text_normalize_and_tokenize(text, is_keep_blank=True, is_lower=True, is_alphanum_only=False):
    text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
    text = re.sub(r'\s+', ' ', text)
    if not is_keep_blank:
        text = text.replace(" ", "")
    text_tokens = text.split(" ") if is_keep_blank else list(text)
    text_token_normalized = [token_normalize(t, is_lower, is_alphanum_only) for t in text_tokens]
    text_token_normalized = [x for x in text_token_normalized if len(x) > 0]
    return text_token_normalized


def evaluate_single_sample(gts, preds):
    right_num = 0
    gt_counter_info = dict(Counter(gts))
    pdt_counter_info = dict(Counter(preds))
    for gt_token, gt_count in gt_counter_info.items():
        pred_count = pdt_counter_info.get(gt_token, 0)
        right_num += min(gt_count, pred_count)
    return right_num


def calculate_metrics(response_info, gt_info, is_verbose=False):
    macro_recall_list, macro_precision_list, macro_f1_list = [], [], []
    per_file = []
    total_gt_num, total_pred_num, total_right_num = 0, 0, 0
    for file_name, fullbox_gts in gt_info.items():
        fullbox_preds = response_info.get(file_name, [])
        right_num = evaluate_single_sample(fullbox_gts, fullbox_preds)
        total_right_num += right_num
        total_gt_num += len(fullbox_gts)
        total_pred_num += len(fullbox_preds)

        macro_recall = right_num / (len(fullbox_gts) + 1e-9)
        macro_precision = right_num / (len(fullbox_preds) + 1e-9)
        macro_f1 = 2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
        macro_recall_list.append(macro_recall)
        macro_precision_list.append(macro_precision)
        macro_f1_list.append(macro_f1)
        per_file.append({"file": file_name, "macro_f1": macro_f1})

    final_macro_recall = sum(macro_recall_list) / (len(macro_recall_list) + 1e-9)
    final_macro_precision = sum(macro_precision_list) / (len(macro_precision_list) + 1e-9)
    final_macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)

    recall_acc = total_right_num / (total_gt_num + 1e-9)
    preci_acc = total_right_num / (total_pred_num + 1e-9)
    hmean = 2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)
    vbs_eval_result = {
        'macro_recall': final_macro_recall, 'macro_precision': final_macro_precision, 'macro_f1_score': final_macro_f1,
        'micro_recall': recall_acc, 'micro_precision': preci_acc, 'mirco_f1_score': hmean,
        'per_file': per_file
    }
    eval_result = vbs_eval_result if is_verbose else {'macro_f1_score': final_macro_f1, 'mirco_f1_score': hmean, 'per_file': per_file}
    return eval_result


class OcrEvaluator(BaseMetric):
    def response_post_func(self, response_text, **kwargs):
        return response_text

    def evaluate(self, response_info, gt_info, **kwargs):
        # hard code here
        dataset_name = kwargs['dataset']
        is_word_level, is_lower, is_alphanum_only = True, True, False
        if dataset_name in ["Arabic", "Japanese", "Korean"] or "zh" in dataset_name:
            is_word_level = False
        if "multi_scene_ocr" in self.group_name and is_word_level:
            is_alphanum_only = True
        eval_config = {"word_level": is_word_level, "alphanum_only": is_alphanum_only, "lowercase": is_lower}

        image_pdt_info, image_gt_info = {}, {}
        for file_name, gt_src in gt_info.items():
            pred_src = response_info.get(file_name, "")
            pdt_token_list = text_normalize_and_tokenize(str(pred_src).strip(), is_word_level, is_lower, is_alphanum_only)
            gt_token_list = text_normalize_and_tokenize(str(gt_src).strip(), is_word_level, is_lower, is_alphanum_only)
            image_pdt_info[file_name] = pdt_token_list
            image_gt_info[file_name] = gt_token_list
        eval_result = calculate_metrics(image_pdt_info, image_gt_info, is_verbose=False)
        return {"summary": eval_result, "metric_config": eval_config}


def strip_code_fence(s: str) -> str:
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def _has_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


def _path_indicates_multi_scene(path: Path) -> bool:
    """
    True when GT path lives under CC-OCR++ multi_scene layout, e.g.
    .../recognition/multi_scene_recognition/answer/multi_scene_ocr_document_text_CORD_100/0_....txt
    Mirrors CC-OCR OcrEvaluator: group multi_scene_ocr + word-level -> is_alphanum_only=True.
    """
    try:
        parts = path.resolve().parts
    except OSError:
        parts = path.parts
    for part in parts:
        if part == "multi_scene_recognition" or part.startswith("multi_scene_ocr"):
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Quickly evaluate Recognition accuracy for directory-based predictions")
    parser.add_argument("--pred_dir", type=str, required=True,
                        help="Directory containing prediction .txt files")
    parser.add_argument("--gt_dir", type=str, required=True,
                        help="Directory containing ground truth .txt files")
    parser.add_argument("--mode", choices=("auto", "en", "cn"), default="auto",
                        help="auto: detect CJK for word-level vs char-level")
    parser.add_argument(
        "--recognition-group",
        choices=("auto", "multi_scene", "multi_lan", "other"),
        default="auto",
        help="auto: infer from path (multi_scene_recognition / multi_scene_ocr_* -> multi_scene). "
        "multi_scene: force CC-OCR alphanum-only when word-level. multi_lan/other: never alphanum-only.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="dataset internal parallel workers")

    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()

    if not gt_dir.exists():
        print(f"Error: gt_dir does not exist: {gt_dir}")
        return
    if not pred_dir.exists():
        print(f"Warning: pred_dir does not exist: {pred_dir}")

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

    image_pdt_info, image_gt_info = {}, {}

    def use_multi_scene_rules_for_path(gt_path: Path) -> bool:
        if args.recognition_group == "multi_scene":
            return True
        if args.recognition_group in ("multi_lan", "other"):
            return False
        return _path_indicates_multi_scene(gt_path)

    multi_scene_samples = sum(1 for g, _ in pairs if use_multi_scene_rules_for_path(g))
    if multi_scene_samples and args.recognition_group == "auto":
        print(
            f"--> recognition: {multi_scene_samples}/{len(pairs)} GT paths under multi_scene layout "
            f"(alphanum_only=True when word-level, same as CC-OCR multi_scene_ocr)"
        )

    def process_pair(gt_path: Path, pred_path: Path):
        gt_text = gt_path.read_text(encoding="utf-8", errors="replace").strip()
        pred_text = strip_code_fence(pred_path.read_text(encoding="utf-8", errors="replace")).strip()

        is_word_level = True
        if args.mode == "cn" or (args.mode == "auto" and _has_cjk(gt_text)):
            is_word_level = False

        is_lower = True
        is_alphanum_only = bool(
            use_multi_scene_rules_for_path(gt_path) and is_word_level
        )
        pdt_token_list = text_normalize_and_tokenize(pred_text, is_word_level, is_lower, is_alphanum_only)
        gt_token_list = text_normalize_and_tokenize(gt_text, is_word_level, is_lower, is_alphanum_only)
        file_key = str(gt_path.relative_to(gt_dir))
        return file_key, pdt_token_list, gt_token_list

    jobs = max(1, int(args.jobs))
    if jobs == 1:
        for gt_path, pred_path in pairs:
            try:
                key, pdt_tokens, gt_tokens = process_pair(gt_path, pred_path)
                image_pdt_info[key] = pdt_tokens
                image_gt_info[key] = gt_tokens
            except Exception as e:
                print(f"Error processing {gt_path.name}: {e}")
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {ex.submit(process_pair, gt_path, pred_path): gt_path for gt_path, pred_path in pairs}
            for fut in as_completed(futures):
                gt_path = futures[fut]
                try:
                    key, pdt_tokens, gt_tokens = fut.result()
                    image_pdt_info[key] = pdt_tokens
                    image_gt_info[key] = gt_tokens
                except Exception as e:
                    print(f"Error processing {gt_path.name}: {e}")

    eval_result = calculate_metrics(image_pdt_info, image_gt_info, is_verbose=True)
    
    report = {
        "pred_dir": str(pred_dir),
        "gt_dir": str(gt_dir),
        "num_samples": len(pairs),
        "recognition_group": args.recognition_group,
        "multi_scene_path_samples": multi_scene_samples,
        "results": eval_result,
    }
    
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
