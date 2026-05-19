#!/usr/bin/env python3
import ast
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment


def iou_box_xyxy(a, b):
    ax1, ay1, ax2, ay2 = [float(x) for x in a]
    bx1, by1, bx2, by2 = [float(x) for x in b]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + bb - inter
    return inter / union if union > 0 else 0.0


def strip_code_fence(s):
    s = s.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return s


def parse_text_gt(path):
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    return ast.literal_eval(raw)


def parse_text_pred(s):
    s = strip_code_fence(s)
    m = re.search(
        r"[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)\s*[\)\]]",
        s,
    )
    if not m:
        return None
    return [float(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))]


def _to_float(v):
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.strip())
        except ValueError:
            return None
    return None


def normalize_bbox_xyxy(bb):
    """将 bbox 统一为 [x1, y1, x2, y2]；支持 list/tuple 与多种 dict 键名（Gemini 等）。"""
    if bb is None:
        return None
    if isinstance(bb, (list, tuple)):
        if len(bb) < 4:
            return None
        nums = [_to_float(bb[i]) for i in range(4)]
        if any(x is None for x in nums):
            return None
        return nums
    if not isinstance(bb, dict):
        return None

    def lower_map(d):
        out = {}
        for k, v in d.items():
            if k is None:
                continue
            out[str(k).lower().strip()] = v
        return out

    d = lower_map(bb)

    def g(*names):
        for n in names:
            if n in d:
                f = _to_float(d[n])
                if f is not None:
                    return f
        return None

    x1, y1, x2, y2 = g("x1"), g("y1"), g("x2"), g("y2")
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2, y2]

    x1, y1, x2, y2 = g("x1"), g("y"), g("x2"), g("y2")
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2, y2]

    x1, y1, x2, y2 = g("xmin"), g("ymin"), g("xmax"), g("ymax")
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2, y2]

    x1, y1, x2, y2 = g("left"), g("top"), g("right"), g("bottom")
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2, y2]

    x1, y1 = g("x0"), g("y0")
    x2, y2 = g("x1"), g("y1")
    if x1 is not None and y1 is not None and x2 is not None and y2 is not None:
        return [x1, y1, x2, y2]

    ox, oy = g("x"), g("y")
    w, h = g("width", "w"), g("height", "h")
    if ox is not None and oy is not None and w is not None and h is not None:
        return [ox, oy, ox + w, oy + h]

    return None


def parse_object_list(s):
    s = strip_code_fence(s)
    try:
        data = json.loads(s)
    except json.JSONDecodeError:
        try:
            data = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            return None
    if not isinstance(data, list):
        return None
    out = []
    for item in data:
        if not isinstance(item, dict):
            continue
        # Gemini 等用 category_name；标准答案用 label
        lab = item.get("label")
        if lab is None:
            lab = item.get("category_name")
        bb = item.get("bbox_2d") or item.get("bbox") or item.get("box_2d")
        if lab is None or not bb:
            continue
        xyxy = normalize_bbox_xyxy(bb)
        if xyxy is None:
            continue
        out.append({"label": str(lab).strip(), "bbox": xyxy})
    return out


def get_image_size(gt_path, ccocr_grounding, task):
    """根据 gt 路径找到对应图像，返回 (width, height)"""
    # gt_path: .../answer/{subset}/{stem}.txt
    subset = gt_path.parent.name
    stem = gt_path.stem
    img_dir = ccocr_grounding / task / "images" / subset
    for ext in (".jpg", ".jpeg", ".png", ".webp", ".JPG", ".JPEG", ".PNG"):
        p = img_dir / (stem + ext)
        if p.is_file():
            with Image.open(p) as im:
                return im.size  # (width, height)
    return None


def detect_coord_scale(boxes):
    """检测坐标范围：0-1 归一化 vs 0-1000 归一化。返回 scale factor。"""
    if not boxes:
        return 1000
    max_val = max(v for box in boxes for v in box[:4])
    if max_val <= 1.0:
        return 1
    return 1000


def pred_to_pixel(bbox, w, h, scale=1000):
    """将 pred 的归一化坐标转为像素坐标"""
    return [
        bbox[0] * w / scale,
        bbox[1] * h / scale,
        bbox[2] * w / scale,
        bbox[3] * h / scale,
    ]


def score_text_sample(gt_path, pred_path, ccocr_grounding, task):
    try:
        gt = parse_text_gt(gt_path)
    except (ValueError, SyntaxError):
        return None, "bad_gt"
    if not isinstance(gt, (list, tuple)) or len(gt) < 4:
        return None, "bad_gt"
    raw = pred_path.read_text(encoding="utf-8", errors="replace")
    pred = parse_text_pred(raw)
    if pred is None:
        return 0.0, "no_pred_box"
    size = get_image_size(gt_path, ccocr_grounding, task)
    if size:
        scale = detect_coord_scale([pred])
        pred = pred_to_pixel(pred, size[0], size[1], scale)
    return iou_box_xyxy(pred, gt[:4]), None


def score_object_sample(gt_path, pred_path, ccocr_grounding, task):
    try:
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
        gt_list = parse_object_list(gt_raw)
    except (OSError, UnicodeDecodeError):
        return None, "bad_gt"
    if not gt_list:
        return None, "bad_gt"
    pred_list = parse_object_list(pred_path.read_text(encoding="utf-8", errors="replace"))
    if pred_list is None:
        return 0.0, "bad_pred_json"
    if len(pred_list) == 0:
        return 0.0, None
    size = get_image_size(gt_path, ccocr_grounding, task)
    raw_pred_boxes = [p["bbox"] for p in pred_list]
    scale = detect_coord_scale(raw_pred_boxes)
    pred_boxes = []
    for bbox in raw_pred_boxes:
        if size:
            bbox = pred_to_pixel(bbox, size[0], size[1], scale)
        pred_boxes.append(bbox)
    gt_boxes = [g["bbox"] for g in gt_list]
    n_gt = len(gt_boxes)
    n_pred = len(pred_boxes)
    # 构建 IoU 代价矩阵，用匈牙利算法做最优一对一匹配
    iou_matrix = np.zeros((n_gt, n_pred))
    for i, gb in enumerate(gt_boxes):
        for j, pb in enumerate(pred_boxes):
            iou_matrix[i, j] = iou_box_xyxy(gb, pb)
    # 匈牙利算法求最大匹配（转为最小化问题）
    row_ind, col_ind = linear_sum_assignment(-iou_matrix)
    matched_ious = [iou_matrix[r, c] for r, c in zip(row_ind, col_ind)]
    # 未匹配的 GT 框 IoU 记为 0
    total_iou = sum(matched_ious)
    return total_iou / n_gt, None


def collect_pairs(ccocr_grounding, eval_grounding, task, pred_folder_name):
    gt_root = ccocr_grounding / task / "answer"
    pred_root = eval_grounding / task
    pred_sub = pred_folder_name
    pairs = []
    if not gt_root.is_dir():
        return pairs
    for gt_file in sorted(gt_root.rglob("*.txt")):
        rel = gt_file.relative_to(gt_root)
        pred_file = pred_root / rel.parent / pred_sub / gt_file.name
        if not pred_file.is_file():
            continue
        pairs.append((gt_file, pred_file))
    return pairs


def summarize(scores_by_subset, thresh=0.5):
    if not scores_by_subset:
        return {"n": 0, "mean_iou": 0.0, "acc_at_05": 0.0, "subset_means": {}}
    
    subset_means = {}
    subset_accs = {}
    total_n = 0
    
    for subset, scores in scores_by_subset.items():
        n = len(scores)
        if n == 0:
            continue
        total_n += n
        subset_means[subset] = sum(scores) / n
        subset_accs[subset] = sum(1 for s in scores if s >= thresh) / n
        
    if not subset_means:
        return {"n": 0, "mean_iou": 0.0, "acc_at_05": 0.0, "subset_means": {}}
        
    # Macro-average over subsets
    macro_mean_iou = sum(subset_means.values()) / len(subset_means)
    macro_mean_acc = sum(subset_accs.values()) / len(subset_accs)
    
    return {
        "n": total_n, 
        "mean_iou": macro_mean_iou, 
        "acc_at_05": macro_mean_acc,
        "subset_means": subset_means
    }


def main():
    jobs = 1
    if len(sys.argv) >= 3:
        ccocr_g = Path(sys.argv[1]).resolve()
        eval_g = Path(sys.argv[2]).resolve()
        pred_folder = sys.argv[3] if len(sys.argv) > 3 else "pred_qwen-vl-max"
        jobs = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    else:
        raise SystemExit("Usage: evaluate_grounding.py <gt_root> <pred_root> [pred_folder] [jobs]")

    report = {"pred_folder": pred_folder, "ccocr_grounding": str(ccocr_g), "eval_grounding": str(eval_g)}

    for task, scorer in (
        ("text_grounding", "text"),
        ("object_grounding", "object"),
    ):
        pairs = collect_pairs(ccocr_g, eval_g, task, pred_folder)
        scores_by_subset = {}
        per_file = []
        skipped = {}
        jobs = max(1, jobs)
        if jobs == 1:
            for gt_path, pred_path in pairs:
                subset_name = gt_path.parent.name
                if scorer == "text":
                    sc, err = score_text_sample(gt_path, pred_path, ccocr_g, task)
                else:
                    sc, err = score_object_sample(gt_path, pred_path, ccocr_g, task)
                if err == "bad_gt":
                    skipped["bad_gt"] = skipped.get("bad_gt", 0) + 1
                    continue
                if sc is None:
                    skipped[err or "unknown"] = skipped.get(err or "unknown", 0) + 1
                    continue
                scores_by_subset.setdefault(subset_name, []).append(sc)
                per_file.append({"file": gt_path.name, "subset": subset_name, "iou": sc})
        else:
            with ThreadPoolExecutor(max_workers=jobs) as ex:
                if scorer == "text":
                    futures = {
                        ex.submit(score_text_sample, gt_path, pred_path, ccocr_g, task): (gt_path, pred_path)
                        for gt_path, pred_path in pairs
                    }
                else:
                    futures = {
                        ex.submit(score_object_sample, gt_path, pred_path, ccocr_g, task): (gt_path, pred_path)
                        for gt_path, pred_path in pairs
                    }
                for fut in as_completed(futures):
                    gt_path, pred_path = futures[fut]
                    subset_name = gt_path.parent.name
                    sc, err = fut.result()
                    if err == "bad_gt":
                        skipped["bad_gt"] = skipped.get("bad_gt", 0) + 1
                        continue
                    if sc is None:
                        skipped[err or "unknown"] = skipped.get(err or "unknown", 0) + 1
                        continue
                    scores_by_subset.setdefault(subset_name, []).append(sc)
                    per_file.append({"file": gt_path.name, "subset": subset_name, "iou": sc})
        report[task] = summarize(scores_by_subset)
        report[task]["skipped"] = skipped
        report[task]["per_file"] = per_file

    # 总分：直接对所有三级子数据集的 mean_iou 求平均（Macro-average over level 3 datasets）
    all_subset_means = []
    for task in ("text_grounding", "object_grounding"):
        if "subset_means" in report[task]:
            for subset, mean_val in report[task]["subset_means"].items():
                all_subset_means.append(mean_val)
            
    if not all_subset_means:
        report["overall_mean_iou"] = 0.0
    else:
        report["overall_mean_iou"] = sum(all_subset_means) / len(all_subset_means)


    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("\noverall_mean_iou:", report["overall_mean_iou"])


if __name__ == "__main__":
    main()
