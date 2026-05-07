"""
文档解析评估：根据数据集前缀自动选择评估策略。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import nltk
from tqdm import tqdm

try:
    from evaluate_recognition import (
        calculate_metrics,
        convert_to_halfwidth,
        text_normalize_and_tokenize,
    )
except ImportError:
    def convert_to_halfwidth(text: str) -> str:
        halfwidth_chars = str.maketrans(
            "！＂＃＄％＆＇（）＊＋，－．／０１２３４５６７８９：；＜＝＞？＠ＡＢＣＤＥＦＧＨＩＪＫＬＭＮＯＰＱＲＳＴＵＶＷＸＹＺ［＼］＾＿｀ａｂｃｄｅｆｇｈｉｊｋｌｍｎｏｐｑｒｓｔｕｖｗｘｙｚ｛｜｝～",
            "!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~",
        )
        return text.translate(halfwidth_chars)

    def text_normalize_and_tokenize(
        text, is_keep_blank=True, is_lower=True, is_alphanum_only=False
    ):
        text = text.replace("\t", " ").replace("\n", " ").replace("###", "").replace("***", "")
        text = re.sub(r"\s+", " ", text)
        if not is_keep_blank:
            text = text.replace(" ", "")
        text_tokens = text.split(" ") if is_keep_blank else list(text)
        return [t for t in text_tokens if t]

    def calculate_metrics(response_info, gt_info, is_verbose=False):
        from collections import Counter

        macro_f1_list = []
        total_gt_num, total_pred_num, total_right_num = 0, 0, 0
        for file_name, fullbox_gts in gt_info.items():
            fullbox_preds = response_info.get(file_name, [])
            gt_counter = dict(Counter(fullbox_gts))
            pdt_counter = dict(Counter(fullbox_preds))
            right_num = 0
            for gt_token, gt_count in gt_counter.items():
                right_num += min(gt_count, pdt_counter.get(gt_token, 0))
            total_right_num += right_num
            total_gt_num += len(fullbox_gts)
            total_pred_num += len(fullbox_preds)
            macro_recall = right_num / (len(fullbox_gts) + 1e-9)
            macro_precision = right_num / (len(fullbox_preds) + 1e-9)
            macro_f1 = (
                2 * macro_recall * macro_precision / (macro_recall + macro_precision + 1e-9)
            )
            macro_f1_list.append(macro_f1)
        final_macro_f1 = sum(macro_f1_list) / (len(macro_f1_list) + 1e-9)
        recall_acc = total_right_num / (total_gt_num + 1e-9)
        preci_acc = total_right_num / (total_pred_num + 1e-9)
        hmean = 2 * recall_acc * preci_acc / (recall_acc + preci_acc + 1e-9)
        return {"macro_f1_score": final_macro_f1, "mirco_f1_score": hmean}


try:
    from collections import deque
    from apted.helpers import Tree
    from apted import APTED, Config

    class TableTree(Tree):
        """
        # Copyright 2020 IBM
        # Author: peter.zhong@au1.ibm.com
        # License:  Apache 2.0 License.
        """
        def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
            self.tag = tag
            self.colspan = colspan
            self.rowspan = rowspan
            self.content = content
            self.children = list(children)

        def bracket(self):
            """Show tree using brackets notation"""
            if self.tag == "td":
                result = '"tag": %s, "colspan": %d, "rowspan": %d, "text": %s' % (
                    self.tag,
                    self.colspan,
                    self.rowspan,
                    self.content,
                )
            else:
                result = '"tag": %s' % self.tag
            for child in self.children:
                result += child.bracket()
            return "{{{}}}".format(result)

    class CustomConfig(Config):
        """
        # Copyright 2020 IBM
        # Author: peter.zhong@au1.ibm.com
        # License:  Apache 2.0 License.
        """
        def rename(self, node1, node2):
            """Compares attributes of trees"""
            if (
                (node1.tag != node2.tag)
                or (node1.colspan != node2.colspan)
                or (node1.rowspan != node2.rowspan)
            ):
                return 1.0
            if node1.tag == "td":
                if node1.content or node2.content:
                    return nltk.edit_distance(node1.content, node2.content) / max(len(node1.content), len(node2.content))
            return 0.0

    class TEDS(object):
        """Tree Edit Distance basead Similarity
        # Copyright 2020 IBM
        # Author: peter.zhong@au1.ibm.com
        # License:  Apache 2.0 License.
        """
        def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
            assert isinstance(n_jobs, int) and (
                n_jobs >= 1
            ), "n_jobs must be an integer greather than 1"
            self.structure_only = structure_only
            self.n_jobs = n_jobs
            self.ignore_nodes = ignore_nodes
            self.__tokens__ = []

        def tokenize(self, node):
            """Tokenizes table cells"""
            self.__tokens__.append("<%s>" % node.tag)
            if node.text is not None:
                self.__tokens__ += list(node.text)
            for n in node.getchildren():
                self.tokenize(n)
            if node.tag != "unk":
                self.__tokens__.append("</%s>" % node.tag)
            if node.tag != "td" and node.tail is not None:
                self.__tokens__ += list(node.tail)

        def load_html_tree(self, node, parent=None):
            """Converts HTML tree to the format required by apted"""
            global __tokens__
            if node.tag == "td":
                if self.structure_only:
                    cell = []
                else:
                    self.__tokens__ = []
                    self.tokenize(node)
                    cell = self.__tokens__[1:-1].copy()
                new_node = TableTree(
                    node.tag,
                    int(node.attrib.get("colspan", "1")),
                    int(node.attrib.get("rowspan", "1")),
                    cell,
                    *deque(),
                )
            else:
                new_node = TableTree(node.tag, None, None, None, *deque())
            if parent is not None:
                parent.children.append(new_node)
            if node.tag != "td":
                for n in node.getchildren():
                    self.load_html_tree(n, new_node)
            if parent is None:
                return new_node

        def evaluate(self, pred, true):
            """Computes TEDS score between the prediction and the ground truth of a
            given sample
            """
            from lxml import etree, html
            if (not pred) or (not true):
                return 0.0

            parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
            pred = html.fromstring(pred, parser=parser)
            true = html.fromstring(true, parser=parser)
            if pred.xpath("body/table") and true.xpath("body/table"):
                pred = pred.xpath("body/table")[0]
                true = true.xpath("body/table")[0]
                if self.ignore_nodes:
                    etree.strip_tags(pred, *self.ignore_nodes)
                    etree.strip_tags(true, *self.ignore_nodes)
                n_nodes_pred = len(pred.xpath(".//*"))
                n_nodes_true = len(true.xpath(".//*"))
                n_nodes = max(n_nodes_pred, n_nodes_true)
                tree_pred = self.load_html_tree(pred)
                tree_true = self.load_html_tree(true)
                distance = APTED(
                    tree_pred, tree_true, CustomConfig()
                ).compute_edit_distance()
                return 1.0 - (float(distance) / n_nodes)
            else:
                return 0.0

except ImportError as e:
    print(f"Warning: Cannot import TEDS dependencies: {e}")
    TEDS = None


# 移除指定的LaTeX命令
patterns = [
    r'\\documentclass\{.*?\}',
    r'\\usepackage\[.*?\]\{.*?\}',
    r'\\usepackage\{.*?\}',
    r'\\geometry\{.*?\}',
    r'\\begin\{document\}',
    r'\\end\{document\}',
    r'\\noindent'
]


def extract_and_clean_tables(text):
    if '</table>' not in text:
        text += '</table>'

    # Use regular expressions to find all table parts
    tables = re.findall(r'<table.*?>.*?</table>', text, re.DOTALL)

    clean_tables = []
    for table in tables:
        # Remove extra information from the table header, keeping only <table>...</table>.
        table_content = re.sub(r'<table.*?>', '<table>', table)

        # Remove line breaks and excessive spaces between tags without affecting the information inside the tags, such as attributes.
        table_content = re.sub(r'>\s+<', '><', table_content)

        # Eliminate line breaks and redundant spaces within tags (i.e., between '>' and '<').
        table_content = re.sub(r'>(.*?)<', lambda m: '>' + m.group(1).replace('\n', '').replace(' ', '') + '<', table_content, flags=re.DOTALL)

        # Flatten the table content by removing all line breaks.
        table_content = table_content.replace('\n', '').strip()
        clean_tables.append(table_content)

    flat_table = ''.join(clean_tables)
    return flat_table


# 去掉模型在 ```latex / ```html 内外的说明性文字（Note、### Notes 等），不参与打分
_LATEX_META_TAIL = re.compile(
    r"(?is)\n\s*(?:"
    r"#{1,6}\s*(?:📝\s*)?(?:Notes(?:\s+on\s+Transcription)?\s*:?[^\n]*)"
    r"|📌\s*Notes[^\n]*"
    r")(?:\n.*)*\Z"
)
_LATEX_LEADING_FLUFF = re.compile(
    r"(?is)^(?:"
    r"Here (?:is|’s|'s|are) (?:the )?(?:LaTeX|latex) [^\n]+(?:\n|$)"
    r"|Note that [^\n]+(?:\n|$)"
    r")+"
)

_HTML_META_TAIL = re.compile(
    r"(?is)\n\s*(?:"
    r"#{1,6}\s*(?:📝\s*)?Notes[^\n]*"
    r"|📌\s*Notes[^\n]*"
    r"|✅\s*\*{0,2}Note\*{0,2}\s*:"
    r"|>{0,1}\s*\*{0,2}\s*Note\*{0,2}\s*:"
    r")(?:\n.*)*\Z"
)


def strip_model_chatter_latex_fragment(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = _LATEX_META_TAIL.sub("", s)
    m_doc = re.search(r"(?m)^\\documentclass\b", s)
    m_begin = re.search(r"(?m)^\\begin\s*\{", s)
    candidates = [m for m in (m_doc, m_begin) if m is not None]
    if candidates:
        first = min(candidates, key=lambda x: x.start())
        if first.start() > 0:
            s = s[first.start() :]
    else:
        s = _LATEX_LEADING_FLUFF.sub("", s)
    return s.strip()


def strip_model_chatter_html_fragment(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return s
    s = _HTML_META_TAIL.sub("", s)
    m = re.search(
        r"(?i)<\s*(?:!DOCTYPE\b|html\b|body\b|table\b|div\b|h[1-6]\b|p\b|main\b|section\b)",
        s,
    )
    if m is not None and m.start() > 0:
        s = s[m.start() :]
    return s.lstrip()


def evaluate_single_doc_ccocr(gt, pred):
    """单样本 LaTeX 编辑相似度。"""
    for pattern in patterns:
        pred = re.sub(pattern, '', pred)

    try:
        pattern = r'```latex(.+?)```'
        pred = re.search(pattern, pred, re.DOTALL).group(1)
    except:
        if '```latex' in pred:
            pred = pred.split('```latex')[1]
            if '```' in pred:
                pred = pred.split('```', 1)[0]

    pred = strip_model_chatter_latex_fragment(pred)

    pred = pred.replace(' ', '').replace('\n', '')
    gt = gt.replace(' ', '').replace('\n', '')

    edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return 1 - edit_dist


def evaluate_single_table_ccocr(gt_raw, pred_raw):
    """单样本表格 TEDS。"""
    if TEDS is None:
        return 0.0
    teds = TEDS(structure_only=False, n_jobs=1)
    pred = _unwrap_fenced_html(pred_raw)

    pred = extract_and_clean_tables(pred)
    pred = convert_to_halfwidth(pred)
    gt = extract_and_clean_tables(gt_raw)
    gt = convert_to_halfwidth(gt)

    pred_html = '<html><body>{}</body></html>'.format(pred)
    gt_html = '<html><body>{}</body></html>'.format(gt)
    return teds.evaluate(pred_html, gt_html)


# ----- custom：非表 = 全部标签内文本（不区分 h2/p），去掉 / ／ | ｜；表 = TEDS（多表按索引配对平均） -----

_TABLE_BLOCK_RE = re.compile(r"<table\b[^>]*>.*?</table>", re.IGNORECASE | re.DOTALL)
_CUSTOM_SEP_RE = re.compile(r"[/／|｜]+")


def _strip_table_blocks(s: str) -> str:
    return _TABLE_BLOCK_RE.sub("", s)


def _unwrap_fenced_html(pred_raw: str) -> str:
    """取出 ```html ... ``` 内层并去掉模型 Note / ### Notes 等说明段。"""
    s = pred_raw.strip()
    try:
        m = re.search(r"```html(.+?)```", s, re.DOTALL)
        if m:
            return strip_model_chatter_html_fragment(m.group(1).strip())
    except Exception:
        pass
    if "```html" in s:
        rest = s.split("```html", 1)[1]
        if "```" in rest:
            rest = rest.split("```", 1)[0]
        return strip_model_chatter_html_fragment(rest.strip())
    return strip_model_chatter_html_fragment(s)


def _html_to_plain_text(html_snippet: str) -> str:
    if not (html_snippet and html_snippet.strip()):
        return ""
    try:
        from lxml import html as lxml_html

        parser = lxml_html.HTMLParser(encoding="utf-8", remove_comments=True)
        root = lxml_html.fromstring(f"<div>{html_snippet}</div>", parser=parser)
        return (root.text_content() or "").strip()
    except Exception:
        t = re.sub(r"<script[^>]*>.*?</script>", "", html_snippet, flags=re.I | re.DOTALL)
        t = re.sub(r"<[^>]+>", " ", t)
        return t.strip()


def normalize_custom_text_for_match(s: str) -> str:
    s = convert_to_halfwidth(s)
    s = _CUSTOM_SEP_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def custom_text_tokens_from_mixed(mixed: str) -> list[str]:
    plain = _html_to_plain_text(_strip_table_blocks(mixed))
    plain = normalize_custom_text_for_match(plain)
    return text_normalize_and_tokenize(
        plain,
        is_keep_blank=False,
        is_lower=False,
        is_alphanum_only=False,
    )


def evaluate_custom_text_score(gt_raw: str, pred_raw: str) -> float:
    pred_inner = _unwrap_fenced_html(pred_raw)
    gt_tok = custom_text_tokens_from_mixed(gt_raw)
    pred_tok = custom_text_tokens_from_mixed(pred_inner)
    if not gt_tok and not pred_tok:
        return 1.0
    r = calculate_metrics({"s": pred_tok}, {"s": gt_tok}, is_verbose=False)
    return float(r["mirco_f1_score"])


def evaluate_custom_table_teds(gt_raw: str, pred_raw: str) -> float:
    if TEDS is None:
        return 0.0
    teds = TEDS(structure_only=False, n_jobs=1)
    pred_inner = _unwrap_fenced_html(pred_raw)
    gt_flat = extract_and_clean_tables(gt_raw)
    pred_flat = extract_and_clean_tables(pred_inner)
    if not gt_flat.strip() and not pred_flat.strip():
        return 1.0
    if not gt_flat.strip() or not pred_flat.strip():
        return 0.0
    gt_flat = convert_to_halfwidth(gt_flat)
    pred_flat = convert_to_halfwidth(pred_flat)
    from lxml import etree, html as lxml_html

    parser = lxml_html.HTMLParser(encoding="utf-8", remove_comments=True)
    wrap = "<html><body>{}</body></html>"
    doc_gt = lxml_html.fromstring(wrap.format(gt_flat), parser=parser)
    doc_pd = lxml_html.fromstring(wrap.format(pred_flat), parser=parser)
    tabs_gt = doc_gt.xpath("body/table")
    tabs_pd = doc_pd.xpath("body/table")
    if not tabs_gt and not tabs_pd:
        return 1.0
    if not tabs_gt or not tabs_pd:
        return 0.0
    n = max(len(tabs_gt), len(tabs_pd))
    scores: list[float] = []
    for i in range(n):
        g_el = tabs_gt[i] if i < len(tabs_gt) else None
        p_el = tabs_pd[i] if i < len(tabs_pd) else None
        if g_el is None or p_el is None:
            scores.append(0.0)
            continue
        gs = etree.tostring(g_el, encoding="unicode", method="html")
        ps = etree.tostring(p_el, encoding="unicode", method="html")
        scores.append(float(teds.evaluate(wrap.format(ps), wrap.format(gs))))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate_single_custom_sample(gt_raw: str, pred_raw: str) -> tuple[float, float]:
    return (
        evaluate_custom_text_score(gt_raw, pred_raw),
        evaluate_custom_table_teds(gt_raw, pred_raw),
    )


def combined_custom_score(
    text_micro_f1: float, table_teds: float, table_weight: float = 0.9
) -> float:
    """综合分：(1-w)*text + w*table，w 为表格权重。"""
    w = max(0.0, min(1.0, float(table_weight)))
    return (1.0 - w) * text_micro_f1 + w * table_teds


def edit_similarity(pred: str, gt: str) -> float:
    """编辑距离相似度：1 - edit_dist / max(len(pred), len(gt))。"""
    edit_dist = nltk.edit_distance(pred, gt) / max(len(pred), len(gt))
    return 1 - edit_dist


def normalize_formula_strings(pred: str, gt: str) -> tuple[str, str]:
    """公式 pred/gt 字符串归一化。"""
    p = (
        pred.replace('\n', ' ')
        .replace('```latex', '')
        .replace('```', '')
        .replace('\t', ' ')
        .replace(' ', '')
    )
    g = gt.replace(' ', '')
    return p, g


def normalize_molecular_strings(pred: str, gt: str) -> tuple[str, str]:
    """分子式 pred/gt 字符串归一化。"""
    p = (
        pred.replace('\n', '')
        .replace(' ', '')
        .replace('<smiles>', '')
        .replace('</smiles>', '')
    )
    g = gt.replace(' ', '')
    return p, g


def collect_pairs(pred_dir: Path, gt_dir: Path) -> list[tuple[Path, Path]]:
    """收集预测和 GT 文件对"""
    pairs: list[tuple[Path, Path]] = []
    for gt_file in sorted(gt_dir.rglob("*.txt")):
        rel = gt_file.relative_to(gt_dir)
        pred_file = pred_dir / rel
        if not pred_file.is_file():
            matches = list(pred_dir.rglob(gt_file.name))
            if matches:
                pred_file = matches[0]
        if pred_file.is_file():
            pairs.append((gt_file, pred_file))
    return pairs


def detect_dataset_type(gt_dir: Path) -> str:
    """根据 GT 目录名推断 doc / table / formula / molecular。"""
    dir_name = gt_dir.name.lower()

    if "formula" in dir_name:
        return "formula"
    if "molecular" in dir_name:
        return "molecular"
    if "custom" in dir_name:
        return "custom"
    # table_photo_chn 等：纯表 HTML，op=table
    if "table" in dir_name:
        return "table"
    # doc_photo_chn / doc_scan_eng 等：整页 LaTeX，op=doc
    if "_doc_doc_" in dir_name:
        return "doc"
    # 未识别则 auto 下按 doc 处理，可显式传 --op
    return "doc"


def evaluate_doc_latex_ccocr(
    pairs: list[tuple[Path, Path]],
    gt_dir: Path,
    jobs: int = 1,
) -> dict:
    """多文件 LaTeX，按文件平均编辑相似度。"""
    scores: list[float] = []
    per_file: list[dict] = []

    def process_pair(gt_path: Path, pred_path: Path):
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
        pred_raw = pred_path.read_text(encoding="utf-8", errors="replace").strip()
        sc = evaluate_single_doc_ccocr(gt_raw, pred_raw)
        key = str(gt_path.relative_to(gt_dir))
        return sc, {
            "file": key,
            "edit_similarity": sc,
        }

    jobs = max(1, int(jobs))
    if jobs == 1:
        for gt_path, pred_path in tqdm(pairs, desc="Evaluating doc (LaTeX)"):
            sc, pf = process_pair(gt_path, pred_path)
            scores.append(sc)
            per_file.append(pf)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(process_pair, g, p) for g, p in pairs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating doc (LaTeX)"):
                sc, pf = fut.result()
                scores.append(sc)
                per_file.append(pf)

    mean_sc = sum(scores) / len(scores) if scores else 0.0
    return {
        "pred_dir": str(pairs[0][1].parent if pairs else ""),
        "gt_dir": str(gt_dir),
        "op": "doc",
        "num_files": len(pairs),
        "mean_edit_similarity": mean_sc,
        "per_file": per_file,
    }


def evaluate_formula_or_molecular(
    pairs: list[tuple[Path, Path]], 
    gt_dir: Path, 
    op: str,
    jobs: int = 1,
) -> dict:
    """评估公式或分子式（编辑距离）"""
    scores: list[float] = []
    per_file: list[dict] = []
    norm_fn = normalize_formula_strings if op == "formula" else normalize_molecular_strings
    
    def process_pair(gt_path: Path, pred_path: Path):
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
        pred_raw = pred_path.read_text(encoding="utf-8", errors="replace").strip()
        p, g = norm_fn(pred_raw, gt_raw)
        sc = edit_similarity(p, g)
        key = str(gt_path.relative_to(gt_dir))
        return sc, {
            "file": key,
            "edit_similarity": sc,
            "pred_norm_len": len(p),
            "gt_norm_len": len(g),
        }

    jobs = max(1, int(jobs))
    if jobs == 1:
        for gt_path, pred_path in tqdm(pairs, desc=f"Evaluating {op}"):
            sc, pf = process_pair(gt_path, pred_path)
            scores.append(sc)
            per_file.append(pf)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(process_pair, gt_path, pred_path) for gt_path, pred_path in pairs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc=f"Evaluating {op}"):
                sc, pf = fut.result()
                scores.append(sc)
                per_file.append(pf)
    
    mean_sc = sum(scores) / len(scores) if scores else 0.0
    report = {
        "pred_dir": str(pairs[0][1].parent if pairs else ""),
        "gt_dir": str(gt_dir),
        "op": op,
        "num_files": len(pairs),
        "mean_edit_similarity": mean_sc,
        "per_file": per_file,
    }
    return report


def evaluate_pure_table(
    pairs: list[tuple[Path, Path]],
    gt_dir: Path,
    jobs: int = 1,
) -> dict:
    """多文件表格：```html``` 抽取、清洗、半角化后按文件算 TEDS 再平均。"""
    if TEDS is None:
        print("Error: TEDS not available, cannot evaluate tables", file=sys.stderr)
        return {"error": "TEDS not available"}

    scores: list[float] = []
    per_file: list[dict] = []

    def process_pair(gt_path: Path, pred_path: Path):
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
        pred_raw = pred_path.read_text(encoding="utf-8", errors="replace")
        pred_raw = pred_raw.strip()
        sc = evaluate_single_table_ccocr(gt_raw, pred_raw)
        key = str(gt_path.relative_to(gt_dir))
        return sc, {"file": key, "teds": sc}

    jobs = max(1, int(jobs))
    if jobs == 1:
        for gt_path, pred_path in tqdm(pairs, desc="Evaluating table (TEDS)"):
            sc, pf = process_pair(gt_path, pred_path)
            scores.append(sc)
            per_file.append(pf)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(process_pair, g, p) for g, p in pairs]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Evaluating table (TEDS)"):
                sc, pf = fut.result()
                scores.append(sc)
                per_file.append(pf)

    mean_teds = sum(scores) / len(scores) if scores else 0.0
    return {
        "pred_dir": str(pairs[0][1].parent if pairs else ""),
        "gt_dir": str(gt_dir),
        "op": "table",
        "num_files": len(pairs),
        "mean_teds": mean_teds,
        "per_file": per_file,
    }


def evaluate_custom(
    pairs: list[tuple[Path, Path]],
    gt_dir: Path,
    jobs: int = 1,
    table_weight: float = 0.9,
) -> dict:
    """custom：非表 micro-F1（字符级）+ 多表 TEDS；综合分 = (1-w)*text + w*table。"""
    text_scores: list[float] = []
    table_scores: list[float] = []
    combined_scores: list[float] = []
    per_file: list[dict] = []

    def process_pair(gt_path: Path, pred_path: Path):
        gt_raw = gt_path.read_text(encoding="utf-8", errors="replace")
        pred_raw = pred_path.read_text(encoding="utf-8", errors="replace").strip()
        ts, tbs = evaluate_single_custom_sample(gt_raw, pred_raw)
        cb = combined_custom_score(ts, tbs, table_weight)
        key = str(gt_path.relative_to(gt_dir))
        return ts, tbs, cb, {
            "file": key,
            "text_micro_f1": ts,
            "table_teds": tbs,
            "combined": cb,
        }

    jobs = max(1, int(jobs))
    if jobs == 1:
        for gt_path, pred_path in tqdm(pairs, desc="Evaluating custom (text+table)"):
            ts, tbs, cb, pf = process_pair(gt_path, pred_path)
            text_scores.append(ts)
            table_scores.append(tbs)
            combined_scores.append(cb)
            per_file.append(pf)
    else:
        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = [ex.submit(process_pair, g, p) for g, p in pairs]
            for fut in tqdm(
                as_completed(futures), total=len(futures), desc="Evaluating custom (text+table)"
            ):
                ts, tbs, cb, pf = fut.result()
                text_scores.append(ts)
                table_scores.append(tbs)
                combined_scores.append(cb)
                per_file.append(pf)

    n = len(pairs)
    mean_text = sum(text_scores) / n if n else 0.0
    mean_tbl = sum(table_scores) / n if n else 0.0
    mean_combined = sum(combined_scores) / n if n else 0.0
    return {
        "pred_dir": str(pairs[0][1].parent if pairs else ""),
        "gt_dir": str(gt_dir),
        "op": "custom",
        "custom_table_weight": max(0.0, min(1.0, float(table_weight))),
        "num_files": n,
        "mean_text_micro_f1": mean_text,
        "mean_table_teds": mean_tbl,
        "mean_combined": mean_combined,
        "per_file": per_file,
    }


def main():
    parser = argparse.ArgumentParser(
        description="文档解析评估：根据数据集前缀自动选择评估策略"
    )
    parser.add_argument("--pred_dir", type=str, required=True, help="预测结果目录")
    parser.add_argument("--gt_dir", type=str, required=True, help="Ground Truth 目录")
    parser.add_argument(
        "--op",
        choices=("auto", "doc", "table", "formula", "molecular", "custom"),
        default="auto",
        help="auto=目录名检测；custom=非表文本micro-F1+表TEDS；doc/table/formula/molecular 见各任务",
    )
    parser.add_argument("--jobs", type=int, default=1, help="dataset internal parallel workers")
    parser.add_argument(
        "--custom_table_weight",
        type=float,
        default=0.9,
        help="custom 任务：表格 TEDS 权重 w，综合分=(1-w)*text_micro_f1+w*table_teds，默认 0.9",
    )
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir).resolve()
    gt_dir = Path(args.gt_dir).resolve()
    
    if not gt_dir.is_dir():
        print(f"Error: gt_dir does not exist: {gt_dir}")
        return 1

    pairs = collect_pairs(pred_dir, gt_dir)
    if not pairs:
        print("Error: No matching prediction and ground truth files.")
        return 1

    # 自动检测数据集类型
    if args.op == "auto":
        detected_type = detect_dataset_type(gt_dir)
        print(f"Auto-detected dataset type: {detected_type}")
        print(f"Dataset directory: {gt_dir.name}")
        op = detected_type
    else:
        op = args.op

    # 根据类型调用不同的评估函数
    if op in ("formula", "molecular"):
        report = evaluate_formula_or_molecular(pairs, gt_dir, op, jobs=args.jobs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nmean edit similarity: {round(report['mean_edit_similarity'], 4)}")

    elif op == "doc":
        report = evaluate_doc_latex_ccocr(pairs, gt_dir, jobs=args.jobs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"\nmean edit similarity: {round(report['mean_edit_similarity'], 4)}")

    elif op == "table":
        report = evaluate_pure_table(pairs, gt_dir, jobs=args.jobs)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if "mean_teds" in report:
            print(f"\nmean TEDS: {round(report['mean_teds'], 4)}")

    elif op == "custom":
        report = evaluate_custom(
            pairs, gt_dir, jobs=args.jobs, table_weight=args.custom_table_weight
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        w = report["custom_table_weight"]
        print(
            f"\nmean text micro-F1: {round(report['mean_text_micro_f1'], 4)}"
            f"  |  mean table TEDS: {round(report['mean_table_teds'], 4)}"
            f"  |  mean combined (table w={w}): {round(report['mean_combined'], 4)}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
