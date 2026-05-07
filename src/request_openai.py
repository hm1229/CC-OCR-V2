import re
import json
import asyncio
import base64
from pathlib import Path
from typing import Any, Dict, List, Optional

import tqdm
import aiofiles
from openai import AsyncOpenAI
from io import BytesIO
from PIL import Image

# ================== Configuration ==================
IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
REPO_ROOT = Path(__file__).resolve().parent.parent
OCR_DATASETS_ROOT = REPO_ROOT / "ocr_datasets"  # base to compute task-relative path (2/3-level adaptive)
MODEL_NAME = ""
MAX_CONC = 8
MAX_RETRIES = 10
OPENAI_KEY = ""
API_BASE = ""

client: Optional[AsyncOpenAI] = None


def natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def get_task_rel_path(ocr_root: Path, ocr_datasets_root: Path) -> str:
    """Task path relative to ocr_datasets (e.g. grounding/object_grounding or vqa). Adaptive for 2/3-level."""
    try:
        rel = ocr_root.resolve().relative_to(ocr_datasets_root.resolve())
    except ValueError:
        rel = Path(ocr_root.name)
    return str(rel).replace("\\", "/")


def get_model_folder_name(model_name: str) -> str:
    clean = re.sub(r"[^\w\-_]", "_", model_name)
    return f"pred_{clean}"


def get_output_base(model_name: str) -> Path:
    clean = re.sub(r"[^\w\-_]", "_", model_name)
    return REPO_ROOT / "results" / clean


def encode_image_to_base64(image_path: Path) -> str:
    with Image.open(image_path) as im:
        im = im.convert("RGB")
        buf = BytesIO()
        im.save(buf, format="JPEG", quality=95, optimize=True)
        data = buf.getvalue()
    return base64.b64encode(data).decode("utf-8")


def build_messages(images: List[Path], user_text: str) -> list:
    content = []
    for img in images:
        b64 = encode_image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    content.append({"type": "text", "text": user_text})
    return [{"role": "user", "content": content}]


def find_image_for_stem(images_subset_dir: Path, stem: str) -> List[Path]:
    """
    Find image(s) for a given stem in images/<subset>/.
    Returns a list of image paths:
    - Single image: [<stem>.<ext>]
    - Multi-page: [<stem>/page_1.jpg, <stem>/page_2.jpg, ...]
    """
    if not images_subset_dir.is_dir():
        raise FileNotFoundError(f"Image subset folder not found: {images_subset_dir}")
    
    # Check for multi-page directory first
    multipage_dir = images_subset_dir / stem
    if multipage_dir.is_dir():
        pages = sorted(
            [p for p in multipage_dir.iterdir() 
             if p.is_file() and p.suffix.lower() in IMG_EXTS],
            key=lambda p: natural_key(p.name)
        )
        if pages:
            return pages
    
    # Check for single image file(s)
    matches: List[Path] = []
    for p in images_subset_dir.iterdir():
        if p.is_file() and p.stem == stem and p.suffix.lower() in IMG_EXTS:
            matches.append(p)
    
    if not matches:
        raise FileNotFoundError(f"No image for stem {stem!r} under {images_subset_dir}")
    
    matches.sort(key=lambda p: natural_key(p.name))
    return matches


def load_ocr_samples(
    ocr_root: Path,
    subsets: Optional[List[str]] = None,
    limit: Optional[int] = None,
    ocr_datasets_root: Optional[Path] = None,
) -> List[dict]:
    """
    <ocr_root>/question/<subset>/*.txt + images/<subset>/<stem>.<img>
    Each sample gets task_rel_path (relative to ocr_datasets_root) for mirror output layout.
    """
    if ocr_datasets_root is None:
        ocr_datasets_root = OCR_DATASETS_ROOT
    task_rel_path = get_task_rel_path(ocr_root, ocr_datasets_root)

    q_root = ocr_root / "question"
    if not q_root.is_dir():
        raise FileNotFoundError(f"Missing question/ under ocr root: {ocr_root}")

    rows: List[dict] = []
    subset_dirs = [p for p in q_root.iterdir() if p.is_dir()]
    subset_dirs.sort(key=lambda p: natural_key(p.name))

    for subdir in subset_dirs:
        name = subdir.name
        if subsets is not None and name not in subsets:
            continue
        q_files = [p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"]
        q_files.sort(key=lambda p: natural_key(p.stem))
        img_sub = ocr_root / "images" / name

        for qfile in q_files:
            stem = qfile.stem
            prompt = qfile.read_text(encoding="utf-8")
            img_paths = find_image_for_stem(img_sub, stem)
            rows.append({
                "ocr_root": str(ocr_root.resolve()),
                "task": ocr_root.name,
                "task_rel_path": task_rel_path,
                "subset": name,
                "id": stem,
                "prompt": prompt,
                "question_path": str(qfile.resolve()),
                "image_paths": [str(p.resolve()) for p in img_paths],  # Now a list
            })
            if limit is not None and limit > 0 and len(rows) >= limit:
                return rows
    return rows


def prediction_output_path(
    output_base: Path,
    task_rel_path: str,
    subset: str,
    model_folder_name: str,
    sample_id: str,
) -> Path:
    """Target path for a sample's prediction txt (same layout as write_ocr_prediction)."""
    return output_base / task_rel_path / subset / model_folder_name / f"{sample_id}.txt"


def prediction_exists_nonempty(path: Path) -> bool:
    """True if prediction file exists and has content (treat empty as not done, for retry)."""
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


async def write_ocr_prediction(
    output_base: Path,
    task_rel_path: str,
    subset: str,
    model_folder_name: str,
    sample_id: str,
    answer_text: str,
) -> Path:
    """Save to output_base / task_rel_path / subset / model_folder_name / sample_id.txt (mirrors original layout)."""
    dest = prediction_output_path(output_base, task_rel_path, subset, model_folder_name, sample_id)
    dest.parent.mkdir(parents=True, exist_ok=True)
    async with aiofiles.open(dest, "w", encoding="utf-8") as fh:
        await fh.write(answer_text)
    return dest


async def call_api_once(messages, model_name: str) -> str:
    try:
        resp = await client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.0,
            extra_body={"enable_thinking":False}
        )
        return resp.choices[0].message.content
    except Exception as e:
        raise RuntimeError(f"OpenAI API call failed: {e}")


def _sample_key(sample: dict) -> str:
    return f"{sample['subset']}/{sample['id']}"


async def process_one(
    sample: dict,
    sem: asyncio.Semaphore,
    model_name: str,
    max_images: Optional[int],
    output_base: Path,
    model_folder_name: str,
) -> dict:
    async with sem:
        task = sample["task"]
        key = _sample_key(sample)
        user_text = sample["prompt"]

        try:
            images = [Path(p) for p in sample["image_paths"]]
        except Exception as e:
            return {
                "task": task,
                "subset": sample.get("subset"),
                "id": sample.get("id"),
                "key": key,
                "error": str(e),
                "retry_attempts": 0,
            }

        try:
            if max_images and max_images > 0:
                images = images[:max_images]
            messages = build_messages(images, user_text)

            raw_text = None
            attempts = 0
            last_err = None
            for attempt in range(MAX_RETRIES):
                try:
                    raw_text = await call_api_once(messages, model_name)
                    attempts = attempt + 1
                    break
                except Exception as e:
                    last_err = str(e)
                    attempts = attempt + 1
                    if attempt < MAX_RETRIES - 1:
                        print(f"[RETRY] {task}/{key} attempt {attempt+1} failed: {e}")
                    else:
                        print(f"[FAILED] {task}/{key} all {MAX_RETRIES} attempts failed: {e}")

            if raw_text is None:
                return {
                    "task": task,
                    "subset": sample.get("subset"),
                    "id": sample.get("id"),
                    "key": key,
                    "error": f"API failed: {last_err}",
                    "retry_attempts": attempts,
                }

            out_txt = raw_text or ""
            dest = await write_ocr_prediction(
                output_base,
                sample["task_rel_path"],
                sample["subset"],
                model_folder_name,
                sample["id"],
                out_txt,
            )
            written_path = str(dest)

            return {
                "task": task,
                "subset": sample["subset"],
                "id": sample["id"],
                "key": key,
                "ocr_root": sample["ocr_root"],
                "raw_response": raw_text,
                "retry_attempts": attempts,
                "images": [str(p) for p in images],
                "answer_path": written_path,
            }

        except Exception as e:
            return {
                "task": sample.get("task"),
                "subset": sample.get("subset"),
                "id": sample.get("id"),
                "key": key,
                "error": str(e),
                "retry_attempts": 0,
            }


async def main_async(
    samples: List[dict],
    model_name: str,
    concurrency: int,
    api_key: str,
    api_base: str,
    max_images: Optional[int],
    output_base: Path,
    model_folder_name: str,
    task_rel_path: str,
):
    global client
    client = AsyncOpenAI(api_key=api_key, base_url=api_base)

    meta_dir = output_base / task_rel_path
    meta_dir.mkdir(parents=True, exist_ok=True)
    meta_path = meta_dir / f"run_meta_{model_folder_name}.jsonl"

    sem = asyncio.Semaphore(concurrency)
    tasks = [
        asyncio.create_task(
            process_one(s, sem, model_name, max_images, output_base, model_folder_name)
        )
        for s in samples
    ]

    ok = 0
    fail = 0
    async with aiofiles.open(meta_path, "w", encoding="utf-8") as meta_fh:
        for coro in tqdm.tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Processing progress"):
            res = await coro
            await meta_fh.write(json.dumps(res, ensure_ascii=False) + "\n")
            await meta_fh.flush()
            if res.get("error"):
                fail += 1
            else:
                ok += 1

    print(f"\nProcessing completed! Success: {ok}  Failed: {fail}")
    print(f"Predictions under: {output_base / task_rel_path} / <subset> / {model_folder_name} /")
    print(f"Per-sample metadata: {meta_path}")


def main():
    import argparse
    global MAX_CONC, MAX_RETRIES, OPENAI_KEY, API_BASE

    parser = argparse.ArgumentParser(
        description=(
            "Run OpenAI-compatible vision API on ocr_datasets layout: "
            "question/, images/ → writes raw model text to <subset>/<id>.txt and run_meta.jsonl (raw_response only, no JSON parse)."
        )
    )
    parser.add_argument(
        "--ocr-root",
        type=str,
        required=True,
        help="Task folder containing question/ and images/ (e.g. ocr_datasets/grounding/object_grounding)",
    )
    parser.add_argument(
        "--subset",
        action="append",
        default=None,
        help="Only this subset under question/ (repeatable). Default: all.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output base dir (default: results/<model_name>). Saves to <output>/<task_rel>/<subset>/<model>/<id>.txt",
    )
    parser.add_argument(
        "--ocr-datasets-base",
        type=str,
        default=None,
        help="Base path for ocr_datasets to compute task-relative path (default: repo/ocr_datasets)",
    )
    parser.add_argument("--model", type=str, default=MODEL_NAME, help="Model name")
    parser.add_argument("--api-key", type=str, default=OPENAI_KEY, help="OpenAI API key")
    parser.add_argument("--api-base", type=str, default=API_BASE, help="OpenAI API base url")
    parser.add_argument("--concurrency", type=int, default=MAX_CONC, help="Maximum concurrency")
    parser.add_argument("--max-retries", type=int, default=MAX_RETRIES, help="Maximum retry attempts")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N samples (debug)")
    parser.add_argument("--max-images", type=int, default=None, help="Max images per sample (default: all)")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip API calls when prediction txt already exists and is non-empty",
    )
    args = parser.parse_args()

    MAX_CONC = args.concurrency
    MAX_RETRIES = args.max_retries
    OPENAI_KEY = args.api_key
    API_BASE = args.api_base

    ocr_root = Path(args.ocr_root).expanduser().resolve()
    if not ocr_root.is_dir():
        raise FileNotFoundError(f"--ocr-root is not a directory: {ocr_root}")

    ocr_datasets_root = Path(args.ocr_datasets_base).expanduser().resolve() if args.ocr_datasets_base else OCR_DATASETS_ROOT
    samples = load_ocr_samples(ocr_root, subsets=args.subset, limit=args.limit, ocr_datasets_root=ocr_datasets_root)
    if not samples:
        raise SystemExit(f"No samples under {ocr_root / 'question'} (check --subset).")

    task_rel_path = samples[0]["task_rel_path"]
    model_folder_name = get_model_folder_name(args.model)
    output_base = Path(args.output).expanduser().resolve() if args.output else get_output_base(args.model)

    if args.skip_existing:
        total = len(samples)
        samples = [
            s
            for s in samples
            if not prediction_exists_nonempty(
                prediction_output_path(
                    output_base,
                    s["task_rel_path"],
                    s["subset"],
                    model_folder_name,
                    s["id"],
                )
            )
        ]
        skipped = total - len(samples)
        print(f"--skip-existing: skipped {skipped} already done, {len(samples)} remaining (of {total})")
        if not samples:
            print("Nothing to do.")
            return

    asyncio.run(
        main_async(
            samples,
            args.model,
            args.concurrency,
            args.api_key,
            args.api_base,
            args.max_images,
            output_base,
            model_folder_name,
            task_rel_path,
        )
    )


if __name__ == "__main__":
    main()
