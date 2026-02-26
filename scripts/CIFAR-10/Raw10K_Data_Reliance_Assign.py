"""
Assign data_reliance ∈ [0,1] for the RAW 10k pool using OpenAI vision model.

Input (read-only from cl_data):
  - $CL_DATA_RAW/cifar10/v1_seed42/annotations/index_map.jsonl
  - images referenced by relpath under $CL_DATA_RAW/cifar10/v1_seed42/

Prompt:
  - system prompt from Prompt/prompt.txt (or --prompt_path)

Output (local repo ONLY, no images copied, no ground-truth leaked):
  - ./local_annotations/reliance_raw10k_seed42.jsonl
  - ./local_annotations/reliance_raw10k_seed42_meta.json

Each output row (NO orig_* fields):
  { id, relpath, label, class_name, is_noisy, data_reliance, rationale, model, prompt_path, seed }
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import OpenAI


CIFAR10_CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def to_data_url_png(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_json_maybe(text: str) -> Optional[Dict]:
    t = text.strip()
    if t.startswith("```"):
        t = t.strip("`")
        if t.lower().startswith("json"):
            t = t[4:].strip()
    try:
        return json.loads(t)
    except Exception:
        lb = t.find("{")
        rb = t.rfind("}")
        if lb != -1 and rb != -1 and rb > lb:
            try:
                return json.loads(t[lb:rb + 1])
            except Exception:
                return None
        return None


def load_system_prompt(prompt_path: Path) -> str:
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    text = prompt_path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Prompt file is empty: {prompt_path}")
    return text


def load_done_ids(out_path: Path) -> Set[int]:
    done: Set[int] = set()
    if not out_path.exists():
        return done
    with out_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                done.add(int(r["id"]))
            except Exception:
                continue
    return done


def score_one(client: OpenAI, model: str, system_msg: str, data_url: str, label: int, class_name: str) -> Dict:
    """
    IMPORTANT: We only provide the (possibly noisy) label and the image.
    We do NOT provide any orig_* (ground-truth) fields to the model.
    """
    user_text = (
        f"Label to evaluate: {label} ({class_name}).\n"
        "Score the reliability of THIS label for this image."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_msg},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                ],
            },
        ],
        temperature=0.0,
    )

    content = resp.choices[0].message.content or ""
    parsed = parse_json_maybe(content)
    if not parsed or "data_reliance" not in parsed or "rationale" not in parsed:
        raise ValueError(f"Model output not valid JSON with required keys. Output was:\n{content}")

    dr = round(clamp01(float(parsed["data_reliance"])), 2)
    return {"data_reliance": dr, "rationale": str(parsed["rationale"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = all (10000). Use for quick test.")
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--prompt_path", type=str, default="Prompt/prompt.txt")
    ap.add_argument("--out_path", type=str, default="local_annotations/reliance_raw10k_seed42.jsonl")
    ap.add_argument("--meta_path", type=str, default="local_annotations/reliance_raw10k_seed42_meta.json")
    args = ap.parse_args()

    cl_raw = os.environ.get("CL_DATA_RAW")
    if not cl_raw:
        raise RuntimeError("CL_DATA_RAW not set. Run: source ./env_superpod.sh")

    raw_dir = Path(cl_raw) / "cifar10" / "v1_seed42"
    raw_map = raw_dir / "annotations" / "index_map.jsonl"
    if not raw_map.exists():
        raise FileNotFoundError(f"Missing raw index map: {raw_map}")

    prompt_path = Path(args.prompt_path)
    system_msg = load_system_prompt(prompt_path)

    rows = read_jsonl(raw_map)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out_path)
    meta_path = Path(args.meta_path)
    ensure_dir(out_path.parent)
    ensure_dir(meta_path.parent)

    done = load_done_ids(out_path)
    client = OpenAI()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    n_written = 0

    with out_path.open("a", encoding="utf-8") as out:
        for r in rows:
            rid = int(r["id"])
            if rid in done:
                continue

            img_path = raw_dir / r["relpath"]  # images/xxxxxx.png
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            data_url = to_data_url_png(img_path)

            label = int(r["label"])
            class_name = r.get("class_name") or CIFAR10_CLASS_NAMES[label]
            is_noisy = bool(r.get("is_noisy", False))

            # retry once
            try:
                scored = score_one(client, args.model, system_msg, data_url, label, class_name)
            except Exception:
                time.sleep(0.8)
                scored = score_one(client, args.model, system_msg, data_url, label, class_name)

            out_row = {
                "id": rid,
                "relpath": r["relpath"],  # relative to raw_dir
                "label": label,
                "class_name": class_name,
                "is_noisy": is_noisy,
                "data_reliance": float(scored["data_reliance"]),
                "rationale": scored["rationale"],
                "seed": args.seed,
                "model": args.model,
                "prompt_path": str(prompt_path),
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out.flush()
            n_written += 1

            if args.sleep_s and args.sleep_s > 0:
                time.sleep(args.sleep_s)

    meta = {
        "dataset": "CIFAR-10",
        "step": "data_reliance_raw10k",
        "started_at": started,
        "seed": args.seed,
        "model": args.model,
        "prompt_path": str(prompt_path),
        "raw_index_map": str(raw_map),
        "raw_dir": str(raw_dir),
        "output_jsonl": str(out_path),
        "written_this_run": n_written,
        "note": "Output stored in local_annotations only. No images copied. No orig_* fields stored.",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] meta:  {meta_path}")


if __name__ == "__main__":
    main()