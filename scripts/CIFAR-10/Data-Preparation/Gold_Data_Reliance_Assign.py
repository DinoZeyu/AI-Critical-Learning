"""
Use OpenAI (vision) to assign data_reliance ∈ [0,1] for GOLD 1k images.

Input (read-only from cl_data):
  - $CL_DATA_PROCESSED/cifar10/v1_seed42/annotations/gold_index_map.jsonl
  - Each row has relpath like: gold/images/000123.png (under processed dir)

Prompt:
  - Read system prompt from a text file (default: Prompt/prompt.txt)

Output (local repo, NOT in cl_data):
  - ./local_annotations/reliance_gold_seed42.jsonl

Each output row keeps id alignment:
  { gold_id, raw_id, relpath, label, class_name, data_reliance, rationale, model, prompt_path }
"""

import argparse
import base64
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set

from openai import OpenAI


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def to_data_url_png(image_path: Path) -> str:
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_done_gold_ids(out_path: Path) -> Set[int]:
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
                done.add(int(r["gold_id"]))
            except Exception:
                continue
    return done


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def parse_json_maybe(text: str) -> Optional[Dict]:
    """
    Try to parse JSON from model output.
    Handles cases where model wraps JSON in markdown fences.
    """
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


def score_one(
    client: OpenAI,
    model: str,
    data_url: str,
    label: int,
    class_name: str,
    system_msg: str,
) -> Dict:
    """
    Call Chat Completions with image + system prompt loaded from file.
    Returns dict with keys: data_reliance (float), rationale (str)
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
                    {"type": "image_url", "image_url": {"url": data_url}},
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
    rat = str(parsed["rationale"])
    return {"data_reliance": dr, "rationale": rat}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = process all (1000). Use e.g. 10 for a quick test.")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Optional throttle between requests.")
    ap.add_argument("--out_path", type=str, default="local_annotations/reliance_gold_seed42.jsonl")
    ap.add_argument("--prompt_path", type=str, default="Prompt/prompt.txt",
                    help="Path to system prompt .txt file (relative to repo root).")
    args = ap.parse_args()

    cl_proc = os.environ.get("CL_DATA_PROCESSED")
    if not cl_proc:
        raise RuntimeError("CL_DATA_PROCESSED not set. Run: source ./env_superpod.sh")

    proc_dir = Path(cl_proc) / "cifar10" / "v1_seed42"
    gold_map = proc_dir / "annotations" / "gold_index_map.jsonl"
    if not gold_map.exists():
        raise FileNotFoundError(f"Missing gold_index_map.jsonl: {gold_map}")

    prompt_path = Path(args.prompt_path)
    system_msg = load_system_prompt(prompt_path)

    rows = read_jsonl(gold_map)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]

    out_path = Path(args.out_path)
    ensure_dir(out_path.parent)

    done_gold_ids = load_done_gold_ids(out_path)
    client = OpenAI()

    with out_path.open("a", encoding="utf-8") as out:
        for r in rows:
            gold_id = int(r["gold_id"])
            if gold_id in done_gold_ids:
                continue

            img_path = proc_dir / r["relpath"]  # e.g. gold/images/000123.png
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            data_url = to_data_url_png(img_path)
            label = int(r["label"])
            class_name = r.get("class_name", "")

            # Retry once if JSON parse fails or transient errors
            try:
                scored = score_one(client, args.model, data_url, label, class_name, system_msg)
            except Exception:
                time.sleep(0.8)
                scored = score_one(client, args.model, data_url, label, class_name, system_msg)

            out_row = {
                "gold_id": gold_id,
                "raw_id": int(r["raw_id"]),
                "relpath": r["relpath"],
                "label": label,
                "class_name": class_name,
                "data_reliance": float(scored["data_reliance"]),
                "rationale": scored["rationale"],
                "seed": args.seed,
                "model": args.model,
                "prompt_path": str(prompt_path),
            }
            out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            out.flush()

            if args.sleep_s and args.sleep_s > 0:
                time.sleep(args.sleep_s)

    print(f"[OK] wrote: {out_path}")


if __name__ == "__main__":
    main()