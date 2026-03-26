import argparse
import base64
import json
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


def load_failed_ids(failed_path: Path) -> Set[int]:
    failed: Set[int] = set()
    if not failed_path.exists():
        return failed
    with failed_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                failed.add(int(r["id"]))
            except Exception:
                continue
    return failed


def score_one(
    client: OpenAI,
    model: str,
    system_msg: str,
    data_url: str,
    label: int,
    class_name: str,
) -> Dict:
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
    return {
        "data_reliance": dr,
        "rationale": str(parsed["rationale"]),
    }


def try_score_with_retries(
    client: OpenAI,
    model: str,
    system_msg: str,
    data_url: str,
    label: int,
    class_name: str,
    max_retries: int = 3,
    retry_sleep_s: float = 1.0,
) -> Dict:
    last_err = None
    for attempt in range(max_retries):
        try:
            return score_one(
                client=client,
                model=model,
                system_msg=system_msg,
                data_url=data_url,
                label=label,
                class_name=class_name,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(retry_sleep_s)
    raise last_err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="gpt-4o")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="0 = all train rows")
    ap.add_argument("--sleep_s", type=float, default=0.0)
    ap.add_argument("--prompt_path", type=str, required=True)

    ap.add_argument("--old_reliance_path", type=str, required=True)
    ap.add_argument("--train_map_path", type=str, required=True)
    ap.add_argument("--train_images_root", type=str, required=True)
    ap.add_argument("--out_path", type=str, required=True)
    ap.add_argument("--meta_path", type=str, required=True)
    ap.add_argument("--failed_path", type=str, required=True)

    args = ap.parse_args()

    prompt_path = Path(args.prompt_path)
    system_msg = load_system_prompt(prompt_path)

    old_reliance_path = Path(args.old_reliance_path)
    train_map_path = Path(args.train_map_path)
    train_images_root = Path(args.train_images_root)
    out_path = Path(args.out_path)
    meta_path = Path(args.meta_path)
    failed_path = Path(args.failed_path)

    if not old_reliance_path.exists():
        raise FileNotFoundError(f"Missing old reliance file: {old_reliance_path}")
    if not train_map_path.exists():
        raise FileNotFoundError(f"Missing train index map: {train_map_path}")
    if not train_images_root.exists():
        raise FileNotFoundError(f"Missing train images root: {train_images_root}")

    old_reliance_rows = read_jsonl(old_reliance_path)
    train_rows = read_jsonl(train_map_path)

    if args.limit and args.limit > 0:
        train_rows = train_rows[: args.limit]

    old_by_id: Dict[int, Dict] = {int(r["id"]): r for r in old_reliance_rows}

    ensure_dir(out_path.parent)
    ensure_dir(meta_path.parent)
    ensure_dir(failed_path.parent)

    done = load_done_ids(out_path)
    failed_before = load_failed_ids(failed_path)

    client = OpenAI()

    started = time.strftime("%Y-%m-%d %H:%M:%S")
    n_written = 0
    n_reused_clean = 0
    n_rescored_noisy = 0
    n_failed_this_run = 0

    with out_path.open("a", encoding="utf-8") as out, failed_path.open("a", encoding="utf-8") as failed_out:
        for r in train_rows:
            rid = int(r["id"])
            if rid in done:
                continue
            if rid in failed_before:
                continue

            if rid not in old_by_id:
                raise KeyError(f"id={rid} from train_index_map not found in old reliance file")

            old_row = old_by_id[rid]

            relpath = r["relpath"]
            img_path = train_images_root / Path(relpath).name
            if not img_path.exists():
                raise FileNotFoundError(f"Missing image: {img_path}")

            clean_label = int(r["orig_label"])
            clean_class = r.get("orig_class") or CIFAR10_CLASS_NAMES[clean_label]
            was_noisy = bool(r.get("is_noisy", False))

            if not was_noisy:
                out_row = {
                    "id": rid,
                    "relpath": relpath,
                    "label": clean_label,
                    "class_name": clean_class,
                    "is_noisy": False,
                    "data_reliance": float(round(clamp01(float(old_row["data_reliance"])), 2)),
                    "rationale": str(old_row["rationale"]),
                    "seed": args.seed,
                    "model": old_row.get("model", args.model),
                    "prompt_path": old_row.get("prompt_path", str(prompt_path)),
                }
                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out.flush()
                n_written += 1
                n_reused_clean += 1
                continue

            data_url = to_data_url_png(img_path)

            try:
                scored = try_score_with_retries(
                    client=client,
                    model=args.model,
                    system_msg=system_msg,
                    data_url=data_url,
                    label=clean_label,
                    class_name=clean_class,
                    max_retries=3,
                    retry_sleep_s=1.0,
                )

                out_row = {
                    "id": rid,
                    "relpath": relpath,
                    "label": clean_label,
                    "class_name": clean_class,
                    "is_noisy": False,
                    "data_reliance": float(scored["data_reliance"]),
                    "rationale": scored["rationale"],
                    "seed": args.seed,
                    "model": args.model,
                    "prompt_path": str(prompt_path),
                }

                out.write(json.dumps(out_row, ensure_ascii=False) + "\n")
                out.flush()
                n_written += 1
                n_rescored_noisy += 1

                if args.sleep_s and args.sleep_s > 0:
                    time.sleep(args.sleep_s)

            except Exception as e:
                fail_row = {
                    "id": rid,
                    "relpath": relpath,
                    "label": clean_label,
                    "class_name": clean_class,
                    "error": str(e),
                }
                failed_out.write(json.dumps(fail_row, ensure_ascii=False) + "\n")
                failed_out.flush()
                n_failed_this_run += 1
                print(f"[WARN] failed id={rid}: {e}")
                continue

    meta = {
        "dataset": "CIFAR-10",
        "step": "train_clean_reliance_from_old_reliance_plus_rescore_noisy",
        "started_at": started,
        "seed": args.seed,
        "model_for_rescored_rows": args.model,
        "prompt_path": str(prompt_path),
        "old_reliance_path": str(old_reliance_path),
        "train_index_map": str(train_map_path),
        "train_images_root": str(train_images_root),
        "output_jsonl": str(out_path),
        "failed_jsonl": str(failed_path),
        "written_this_run": n_written,
        "reused_clean_rows_this_run": n_reused_clean,
        "rescored_noisy_rows_this_run": n_rescored_noisy,
        "failed_rows_this_run": n_failed_this_run,
        "note": (
            "Output is a full clean-train reliance file. "
            "Originally clean rows are reused from old reliance; originally noisy rows are rescored "
            "using orig_label/orig_class from train_index_map. All output rows set is_noisy=false. "
            "Rows that fail rescoring are logged to failed_jsonl and skipped."
        ),
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] wrote: {out_path}")
    print(f"[OK] meta:  {meta_path}")
    print(f"[OK] reused_clean_rows_this_run: {n_reused_clean}")
    print(f"[OK] rescored_noisy_rows_this_run: {n_rescored_noisy}")
    print(f"[OK] failed_rows_this_run: {n_failed_this_run}")


if __name__ == "__main__":
    main()