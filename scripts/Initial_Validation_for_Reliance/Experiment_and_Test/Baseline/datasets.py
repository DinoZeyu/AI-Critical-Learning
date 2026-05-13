import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any, Set

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T


@dataclass
class JsonlSample:
    id: int
    relpath: str
    label: int
    class_name: str
    meta: Dict[str, Any]


def _read_jsonl_generic(jsonl_path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_index_map_jsonl(index_jsonl: str) -> List[JsonlSample]:
    rows = _read_jsonl_generic(index_jsonl)
    samples: List[JsonlSample] = []
    for obj in rows:
        sid = int(obj["id"])
        relpath = str(obj["relpath"])
        label = int(obj["label"])
        class_name = str(obj.get("class_name", ""))
        samples.append(JsonlSample(id=sid, relpath=relpath, label=label, class_name=class_name, meta=obj))
    return samples


def _load_golden_gold_raw_ids(golden_gold_jsonl: str) -> Set[int]:
    """
    golden gold jsonl row example:
      {"gg_id": 0, "raw_id": 5736, "raw_relpath": "...", "label": 1, ...}
    We exclude raw_id from training set. raw_id == train_index_map.jsonl["id"].
    """
    raw_ids: Set[int] = set()
    rows = _read_jsonl_generic(golden_gold_jsonl)
    for obj in rows:
        if "raw_id" not in obj:
            raise KeyError(f"Golden gold jsonl missing 'raw_id' field: {golden_gold_jsonl}")
        raw_ids.add(int(obj["raw_id"]))
    return raw_ids


def _resolve_image_path(images_root: Path, relpath: str) -> Path:
    """
    relpath variants we might see:
      - "images/000000.png"
      - "000000.png"
      - "train_images/000000.png"
      - "test_images/000000.png"

    images_root is typically:
      .../train_images   or   .../test_images

    We try multiple candidates in order.
    """
    rel = Path(relpath)

    candidates = []

    # 1) direct join: root / relpath
    candidates.append(images_root / rel)

    # 2) root / basename
    candidates.append(images_root / rel.name)

    # 3) if relpath contains "train_images/..." or "test_images/...", strip that prefix
    parts = rel.parts
    if len(parts) >= 2 and parts[0] in ("train_images", "test_images"):
        candidates.append(images_root / Path(*parts[1:]))
        candidates.append(images_root / Path(*parts[1:]).name)

    # 4) if relpath contains leading "images/...", try stripping "images"
    if len(parts) >= 2 and parts[0] == "images":
        candidates.append(images_root / Path(*parts[1:]))
        candidates.append(images_root / Path(*parts[1:]).name)

    for p in candidates:
        if p.exists():
            return p

    # fallback for error message
    return candidates[0]


class CifarJsonlDataset(Dataset):
    def __init__(
        self,
        images_root: str,
        index_jsonl: str,
        split: str,
        transform: Optional[T.Compose] = None,
        filter_noisy: Optional[bool] = None,         # None=use all; True=keep only noisy; False=keep only clean
        exclude_golden_gold_jsonl: Optional[str] = None,  # path to Golden_Gold_Standard_seed42.jsonl
    ):
        self.images_root = Path(images_root)
        self.index_jsonl = Path(index_jsonl)
        self.split = split
        self.transform = transform

        self.samples: List[JsonlSample] = _read_index_map_jsonl(str(self.index_jsonl))

        # Exclude golden_gold from training (raw_id matches sample.id)
        if exclude_golden_gold_jsonl is not None:
            gg_ids = _load_golden_gold_raw_ids(exclude_golden_gold_jsonl)
            self.samples = [s for s in self.samples if s.id not in gg_ids]

        # Optional noisy filter (baseline default uses all)
        if filter_noisy is not None:
            kept: List[JsonlSample] = []
            for s in self.samples:
                is_noisy = bool(s.meta.get("is_noisy", False))
                if filter_noisy and is_noisy:
                    kept.append(s)
                if (not filter_noisy) and (not is_noisy):
                    kept.append(s)
            self.samples = kept

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = _resolve_image_path(self.images_root, s.relpath)

        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found. Tried: {img_path} "
                f"(images_root={self.images_root}, relpath={s.relpath})."
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        # Training uses label only. class_name is just for human readability.
        y = s.label
        return img, y


def build_transforms(train: bool):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])