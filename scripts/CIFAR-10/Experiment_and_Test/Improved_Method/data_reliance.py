import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Any, Set, Tuple

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


# -------------------------
# JSONL helpers
# -------------------------
def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_reliance_map(reliance_jsonl: str) -> Dict[int, float]:
    """
    Your reliance jsonl row example:
    {"id": 0, ..., "data_reliance": 0.95, ...}
    Returns: id -> data_reliance
    """
    mp: Dict[int, float] = {}
    for obj in read_jsonl(reliance_jsonl):
        if "id" in obj and "data_reliance" in obj:
            mp[int(obj["id"])] = float(obj["data_reliance"])
    return mp


def load_golden_gold_raw_ids(golden_gold_jsonl: str) -> Set[int]:
    """
    golden gold row example:
    {"gg_id": 0, "raw_id": 5736, ...}
    Returns: set(raw_id)
    """
    s: Set[int] = set()
    for obj in read_jsonl(golden_gold_jsonl):
        s.add(int(obj["raw_id"]))
    return s


def resolve_image_path(images_root: Path, relpath: str) -> Path:
    """
    Handles relpath variants:
      - "images/000000.png"
      - "train_images/000000.png"
      - "test_images/000000.png"
      - "000000.png"
    images_root points to .../train_images or .../test_images
    """
    rel = Path(relpath)
    candidates: List[Path] = []

    candidates.append(images_root / rel)
    candidates.append(images_root / rel.name)

    parts = rel.parts
    if len(parts) >= 2 and parts[0] in ("train_images", "test_images"):
        tail = Path(*parts[1:])
        candidates.append(images_root / tail)
        candidates.append(images_root / tail.name)

    if len(parts) >= 2 and parts[0] == "images":
        tail = Path(*parts[1:])
        candidates.append(images_root / tail)
        candidates.append(images_root / tail.name)

    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


# -------------------------
# Transforms
# -------------------------
def build_transforms(train: bool):
    if train:
        return T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )


# -------------------------
# Dataset
# -------------------------
@dataclass
class IndexSample:
    id: int
    relpath: str
    label: int
    class_name: str
    meta: Dict[str, Any]


class CifarJsonlRelianceDataset(Dataset):
    """
    Returns: (image_tensor, label_int, data_reliance_float)
    - data_reliance comes from:
      1) index_jsonl field "data_reliance" if exists
      2) else reliance_jsonl mapping (id -> data_reliance)
      3) else default 1.0 (if require_reliance=False)
    """

    def __init__(
        self,
        images_root: str,
        index_jsonl: str,
        split: str,
        transform=None,
        *,
        reliance_jsonl: Optional[str] = None,
        require_reliance: bool = False,
        exclude_golden_gold_jsonl: Optional[str] = None,  # train only
        filter_noisy: Optional[str] = None,  # "all"|"clean"|"noisy"
    ):
        self.images_root = Path(images_root)
        self.index_jsonl = index_jsonl
        self.split = split
        self.transform = transform
        self.require_reliance = require_reliance

        self.reliance_map: Dict[int, float] = load_reliance_map(reliance_jsonl) if reliance_jsonl else {}

        rows = read_jsonl(index_jsonl)
        samples: List[IndexSample] = []
        for obj in rows:
            samples.append(
                IndexSample(
                    id=int(obj["id"]),
                    relpath=str(obj["relpath"]),
                    label=int(obj["label"]),
                    class_name=str(obj.get("class_name", "")),
                    meta=obj,
                )
            )

        # Exclude golden_gold from TRAIN
        if exclude_golden_gold_jsonl is not None:
            gg_ids = load_golden_gold_raw_ids(exclude_golden_gold_jsonl)
            samples = [s for s in samples if s.id not in gg_ids]

        # Optional noisy filtering
        if filter_noisy is not None and filter_noisy != "all":
            kept: List[IndexSample] = []
            want_noisy = (filter_noisy == "noisy")
            for s in samples:
                is_noisy = bool(s.meta.get("is_noisy", False))
                if want_noisy and is_noisy:
                    kept.append(s)
                if (not want_noisy) and (not is_noisy):
                    kept.append(s)
            samples = kept

        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def _get_reliance(self, sample: IndexSample) -> float:
        r = sample.meta.get("data_reliance", None)
        if r is None:
            r = self.reliance_map.get(sample.id, None)

        if r is None:
            if self.require_reliance:
                raise ValueError(
                    f"Missing data_reliance for id={sample.id}. "
                    f"Provide reliance_jsonl or store data_reliance in index jsonl."
                )
            return 1.0
        return float(r)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img_path = resolve_image_path(self.images_root, s.relpath)
        if not img_path.exists():
            raise FileNotFoundError(
                f"Image not found: {img_path} (images_root={self.images_root}, relpath={s.relpath})"
            )

        img = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        y = s.label
        r = self._get_reliance(s)
        return img, y, r