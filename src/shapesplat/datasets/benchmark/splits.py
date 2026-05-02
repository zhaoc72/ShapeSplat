from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path


def create_split_file(records, out_path: str | Path) -> None:
    """写固定 splits.json；正式实验应固定 split，避免 retrieval bank 泄漏测试集。"""

    splits = summarize_splits(records)["splits"]
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(splits, indent=2), encoding="utf-8")


def filter_records_by_split(records, split: str | list[str]):
    wanted = {split} if isinstance(split, str) else set(split)
    return [r for r in records if r.split in wanted]


def split_records_random(records, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=123, group_by_scene: bool = True):
    """随机生成 split；group_by_scene=True 时同一 scene_id 不跨 split。"""

    rng = random.Random(seed)
    groups = defaultdict(list)
    if group_by_scene:
        for r in records:
            groups[r.scene_id or r.image_id].append(r)
        items = list(groups.values())
    else:
        items = [[r] for r in records]
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    for idx, group in enumerate(items):
        split = "train" if idx < n_train else ("val" if idx < n_train + n_val else "test")
        for record in group:
            record.split = split
    return records


def summarize_splits(records) -> dict:
    splits = defaultdict(list)
    for r in records:
        splits[r.split or "unknown"].append(r.image_id)
    return {"counts": {k: len(v) for k, v in splits.items()}, "splits": dict(splits)}

