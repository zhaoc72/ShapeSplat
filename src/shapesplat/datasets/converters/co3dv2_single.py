from __future__ import annotations

import gzip
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from shapesplat.datasets.benchmark.manifest_v2 import BenchmarkRecord, save_benchmark_manifest
from shapesplat.datasets.benchmark.splits import create_split_file
from shapesplat.datasets.benchmark.builder_v2 import write_benchmark_info
from shapesplat.datasets.converters.base import DatasetConverter


IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".npy", ".npz"}
DEPTH_EXTS = {".png", ".jpg", ".jpeg", ".npy", ".npz"}


@dataclass
class CO3DFrame:
    category: str
    sequence_name: str
    frame_index: int
    image_path: Path
    mask_path: Path
    depth_path: Path | None = None
    depth_mask_path: Path | None = None
    pointcloud_path: Path | None = None
    camera: dict | None = None
    split: str = "test"
    annotation: dict | None = None


def read_jgz_json(path: str | Path) -> Any:
    """读取 CO3D 的 gzip json / jgz annotation。

    中文注释：不依赖 facebookresearch/co3d package，直接读取 gzip/json。
    """
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return json.load(f)


def find_co3d_categories(root: str | Path) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"set_lists", "eval_batches"})


def find_co3d_sequences(category_dir: str | Path) -> list[Path]:
    category_dir = Path(category_dir)
    skip = {"set_lists", "eval_batches", "__pycache__"}
    return sorted(p for p in category_dir.iterdir() if p.is_dir() and p.name not in skip)


def _first_existing(base: Path, names: list[str]) -> Path | None:
    for name in names:
        p = base / name
        if p.exists():
            return p
    return None


def _stem_map(folder: Path | None, exts: set[str]) -> dict[str, Path]:
    if folder is None or not folder.exists():
        return {}
    return {p.stem: p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts}


def find_frame_files(sequence_dir: str | Path) -> list[CO3DFrame]:
    """扫描 sequence/images 和 masks，返回逐帧匹配。

    中文注释：这是 annotation 解析失败时的兜底路径扫描；CO3Dv2 single 通常是单主体 visible foreground mask。
    """
    seq = Path(sequence_dir)
    category = seq.parent.name
    sequence_name = seq.name
    images_dir = _first_existing(seq, ["images", "image", "rgb"])
    masks_dir = _first_existing(seq, ["masks", "mask", "foreground_masks"])
    depths_dir = _first_existing(seq, ["depths", "depth"])
    depth_masks_dir = _first_existing(seq, ["depth_masks", "depth_mask"])
    if images_dir is None or masks_dir is None:
        return []
    images = _stem_map(images_dir, IMAGE_EXTS)
    masks = _stem_map(masks_dir, MASK_EXTS)
    depths = _stem_map(depths_dir, DEPTH_EXTS)
    depth_masks = _stem_map(depth_masks_dir, MASK_EXTS)
    pointcloud = seq / "pointcloud.ply" if (seq / "pointcloud.ply").exists() else None
    frames = []
    for idx, (stem, image) in enumerate(images.items()):
        mask = masks.get(stem)
        if mask is None:
            continue
        frames.append(
            CO3DFrame(
                category=category,
                sequence_name=sequence_name,
                frame_index=idx,
                image_path=image,
                mask_path=mask,
                depth_path=depths.get(stem),
                depth_mask_path=depth_masks.get(stem),
                pointcloud_path=pointcloud,
            )
        )
    return frames


def convert_mask_to_stack(mask_path: str | Path, out_path: str | Path) -> Path:
    """把 CO3D foreground mask 转成 [1,H,W] npy。

    中文注释：这里保存 retained visible foreground mask，不是 amodal mask。
    """
    mask_path = Path(mask_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mask_path.suffix.lower() == ".npy":
        arr = np.load(mask_path)
    elif mask_path.suffix.lower() == ".npz":
        data = np.load(mask_path)
        key = "mask" if "mask" in data else list(data.keys())[0]
        arr = data[key]
    else:
        img = Image.open(mask_path)
        arr = np.asarray(img.convert("L"))
    arr = np.asarray(arr)
    if arr.ndim == 3:
        arr = arr[..., 0]
    mask = (arr.astype("float32") > 0).astype("float32")
    np.save(out_path, mask[None])
    return out_path


def parse_camera_from_frame_annotation(frame_ann: dict) -> dict | None:
    """尽量从 CO3D frame annotation 中提取相机字段，失败返回 None。"""
    if not isinstance(frame_ann, dict):
        return None
    cam = frame_ann.get("viewpoint") or frame_ann.get("camera") or frame_ann.get("camera_params") or {}
    if not isinstance(cam, dict):
        return None
    out = {}
    for key in ["focal_length", "principal_point", "image_size", "R", "T", "intrinsics", "extrinsics"]:
        if key in cam:
            out[key] = cam[key]
    return out or None


def save_camera_json(camera_dict: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(camera_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def copy_or_link_file(src: str | Path, dst: str | Path, copy_files: bool = True, overwrite: bool = False) -> str:
    """复制文件或返回原路径。

    中文注释：Windows 下 symlink 权限不稳定；copy_files=false 时 manifest 直接引用原始绝对路径。
    """
    src = Path(src)
    dst = Path(dst)
    if not copy_files:
        return str(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return str(dst)
    shutil.copy2(src, dst)
    return str(dst)


def _rel(path: Path | str | None, root: Path) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(root)).replace("\\", "/")
    except Exception:
        return str(p)


def _ann_path(root: Path, value: Any) -> Path | None:
    if not value:
        return None
    p = Path(str(value))
    return p if p.is_absolute() else root / p


def _load_annotation_frames(root: Path, category_dir: Path, default_split: str) -> list[CO3DFrame]:
    """annotation-driven 尝试；失败会由调用方 fallback 到 folder scan。"""
    ann_path = category_dir / "frame_annotations.jgz"
    if not ann_path.exists():
        return []
    try:
        data = read_jgz_json(ann_path)
    except Exception:
        return []
    rows = data if isinstance(data, list) else data.get("annotations", []) if isinstance(data, dict) else []
    frames: list[CO3DFrame] = []
    for i, ann in enumerate(rows):
        if not isinstance(ann, dict):
            continue
        seq_name = str(ann.get("sequence_name") or ann.get("sequence") or ann.get("sequence_id") or "sequence")
        image = _ann_path(root, ann.get("image") or ann.get("image_path") or ann.get("image_file"))
        mask = _ann_path(root, ann.get("mask") or ann.get("mask_path") or ann.get("foreground_mask"))
        if image is None or mask is None or not image.exists() or not mask.exists():
            continue
        depth = _ann_path(root, ann.get("depth") or ann.get("depth_path"))
        depth_mask = _ann_path(root, ann.get("depth_mask") or ann.get("depth_mask_path"))
        seq_dir = category_dir / seq_name
        pc = seq_dir / "pointcloud.ply" if (seq_dir / "pointcloud.ply").exists() else None
        frames.append(
            CO3DFrame(
                category=category_dir.name,
                sequence_name=seq_name,
                frame_index=int(ann.get("frame_number") or ann.get("frame_index") or i),
                image_path=image,
                mask_path=mask,
                depth_path=depth if depth and depth.exists() else None,
                depth_mask_path=depth_mask if depth_mask and depth_mask.exists() else None,
                pointcloud_path=pc,
                camera=parse_camera_from_frame_annotation(ann),
                split=str(ann.get("split") or default_split),
                annotation=ann,
            )
        )
    return frames


def inspect_co3dv2_single(root: str | Path, max_categories: int = 5, max_sequences: int = 5) -> dict:
    """转换前结构检查。"""
    root = Path(root)
    categories = find_co3d_categories(root) if root.exists() else []
    rows = []
    for cat in categories[: max(0, int(max_categories))]:
        seqs = find_co3d_sequences(cat)
        sample = []
        for seq in seqs[: max(0, int(max_sequences))]:
            sample.append(
                {
                    "sequence": seq.name,
                    "num_images": len(_stem_map(_first_existing(seq, ["images", "image", "rgb"]), IMAGE_EXTS)),
                    "num_masks": len(_stem_map(_first_existing(seq, ["masks", "mask", "foreground_masks"]), MASK_EXTS)),
                    "num_depths": len(_stem_map(_first_existing(seq, ["depths", "depth"]), DEPTH_EXTS)),
                    "num_depth_masks": len(_stem_map(_first_existing(seq, ["depth_masks", "depth_mask"]), MASK_EXTS)),
                    "has_pointcloud_ply": bool((seq / "pointcloud.ply").exists()),
                }
            )
        rows.append(
            {
                "category": cat.name,
                "num_sequences": len(seqs),
                "has_frame_annotations": bool((cat / "frame_annotations.jgz").exists()),
                "has_sequence_annotations": bool((cat / "sequence_annotations.jgz").exists()),
                "has_set_lists": bool((cat / "set_lists").exists()),
                "has_eval_batches": bool((cat / "eval_batches").exists()),
                "sample_sequences": sample,
            }
        )
    return {"root": str(root), "exists": root.exists(), "num_categories": len(categories), "categories": rows}


class CO3Dv2SingleConverter(DatasetConverter):
    name = "co3dv2_single"

    def convert(self, src: str | Path, out: str | Path, cfg: dict | None = None) -> Path:
        """转换 CO3Dv2 single subset 到 benchmark v2。

        中文注释：CO3Dv2 single 用作 real-image diagnostic，通常是一张图一个 foreground object，
        不是多物体遮挡主 benchmark。
        """
        cfg = cfg or {}
        root = Path(src)
        out = Path(out)
        overwrite = bool(cfg.get("overwrite", False))
        copy_files = bool(cfg.get("copy_files", True))
        if out.exists() and overwrite:
            shutil.rmtree(out)
        for name in ["images", "masks", "depths", "depth_masks", "cameras", "pointclouds", "metadata"]:
            (out / name).mkdir(parents=True, exist_ok=True)

        records: list[BenchmarkRecord] = []
        categories = find_co3d_categories(root)
        selected_categories = set(cfg.get("categories") or [])
        if selected_categories:
            categories = [c for c in categories if c.name in selected_categories]
        max_categories = cfg.get("max_categories")
        if max_categories is not None:
            categories = categories[: int(max_categories)]

        for cat in categories:
            frames = _load_annotation_frames(root, cat, cfg.get("split", "test"))
            if not frames:
                for seq in find_co3d_sequences(cat):
                    if cfg.get("sequences") and seq.name not in set(cfg.get("sequences") or []):
                        continue
                    frames.extend(find_frame_files(seq))
            by_seq: dict[str, list[CO3DFrame]] = {}
            for fr in frames:
                by_seq.setdefault(fr.sequence_name, []).append(fr)
            seq_items = list(by_seq.items())
            max_sequences = cfg.get("max_sequences")
            if max_sequences is not None:
                seq_items = seq_items[: int(max_sequences)]
            for sequence_name, seq_frames in seq_items:
                stride = max(1, int(cfg.get("frame_stride", 1)))
                seq_frames = sorted(seq_frames, key=lambda x: x.frame_index)[::stride]
                max_frames = cfg.get("max_frames_per_sequence")
                if max_frames is not None:
                    seq_frames = seq_frames[: int(max_frames)]
                pc_out_rel = None
                for fr in seq_frames:
                    image_id = f"{fr.category}_{fr.sequence_name}_{fr.frame_index:06d}".replace(" ", "_")
                    image_dst = out / "images" / f"{image_id}{fr.image_path.suffix.lower()}"
                    image_path = copy_or_link_file(fr.image_path, image_dst, copy_files, overwrite)
                    mask_dst = out / "masks" / f"{image_id}.npy"
                    convert_mask_to_stack(fr.mask_path, mask_dst)
                    depth_rel = None
                    depth_mask_rel = None
                    if fr.depth_path and fr.depth_path.exists():
                        depth_dst = out / "depths" / f"{image_id}{fr.depth_path.suffix.lower()}"
                        depth_rel = _rel(copy_or_link_file(fr.depth_path, depth_dst, copy_files, overwrite), out)
                    if fr.depth_mask_path and fr.depth_mask_path.exists():
                        dm_dst = out / "depth_masks" / f"{image_id}{fr.depth_mask_path.suffix.lower()}"
                        depth_mask_rel = _rel(copy_or_link_file(fr.depth_mask_path, dm_dst, copy_files, overwrite), out)
                    camera_rel = None
                    if fr.camera:
                        camera_rel = _rel(save_camera_json(fr.camera, out / "cameras" / f"{image_id}.json"), out)
                    if fr.pointcloud_path and fr.pointcloud_path.exists() and pc_out_rel is None:
                        pc_dst = out / "pointclouds" / f"{fr.category}_{fr.sequence_name}.ply"
                        pc_out_rel = _rel(copy_or_link_file(fr.pointcloud_path, pc_dst, copy_files, overwrite), out)
                    meta = {
                        "image_id": image_id,
                        "category": fr.category,
                        "sequence_name": fr.sequence_name,
                        "frame_index": fr.frame_index,
                        "source_image_path": str(fr.image_path),
                        "source_mask_path": str(fr.mask_path),
                        "source_depth_path": str(fr.depth_path) if fr.depth_path else None,
                        "source_pointcloud_path": str(fr.pointcloud_path) if fr.pointcloud_path else None,
                        "is_co3dv2_single": True,
                        "notes": "CO3Dv2 single is object-centric real-image diagnostic; mask is visible foreground.",
                    }
                    meta_path = out / "metadata" / f"{image_id}.json"
                    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
                    records.append(
                        BenchmarkRecord(
                            image_id=image_id,
                            image_path=_rel(Path(image_path), out) or str(image_path),
                            mask_path=_rel(mask_dst, out) or str(mask_dst),
                            split=fr.split or cfg.get("split", "test"),
                            metadata_path=_rel(meta_path, out),
                            subset="co3dv2_single",
                            category=fr.category,
                            num_objects=1,
                            scene_id=f"{fr.category}_{fr.sequence_name}",
                            source_dataset="co3dv2_single",
                            depth_path=depth_rel,
                            camera_path=camera_rel,
                            gt_pointcloud_path=pc_out_rel,
                            diagnostics={"real_image": "true"},
                            extra={"depth_mask_path": depth_mask_rel or ""},
                        )
                    )

        manifest = out / "manifest.csv"
        save_benchmark_manifest(records, manifest)
        create_split_file(records, out / "splits.json")
        write_benchmark_info(
            out,
            {
                "name": out.name,
                "source_dataset": "co3dv2_single",
                "num_images": len(records),
                "created_at": datetime.now().isoformat(),
                "notes": "CO3Dv2 single subset converted as real-image single-object visible-mask diagnostic.",
            },
        )
        return manifest
