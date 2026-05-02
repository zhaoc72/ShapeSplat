from __future__ import annotations

import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from shapesplat.benchmarks.stress_metadata import ObjectInfo, StressMetadata, save_stress_metadata
from shapesplat.data.image_io import save_tensor_image


DEFAULT_SUBSETS = [
    "normal",
    "light_occlusion",
    "heavy_occlusion",
    "same_category",
    "contact_heavy",
    "truncation",
    "scale_variation",
    "small_object",
    "mixed",
]

COLORS = [
    (226, 80, 80),
    (75, 148, 230),
    (82, 178, 104),
    (238, 181, 72),
    (172, 112, 219),
    (58, 187, 188),
]


def _shape_mask(size: int, category: str, cx: float, cy: float, scale: float, truncated: bool = False) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    r = max(4, int(scale * size * 0.16))
    x0, y0, x1, y1 = int(cx - r), int(cy - r), int(cx + r), int(cy + r)
    if category == "circle":
        draw.ellipse([x0, y0, x1, y1], fill=255)
    elif category == "ellipse":
        draw.ellipse([x0 - r // 2, y0, x1 + r // 2, y1], fill=255)
    elif category == "triangle":
        draw.polygon([(cx, cy - r), (cx - r, cy + r), (cx + r, cy + r)], fill=255)
    elif category == "polygon":
        pts = [(cx - r, cy - r // 2), (cx, cy - r), (cx + r, cy - r // 3), (cx + r // 2, cy + r), (cx - r, cy + r)]
        draw.polygon([(int(x), int(y)) for x, y in pts], fill=255)
    elif category == "rounded_rectangle":
        draw.rounded_rectangle([x0, y0, x1, y1], radius=max(2, r // 4), fill=255)
    else:
        draw.rectangle([x0, y0, x1, y1], fill=255)
    return mask


def _bbox_from_mask(mask: np.ndarray) -> list[int]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return [0, 0, 0, 0]
    return [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]


def _label_png(masks: np.ndarray) -> Image.Image:
    label = np.zeros(masks.shape[1:], dtype=np.uint8)
    for i, m in enumerate(masks, start=1):
        label[m > 0] = i
    return Image.fromarray(label, mode="L")


def _specs_for_subset(subset: str, size: int, rng: random.Random, num_objects: int) -> tuple[list[dict], list[list[int]], list[list[int]], list[list[int]]]:
    cats = ["circle", "rectangle", "triangle", "ellipse", "rounded_rectangle", "polygon"]
    specs: list[dict] = []
    occ_pairs: list[list[int]] = []
    same_pairs: list[list[int]] = []
    depth_pairs: list[list[int]] = []

    base_scales = [0.95, 0.9, 0.85, 0.8][:num_objects]
    positions = [(0.28, 0.35), (0.62, 0.45), (0.42, 0.68), (0.72, 0.72)]

    if subset in {"light_occlusion", "heavy_occlusion"}:
        shift = 0.12 if subset == "light_occlusion" else 0.04
        positions = [(0.42, 0.48), (0.42 + shift, 0.50), (0.70, 0.68), (0.25, 0.72)]
        occ_pairs.append([1, 0])
        depth_pairs.append([1, 0])
    elif subset == "same_category":
        cats = ["circle", "circle", "rectangle", "triangle"]
        same_pairs.append([0, 1])
        positions = [(0.32, 0.45), (0.62, 0.47), (0.46, 0.72), (0.75, 0.68)]
    elif subset == "contact_heavy":
        positions = [(0.36, 0.50), (0.52, 0.50), (0.66, 0.51), (0.48, 0.68)]
    elif subset == "truncation":
        positions = [(-0.02, 0.35), (0.62, 0.48), (0.38, 0.72), (0.82, 0.72)]
    elif subset == "scale_variation":
        base_scales = [1.35, 0.45, 0.85, 0.55][:num_objects]
        positions = [(0.34, 0.52), (0.76, 0.30), (0.66, 0.70), (0.20, 0.25)]
    elif subset == "small_object":
        base_scales = [0.95, 0.35, 0.75, 0.30][:num_objects]
        positions = [(0.35, 0.52), (0.72, 0.34), (0.62, 0.72), (0.20, 0.24)]
    elif subset == "mixed":
        positions = [(0.38, 0.45), (0.44, 0.48), (0.78, 0.20), (-0.02, 0.72)]
        base_scales = [1.1, 0.8, 0.35, 0.9][:num_objects]
        cats = ["circle", "circle", "triangle", "rectangle"]
        occ_pairs.append([1, 0])
        depth_pairs.append([1, 0])
        same_pairs.append([0, 1])

    for i in range(num_objects):
        px, py = positions[i % len(positions)]
        jitter = 0.03 if subset == "normal" else 0.015
        cx = (px + rng.uniform(-jitter, jitter)) * size
        cy = (py + rng.uniform(-jitter, jitter)) * size
        specs.append(
            {
                "object_id": i,
                "category": cats[i % len(cats)],
                "cx": cx,
                "cy": cy,
                "scale": base_scales[i % len(base_scales)],
                "color": COLORS[i % len(COLORS)],
            }
        )
    return specs, occ_pairs, depth_pairs, same_pairs


def generate_stress_sample(
    image_id: str,
    subset: str,
    size: int = 128,
    seed: int = 0,
    num_objects: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, StressMetadata]:
    """生成一个可控 stress 样本。

    masks 是遮挡后的 visible masks，不是 amodal masks；该数据只用于诊断 ownership/edit failure。
    """

    rng = random.Random(seed)
    if num_objects is None:
        num_objects = 3 if subset != "normal" else rng.randint(2, 4)
    num_objects = max(2, int(num_objects))

    bg = (242, 241, 236)
    image = Image.new("RGB", (size, size), bg)
    specs, occ_pairs, depth_pairs, same_pairs = _specs_for_subset(subset, size, rng, num_objects)

    full_masks: list[np.ndarray] = []
    for spec in specs:
        mask_img = _shape_mask(size, spec["category"], spec["cx"], spec["cy"], spec["scale"], subset == "truncation")
        full_masks.append((np.array(mask_img) > 0).astype(np.uint8))

    draw = ImageDraw.Draw(image)
    for spec, full in zip(specs, full_masks):
        overlay = Image.new("RGBA", (size, size), (*spec["color"], 255))
        rgba = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        rgba.paste(overlay, mask=Image.fromarray((full * 255).astype(np.uint8)))
        image = Image.alpha_composite(image.convert("RGBA"), rgba).convert("RGB")
        draw = ImageDraw.Draw(image)

    # 根据绘制顺序反向扣掉前景遮挡，得到 retained visible masks。
    visible_masks: list[np.ndarray] = []
    front_union = np.zeros((size, size), dtype=bool)
    for full in reversed(full_masks):
        visible = (full > 0) & (~front_union)
        visible_masks.append(visible.astype(np.uint8))
        front_union |= full > 0
    visible_masks = list(reversed(visible_masks))

    object_infos: list[ObjectInfo] = []
    for spec, full, visible in zip(specs, full_masks, visible_masks):
        bbox = _bbox_from_mask(visible)
        is_trunc = bool(full.sum() > 0 and (full[0].any() or full[-1].any() or full[:, 0].any() or full[:, -1].any()))
        object_infos.append(
            ObjectInfo(
                object_id=int(spec["object_id"]),
                category=str(spec["category"]),
                color=[int(c) for c in spec["color"]],
                bbox_xyxy=bbox,
                visible_area=int(visible.sum()),
                full_area_approx=int(full.sum()),
                is_truncated=is_trunc,
                scale=float(spec["scale"]),
            )
        )

    masks_np = np.stack(visible_masks, axis=0).astype(np.uint8)
    scales = [info.scale for info in object_infos]
    scale_ratios = {"max_over_min": float(max(scales) / max(1e-6, min(scales)))}
    meta = StressMetadata(
        image_id=image_id,
        subset=subset,
        num_objects=len(object_infos),
        object_infos=object_infos,
        occlusion_pairs=occ_pairs,
        depth_order_pairs=depth_pairs,
        same_category_pairs=same_pairs,
        truncation_flags={str(info.object_id): info.is_truncated for info in object_infos},
        scale_ratios=scale_ratios,
        generation_seed=int(seed),
    )
    image_t = torch.from_numpy(np.asarray(image).astype("float32") / 255.0).permute(2, 0, 1).contiguous()
    masks_t = torch.from_numpy(masks_np.astype("float32"))
    return image_t, masks_t, meta


def create_stress_dataset(
    out_dir: str | Path,
    num_per_subset: int = 4,
    size: int = 128,
    subsets: list[str] | None = None,
    seed: int = 123,
) -> Path:
    """创建 synthetic stress dataset；这是 smoke-test benchmark，不是真实数据集。"""

    out = Path(out_dir)
    image_dir, mask_dir, mask_png_dir, meta_dir = out / "images", out / "masks", out / "mask_png", out / "metadata"
    for d in (image_dir, mask_dir, mask_png_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)
    subsets = subsets or DEFAULT_SUBSETS
    rows = []
    for subset_idx, subset in enumerate(subsets):
        for i in range(int(num_per_subset)):
            image_id = f"{subset}_{i:03d}"
            sample_seed = int(seed) + subset_idx * 1000 + i
            image, masks, meta = generate_stress_sample(image_id, subset, size=size, seed=sample_seed)
            image_path = image_dir / f"{image_id}.png"
            mask_path = mask_dir / f"{image_id}.npy"
            mask_png_path = mask_png_dir / f"{image_id}.png"
            meta_path = meta_dir / f"{image_id}.json"
            save_tensor_image(image, image_path)
            np.save(mask_path, masks.numpy().astype("uint8"))
            _label_png(masks.numpy().astype("uint8")).save(mask_png_path)
            save_stress_metadata(meta, meta_path)
            rows.append(
                {
                    "image_id": image_id,
                    "image_path": image_path.relative_to(out).as_posix(),
                    "mask_path": mask_path.relative_to(out).as_posix(),
                    "metadata_path": meta_path.relative_to(out).as_posix(),
                    "split": "test",
                    "subset": subset,
                    "num_objects": str(meta.num_objects),
                }
            )
    manifest = out / "manifest.csv"
    with open(manifest, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_id", "image_path", "mask_path", "metadata_path", "split", "subset", "num_objects"])
        writer.writeheader()
        writer.writerows(rows)
    return manifest

