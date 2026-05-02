from __future__ import annotations

# 主实验必需字段：same-mask benchmark 至少要知道 image/mask/split。
REQUIRED_COLUMNS = ["image_id", "image_path", "mask_path", "split"]

# 推荐字段：用于论文表格分组、subset 分析和来源追踪。
RECOMMENDED_COLUMNS = ["metadata_path", "subset", "category", "num_objects", "scene_id", "source_dataset"]

# optional GT 字段：只在存在真实几何/深度/相机时使用，缺失不能阻塞普通 same-mask 实验。
OPTIONAL_GT_COLUMNS = [
    "depth_path",
    "camera_path",
    "gt_pointcloud_path",
    "gt_mesh_path",
    "visible_pointcloud_path",
    "hidden_pointcloud_path",
]

# cache 字段：将真实 SAM/DINO/Depth 前端输出绑定到固定 manifest。
OPTIONAL_CACHE_COLUMNS = ["frontend_cache_dir", "frontend_cache_status"]

# 诊断字段：用于 occlusion / same-category / real-image diagnostics 分组。
DIAGNOSTIC_COLUMNS = [
    "occlusion_level",
    "same_category",
    "truncation",
    "contact_heavy",
    "small_object",
    "real_image",
]


def get_all_known_columns() -> list[str]:
    """返回 benchmark v2 已知列，额外列仍允许保存在 record.extra。"""

    cols = REQUIRED_COLUMNS + RECOMMENDED_COLUMNS + OPTIONAL_GT_COLUMNS + OPTIONAL_CACHE_COLUMNS + DIAGNOSTIC_COLUMNS
    return list(dict.fromkeys(cols))


def is_required_column(name: str) -> bool:
    return name in REQUIRED_COLUMNS


def is_optional_gt_column(name: str) -> bool:
    return name in OPTIONAL_GT_COLUMNS

