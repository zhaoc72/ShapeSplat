from __future__ import annotations

from pathlib import Path

from shapesplat.cache.frontend_cache import load_frontend_cache_manifest


def apply_frontend_cache_config(cfg: dict, use_cache=False, cache_root=None, cache_manifest=None, save_cache=False, cache_out=None) -> None:
    """把 CLI cache 参数写入 cfg；默认不改变旧行为。"""

    ccfg = cfg.setdefault("frontend_cache", {})
    if use_cache:
        ccfg["use_cache"] = True
    if save_cache:
        ccfg["save_cache"] = True
    if cache_root:
        ccfg["cache_root"] = str(cache_root)
    if cache_manifest:
        ccfg["cache_manifest"] = str(cache_manifest)
    if cache_out:
        ccfg["cache_root"] = str(cache_out)


def attach_cache_to_dataset(dataset, cache_manifest: str | Path | None = None, cache_root: str | Path | None = None) -> None:
    """给 dataset.records 填充 frontend_cache_dir。

    后续 build_frontend 会从 record.metadata["frontend_cache_dir"] 自动读取 cache。
    """

    mapping = {}
    if cache_manifest:
        mapping = {k: v.cache_dir for k, v in load_frontend_cache_manifest(cache_manifest).items()}
    records = getattr(dataset, "records", dataset if isinstance(dataset, list) else [])
    for record in records:
        if record.image_id in mapping:
            record.metadata["frontend_cache_dir"] = mapping[record.image_id]
        elif cache_root:
            record.metadata["frontend_cache_dir"] = str(Path(cache_root) / record.image_id)
