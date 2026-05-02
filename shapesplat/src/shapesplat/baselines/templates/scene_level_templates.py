from __future__ import annotations

SCENE_LEVEL_METHODS = ["vggt", "dust3r", "mast3r", "anysplat", "mapanything"]


def make_scene_level_command_template(method_name: str) -> str:
    """生成 scene-level baseline 命令模板。

    模板不会执行真实方法；真实接入时仍需用户提供 repo、checkpoint 和协议转换。
    """
    name = method_name.lower()
    return f"python external_methods/{name}/run.py --image {{image}} --masks {{masks}} --out {{output_dir}}"
