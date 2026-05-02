from __future__ import annotations

OBJECT_CENTRIC_METHODS = ["spar3d", "sf3d", "trellis", "hunyuan3d", "zeroshape"]


def make_object_centric_command_template(method_name: str) -> str:
    """生成 object-centric baseline 命令模板。

    这些只是模板，用户后续需要按实际外部 repo 修改参数和输出协议。
    """
    name = method_name.lower()
    return f"python external_methods/{name}/run.py --image {{image}} --crop-dir {{crop_dir}} --masks {{masks}} --out {{output_dir}}"
