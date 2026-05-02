from __future__ import annotations

from pathlib import Path


TEMPLATE = """# 本地真实组件接入模板。
# 用户可填入 checkpoint / model_name / renderer module；不填时 auto 会 fallback 到 stub/soft。
device: auto
image:
  size: 128
  input_path: null

frontend:
  mask_source: sam
  sam_backend: auto
  sam3_checkpoint: null
  sam3_model_type: null
  sam3_device: auto
  sam3_prompt_mode: automatic
  sam3_text_prompts:
    - object
  sam3_score_threshold: 0.5
  sam3_max_masks: 8

  dino_backend: auto
  dino_model_name: null
  dino_checkpoint: null
  dino_device: auto
  dino_feature_layer: last
  dino_input_size: null
  dino_l2_normalize: true

  depth_backend: auto
  depth_model_name: null
  depth_checkpoint: null
  depth_device: auto
  depth_input_size: null
  depth_inverse: false
  depth_normalize: true
  depth_normalize_on_foreground: true

shape_bank:
  backend: auto
  root: null
  fallback_to_toy: true

renderer:
  backend: auto
  real_renderer_module: null
  real_renderer_class: null
  fallback_to_soft: true

external_baselines:
  config_path: configs/external_baselines.yaml
"""


def create_local_backend_template(out_path: str | Path) -> Path:
    """生成本地 backend 配置模板。

    该模板默认安全 fallback，不会强制要求真实模型或 CUDA renderer。
    """

    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(TEMPLATE, encoding="utf-8")
    return path
