from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from shapesplat.config import load_config
from shapesplat.integration.capabilities import detect_all_capabilities
from shapesplat.integration.report import make_capability_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect local backend capabilities without running heavy inference.")
    parser.add_argument("--config", default="configs/local_backend_template.yaml")
    parser.add_argument("--external-config", default=None)
    parser.add_argument("--out", default="outputs/backend_capabilities")
    args = parser.parse_args()
    cfg = load_config(args.config)
    external_cfg = None
    if args.external_config and Path(args.external_config).exists():
        with open(args.external_config, "r", encoding="utf-8") as f:
            external_cfg = yaml.safe_load(f) or {}
    # 只检测配置和 import 能力，不真正跑真实模型。
    caps = detect_all_capabilities(cfg, external_cfg)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "backend_capabilities.json").write_text(json.dumps(caps, indent=2, ensure_ascii=False), encoding="utf-8")
    md = make_capability_markdown(caps)
    (out / "backend_capabilities.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
