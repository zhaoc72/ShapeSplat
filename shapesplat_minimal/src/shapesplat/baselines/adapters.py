from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from shapesplat.baselines.dummy_baselines import identity_mask_baseline, save_baseline_prediction
from shapesplat.baselines.protocol import BaselineInputSpec, BaselineOutputSpec
from shapesplat.baselines.validate_outputs import validate_baseline_output_dir
from shapesplat.data.image_io import load_image


class BaselineAdapter:
    """外部 baseline adapter 基类。

    adapter 的职责是把统一 baseline input protocol 转换为外部方法需要的命令、
    环境和输出格式。基类不绑定任何具体 baseline。
    """

    name: str = "base"

    def prepare_inputs(self, input_spec: BaselineInputSpec, work_dir: Path) -> dict:
        raise NotImplementedError

    def build_command(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict) -> list[str] | str:
        raise NotImplementedError

    def run(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict, dry_run: bool = False) -> BaselineOutputSpec:
        raise NotImplementedError

    def validate_outputs(self, output_dir: Path, input_spec: BaselineInputSpec) -> dict:
        return validate_baseline_output_dir(
            output_dir,
            expected_num_objects=input_spec.num_objects,
            strict=False,
        )


class DummyExternalAdapter(BaselineAdapter):
    """不调用外部程序的 mock adapter。

    它读取 baseline input spec 中的 image/masks，运行 identity_mask_baseline，
    写出标准 baseline output protocol，用于测试 external runner。
    """

    name = "dummy_external"

    def prepare_inputs(self, input_spec: BaselineInputSpec, work_dir: Path) -> dict:
        return {"input_spec": input_spec, "work_dir": str(work_dir)}

    def build_command(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict) -> str:
        return f"dummy_external --image {input_spec.image_path} --masks {input_spec.masks_path} --out {output_dir}"

    def run(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict, dry_run: bool = False) -> BaselineOutputSpec:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        if dry_run:
            return BaselineOutputSpec(
                method_name=self.name,
                image_id=input_spec.image_id,
                output_dir=str(output_dir),
                render_path=None,
                alpha_path=None,
                ownership_path=None,
                metrics_path=None,
                metadata={"dry_run": True, "command": self.build_command(input_spec, output_dir, cfg)},
            )
        image = load_image(input_spec.image_path, size=None)
        masks = np.load(input_spec.masks_path)
        import torch

        masks_t = torch.from_numpy(masks).float()
        pred = identity_mask_baseline(image, masks_t)
        return save_baseline_prediction(pred, output_dir, self.name, input_spec.image_id, metrics={})


class CommandTemplateAdapter(BaselineAdapter):
    """通用 shell command template adapter。

    该 adapter 只负责占位符替换、dry-run、subprocess 执行和日志保存；它不代表
    任何具体外部 baseline，也不保证外部程序存在。
    """

    name = "command_template"

    def prepare_inputs(self, input_spec: BaselineInputSpec, work_dir: Path) -> dict:
        return {
            "image": input_spec.image_path,
            "masks": input_spec.masks_path,
            "input_dir": input_spec.output_dir,
            "crop_dir": input_spec.crop_dir,
            "metadata": input_spec.metadata_path,
        }

    def build_command(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict) -> list[str] | str:
        template = cfg.get("command")
        if not template:
            raise ValueError("CommandTemplateAdapter requires cfg['command'].")
        values = {
            "image": str(Path(input_spec.image_path)),
            "masks": str(Path(input_spec.masks_path)),
            "input_dir": str(Path(input_spec.output_dir)),
            "output_dir": str(Path(output_dir)),
            "crop_dir": str(Path(input_spec.crop_dir)),
            "metadata": str(Path(input_spec.metadata_path)),
            "method_name": cfg.get("name", self.name),
        }
        return str(template).format(**values)

    def run(self, input_spec: BaselineInputSpec, output_dir: Path, cfg: dict, dry_run: bool = False) -> BaselineOutputSpec:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        command = self.build_command(input_spec, output_dir, cfg)
        logs = output_dir / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        (logs / "command.txt").write_text(str(command), encoding="utf-8")
        if dry_run:
            print(f"[dry-run] {command}")
            return BaselineOutputSpec(
                method_name=cfg.get("name", self.name),
                image_id=input_spec.image_id,
                output_dir=str(output_dir),
                render_path=None,
                alpha_path=None,
                ownership_path=None,
                metrics_path=None,
                metadata={"dry_run": True, "command": command},
            )
        result = subprocess.run(
            command,
            shell=isinstance(command, str),
            text=True,
            capture_output=True,
            timeout=int(cfg.get("timeout_sec", 3600)),
        )
        (logs / "stdout.txt").write_text(result.stdout or "", encoding="utf-8")
        (logs / "stderr.txt").write_text(result.stderr or "", encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"External command failed with code {result.returncode}. See {logs}")
        validation = self.validate_outputs(output_dir, input_spec)
        if not validation.get("valid"):
            raise RuntimeError(f"External command finished but output is invalid: {validation.get('errors')}")
        return BaselineOutputSpec(
            method_name=cfg.get("name", self.name),
            image_id=input_spec.image_id,
            output_dir=str(output_dir),
            render_path=str(output_dir / "render.png") if (output_dir / "render.png").exists() else None,
            alpha_path=str(output_dir / "alpha.png") if (output_dir / "alpha.png").exists() else None,
            ownership_path=str(output_dir / "ownership.npy") if (output_dir / "ownership.npy").exists() else None,
            metrics_path=str(output_dir / "metrics.json") if (output_dir / "metrics.json").exists() else None,
            metadata={"command": command},
        )

