from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .app_config import APP_ROOT, UPSCALE_MODELS
from .gimp_runner import (
    GimpExecutionResult,
    GimpInfo,
    collect_gimp_upscale_plugin_dirs,
    has_gimp2_direct_upscale,
    run_gimp_plugin_direct_upscale,
)
from .process_utils import run_command


@dataclass(slots=True)
class ExternalBackend:
    executable_path: Path
    model_dir: Path
    source_label: str


@dataclass(slots=True)
class UpscalePlan:
    gimp_info: GimpInfo
    model_name: str
    scale: int
    direct_plugin_available: bool
    external_backend: ExternalBackend | None


@dataclass(slots=True)
class UpscaleResult:
    execution_method: str
    output_path: Path
    duration_seconds: float
    stdout: str
    stderr: str
    exit_code: int
    backend_source: str


def _candidate_external_backend_dirs(gimp_info: GimpInfo) -> list[Path]:
    candidates: list[Path] = []
    for plugin_dir in collect_gimp_upscale_plugin_dirs(gimp_info):
        candidates.append(plugin_dir / "resrgan")
    candidates.append(APP_ROOT / "vendor" / "gimp_upscale" / "resrgan")
    return candidates


def _resolve_external_backend(gimp_info: GimpInfo, model_name: str) -> ExternalBackend | None:
    for candidate_dir in _candidate_external_backend_dirs(gimp_info):
        executable = candidate_dir / "realesrgan-ncnn-vulkan.exe"
        model_dir = candidate_dir / "models"
        param_path = model_dir / f"{model_name}.param"
        bin_path = model_dir / f"{model_name}.bin"
        if executable.exists() and model_dir.exists() and param_path.exists() and bin_path.exists():
            source_label = "plugin_resrgan" if "plug-ins" in str(candidate_dir).lower() else "vendored_resrgan"
            return ExternalBackend(
                executable_path=executable,
                model_dir=model_dir,
                source_label=source_label,
            )
    return None


def prepare_upscale_plan(gimp_info: GimpInfo, model_name: str, scale: int) -> UpscalePlan:
    if model_name not in UPSCALE_MODELS:
        raise ValueError(f"未対応のアップスケールモデルです: {model_name}")
    direct_plugin_available = has_gimp2_direct_upscale(gimp_info)
    external_backend = _resolve_external_backend(gimp_info, model_name)
    if not direct_plugin_available and external_backend is None:
        raise RuntimeError(
            "AI Upscale 実行手段を見つけられませんでした。"
            " GIMP 2 の AI Upscale プラグインを入れるか、setup.bat で外部 backend を取得してください。"
        )
    return UpscalePlan(
        gimp_info=gimp_info,
        model_name=model_name,
        scale=int(scale),
        direct_plugin_available=direct_plugin_available,
        external_backend=external_backend,
    )


def _run_external_realesrgan(
    backend: ExternalBackend,
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    scale: int,
    stop_check: Callable[[], bool] | None = None,
) -> UpscaleResult:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(backend.executable_path),
        "-i",
        str(input_path),
        "-o",
        str(output_path),
        "-n",
        model_name,
        "-s",
        str(scale),
        "-m",
        str(backend.model_dir),
    ]
    completed = run_command(command, timeout_seconds=7200, stop_check=stop_check)
    if completed.exit_code != 0 or not output_path.exists():
        raise RuntimeError(
            "external_realesrgan に失敗しました。"
            f" exit_code={completed.exit_code}"
            f" stdout={completed.stdout}"
            f" stderr={completed.stderr}"
        )
    return UpscaleResult(
        execution_method="external_realesrgan",
        output_path=output_path,
        duration_seconds=completed.duration_seconds,
        stdout=completed.stdout,
        stderr=completed.stderr,
        exit_code=completed.exit_code,
        backend_source=backend.source_label,
    )


def run_upscale(
    plan: UpscalePlan,
    *,
    gimp_path: Path,
    input_path: Path,
    output_path: Path,
    stop_check: Callable[[], bool] | None = None,
) -> UpscaleResult:
    errors: list[str] = []
    if plan.direct_plugin_available:
        try:
            result: GimpExecutionResult = run_gimp_plugin_direct_upscale(
                gimp_path=gimp_path,
                input_path=input_path,
                output_path=output_path,
                model_name=plan.model_name,
                scale=plan.scale,
                gimp_info=plan.gimp_info,
                stop_check=stop_check,
            )
            return UpscaleResult(
                execution_method="gimp_plugin_direct",
                output_path=result.output_path or output_path,
                duration_seconds=result.duration_seconds,
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.exit_code,
                backend_source="gimp2_plugin_direct",
            )
        except Exception as exc:
            errors.append(f"gimp_plugin_direct={exc}")

    if plan.external_backend is not None:
        try:
            return _run_external_realesrgan(
                plan.external_backend,
                input_path=input_path,
                output_path=output_path,
                model_name=plan.model_name,
                scale=plan.scale,
                stop_check=stop_check,
            )
        except Exception as exc:
            errors.append(f"external_realesrgan={exc}")

    raise RuntimeError("AI Upscale に失敗しました。 " + " | ".join(errors))
