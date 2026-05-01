from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .process_utils import CommandResult, run_command


NOISE_PRESET_ORDER = ("off", "weak", "medium", "strong")
UNSHARP_PRESET_ORDER = ("off", "weak", "medium", "strong")

COMMON_GIMP_PATHS = (
    Path(r"C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe"),
    Path(r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console-3.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console-3.2.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-3.2.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console.exe"),
)

GIMP2_MODEL_ORDER = (
    "realesr-animevideov3-x4",
    "RealESRGAN_General_x4_v3",
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "UltraSharp-4x",
    "AnimeSharp-4x",
)


@dataclass(slots=True)
class GimpInfo:
    executable: Path
    version_text: str
    major_version: int


@dataclass(slots=True)
class NoiseReductionSettings:
    preset: str = "weak"


@dataclass(slots=True)
class UnsharpMaskSettings:
    preset: str = "weak"


@dataclass(slots=True)
class GimpExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    version_text: str
    output_path: Path | None
    duration_seconds: float


def find_first_gimp_path() -> Path | None:
    for candidate in COMMON_GIMP_PATHS:
        if candidate.exists():
            return candidate
    return None


def validate_gimp_path(gimp_path: Path | None) -> None:
    if gimp_path is None or not str(gimp_path).strip():
        raise ValueError("GIMP の実行ファイルパスが未設定です。GUI で指定してください。")
    if not gimp_path.exists() or not gimp_path.is_file():
        raise ValueError(f"GIMP の実行ファイルが見つかりません: {gimp_path}")


def detect_gimp_info(gimp_path: Path) -> GimpInfo:
    validate_gimp_path(gimp_path)
    completed = run_command([str(gimp_path), "--version"], timeout_seconds=20)
    text = (completed.stdout or completed.stderr or "").strip()
    version_match = re.search(r"(?:version|gimp)[^0-9]*(\d+)\.(\d+)\.(\d+)", text, re.IGNORECASE)
    if version_match:
        major_version = int(version_match.group(1))
    else:
        name_match = re.search(r"(\d+)", gimp_path.stem)
        if not name_match:
            raise RuntimeError(f"GIMP のバージョンを判定できませんでした: {gimp_path}")
        major_version = int(name_match.group(1))
    return GimpInfo(executable=gimp_path, version_text=text, major_version=major_version)


def is_gimp_available(gimp_path: Path | None) -> tuple[bool, str]:
    if gimp_path is None or not str(gimp_path).strip():
        return False, "GIMP 未設定"
    if not gimp_path.exists():
        return False, f"GIMP 不明: {gimp_path}"
    try:
        info = detect_gimp_info(gimp_path)
    except Exception as exc:
        return False, f"GIMP 確認失敗: {exc}"
    return True, info.version_text


def normalize_noise_preset(value: str) -> str:
    preset = (value or "off").strip().lower()
    if preset not in NOISE_PRESET_ORDER:
        raise ValueError(f"未対応のノイズ除去設定です: {value}")
    return preset


def normalize_unsharp_preset(value: str) -> str:
    preset = (value or "off").strip().lower()
    if preset not in UNSHARP_PRESET_ORDER:
        raise ValueError(f"未対応のアンシャープ設定です: {value}")
    return preset


def describe_noise_settings(settings: NoiseReductionSettings) -> str:
    return normalize_noise_preset(settings.preset)


def describe_unsharp_settings(settings: UnsharpMaskSettings) -> str:
    return normalize_unsharp_preset(settings.preset)


def _noise_parameters_for_gimp2(settings: NoiseReductionSettings) -> tuple[float, int, int] | None:
    preset = normalize_noise_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return (2.0, 1, 248)
    if preset == "medium":
        return (4.0, 2, 246)
    return (6.0, 3, 244)


def _noise_iterations_for_gimp3(settings: NoiseReductionSettings) -> int | None:
    preset = normalize_noise_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return 1
    if preset == "medium":
        return 2
    return 4


def _unsharp_parameters(settings: UnsharpMaskSettings) -> tuple[float, float, float] | None:
    preset = normalize_unsharp_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return (1.2, 0.35, 0.02)
    if preset == "medium":
        return (1.8, 0.5, 0.02)
    return (2.6, 0.75, 0.01)


def _scheme_string(path_or_text: str) -> str:
    return path_or_text.replace("\\", "/").replace("\"", "\\\"")


def _build_gimp2_preprocess_script(
    *,
    input_path: Path,
    output_path: Path,
    noise_settings: NoiseReductionSettings,
    unsharp_settings: UnsharpMaskSettings,
) -> str:
    input_text = _scheme_string(str(input_path))
    output_text = _scheme_string(str(output_path))
    lines = [
        f'(let* ((image (car (gimp-file-load RUN-NONINTERACTIVE "{input_text}" "{input_text}")))',
        "       (drawable (car (gimp-image-get-active-layer image))))",
    ]
    noise_values = _noise_parameters_for_gimp2(noise_settings)
    if noise_values is not None:
        radius, black_level, white_level = noise_values
        lines.append(
            f"  (plug-in-despeckle RUN-NONINTERACTIVE image drawable {radius:.3f} {black_level:d} -1 {white_level:d})"
        )
    unsharp_values = _unsharp_parameters(unsharp_settings)
    if unsharp_values is not None:
        radius, amount, threshold = unsharp_values
        threshold_value = int(round(threshold * 255.0))
        lines.append(
            "  "
            f"(plug-in-unsharp-mask RUN-NONINTERACTIVE image drawable {radius:.3f} {amount:.3f} {threshold_value:d})"
        )
    lines.extend(
        [
            f'  (file-png-save-defaults RUN-NONINTERACTIVE image drawable "{output_text}" "{output_text}")',
            "  (gimp-image-delete image))",
        ]
    )
    return "\n".join(lines)


def _build_gimp3_preprocess_script(
    *,
    input_path: Path,
    output_path: Path,
    noise_settings: NoiseReductionSettings,
    unsharp_settings: UnsharpMaskSettings,
) -> str:
    input_text = _scheme_string(str(input_path))
    output_text = _scheme_string(str(output_path))
    lines = [
        "(script-fu-use-v3)",
        f'(let* ((image (gimp-file-load RUN-NONINTERACTIVE "{input_text}"))',
        "       (drawable (vector-ref (gimp-image-get-selected-drawables image) 0)))",
    ]
    iterations = _noise_iterations_for_gimp3(noise_settings)
    if iterations is not None:
        lines.append(
            "  "
            f'(gimp-drawable-merge-new-filter drawable "gegl:noise-reduction" "Noise Reduction" '
            f"LAYER-MODE-REPLACE 1.0 #:iterations {iterations:d})"
        )
    unsharp_values = _unsharp_parameters(unsharp_settings)
    if unsharp_values is not None:
        radius, amount, threshold = unsharp_values
        lines.append(
            "  "
            f'(gimp-drawable-merge-new-filter drawable "gegl:unsharp-mask" "Unsharp Mask" '
            f"LAYER-MODE-REPLACE 1.0 #:std-dev {radius:.3f} #:scale {amount:.3f} #:threshold {threshold:.3f})"
        )
    lines.extend(
        [
            f'  (gimp-file-save RUN-NONINTERACTIVE image "{output_text}" 0)',
            "  (gimp-image-delete image))",
        ]
    )
    return "\n".join(lines)


def _build_gimp2_upscale_script(
    *,
    input_path: Path,
    output_path: Path,
    model_name: str,
    scale: int,
) -> str:
    if model_name not in GIMP2_MODEL_ORDER:
        raise ValueError(f"GIMP 2 直接呼び出しに未対応のモデルです: {model_name}")
    input_text = _scheme_string(str(input_path))
    output_text = _scheme_string(str(output_path))
    model_index = GIMP2_MODEL_ORDER.index(model_name)
    lines = [
        f'(let* ((image (car (gimp-file-load RUN-NONINTERACTIVE "{input_text}" "{input_text}")))',
        "       (drawable (car (gimp-image-get-active-layer image))))",
        f"  (python-fu-upscale-with-ncnn RUN-NONINTERACTIVE image drawable {model_index:d} 0 FALSE {float(scale):.3f})",
        "  (set! drawable (car (gimp-image-get-active-layer image)))",
        f'  (file-png-save-defaults RUN-NONINTERACTIVE image drawable "{output_text}" "{output_text}")',
        "  (gimp-image-delete image))",
    ]
    return "\n".join(lines)


def _run_gimp_batch_script(
    gimp_info: GimpInfo,
    script: str,
    *,
    stop_check: Callable[[], bool] | None = None,
    timeout_seconds: float = 1800,
) -> CommandResult:
    command = [
        str(gimp_info.executable),
        "-i",
        "-d",
        "-f",
        "-s",
        "-c",
        "--batch-interpreter=plug-in-script-fu-eval",
        "--batch",
        script,
    ]
    if gimp_info.major_version >= 3:
        command.append("--quit")
    else:
        command.extend(["--batch", "(gimp-quit 0)"])
    return run_command(command, timeout_seconds=timeout_seconds, stop_check=stop_check)


def _build_result(result: CommandResult, gimp_info: GimpInfo, output_path: Path) -> GimpExecutionResult:
    return GimpExecutionResult(
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        version_text=gimp_info.version_text,
        output_path=output_path if output_path.exists() else None,
        duration_seconds=result.duration_seconds,
    )


def run_gimp_preprocess(
    *,
    gimp_path: Path,
    input_path: Path,
    output_path: Path,
    noise_settings: NoiseReductionSettings,
    unsharp_settings: UnsharpMaskSettings,
    gimp_info: GimpInfo | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> GimpExecutionResult:
    info = gimp_info or detect_gimp_info(gimp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = (
        _build_gimp3_preprocess_script(
            input_path=input_path,
            output_path=output_path,
            noise_settings=noise_settings,
            unsharp_settings=unsharp_settings,
        )
        if info.major_version >= 3
        else _build_gimp2_preprocess_script(
            input_path=input_path,
            output_path=output_path,
            noise_settings=noise_settings,
            unsharp_settings=unsharp_settings,
        )
    )
    completed = _run_gimp_batch_script(info, script, stop_check=stop_check)
    result = _build_result(completed, info, output_path)
    if result.exit_code != 0 or result.output_path is None:
        raise RuntimeError(
            "GIMP 前処理に失敗しました。"
            f" exit_code={result.exit_code}"
            f" stdout={result.stdout}"
            f" stderr={result.stderr}"
        )
    return result


def collect_gimp_upscale_plugin_dirs(gimp_info: GimpInfo) -> list[Path]:
    install_root = gimp_info.executable.parent.parent
    appdata_root = Path(os.environ.get("APPDATA", ""))
    version_dirs = ["2.10", "3.0", "3.2"]
    search_roots = [appdata_root / "GIMP" / version / "plug-ins" for version in version_dirs]
    search_roots.append(install_root / "lib" / "gimp" / ("3.0" if gimp_info.major_version >= 3 else "2.0") / "plug-ins")
    candidates: list[Path] = []
    for root in search_roots:
        for name in ("gimp2_upscale", "gimp3_upscale"):
            candidate = root / name
            if candidate.exists() and candidate.is_dir():
                candidates.append(candidate)
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate.resolve()).lower()
        if key not in seen:
            unique.append(candidate)
            seen.add(key)
    return unique


def has_gimp2_direct_upscale(gimp_info: GimpInfo) -> bool:
    if gimp_info.major_version >= 3:
        return False
    for candidate in collect_gimp_upscale_plugin_dirs(gimp_info):
        if (candidate / "gimp2_upscale.py").exists():
            return True
    return False


def run_gimp_plugin_direct_upscale(
    *,
    gimp_path: Path,
    input_path: Path,
    output_path: Path,
    model_name: str,
    scale: int,
    gimp_info: GimpInfo | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> GimpExecutionResult:
    info = gimp_info or detect_gimp_info(gimp_path)
    if info.major_version >= 3:
        raise RuntimeError("GIMP 3 はモデル選択付きの非対話 AI Upscale が安定しないため、直接呼び出しを行いません。")
    if not has_gimp2_direct_upscale(info):
        raise RuntimeError("GIMP 2 の AI Upscale プラグインが見つかりませんでした。")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    script = _build_gimp2_upscale_script(
        input_path=input_path,
        output_path=output_path,
        model_name=model_name,
        scale=scale,
    )
    completed = _run_gimp_batch_script(info, script, stop_check=stop_check, timeout_seconds=3600)
    result = _build_result(completed, info, output_path)
    if result.exit_code != 0 or result.output_path is None:
        raise RuntimeError(
            "GIMP AI Upscale の直接呼び出しに失敗しました。"
            f" exit_code={result.exit_code}"
            f" stdout={result.stdout}"
            f" stderr={result.stderr}"
        )
    return result
