from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


NOISE_PRESET_ORDER = ("off", "weak", "medium", "strong", "detail")
UNSHARP_PRESET_ORDER = ("off", "weak", "medium", "strong", "detail")
POST_UNSHARP_PRESET_ORDER = ("off", "weak", "medium", "strong")

COMMON_GIMP_PATHS = (
    Path(r"C:\Program Files\GIMP 2\bin\gimp-console-2.10.exe"),
    Path(r"C:\Program Files\GIMP 2\bin\gimp-2.10.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console-3.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console-3.2.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-3.2.exe"),
    Path(r"C:\Program Files\GIMP 3\bin\gimp-console.exe"),
)


@dataclass(slots=True)
class GimpInfo:
    executable: Path
    version_text: str
    major_version: int


@dataclass(slots=True)
class NoiseReductionSettings:
    preset: str = "weak"
    detail_radius: float = 3.0
    detail_black_level: int = 1
    detail_white_level: int = 248
    detail_iterations: int = 1


@dataclass(slots=True)
class UnsharpMaskSettings:
    preset: str = "weak"
    detail_radius: float = 1.2
    detail_amount: float = 0.35
    detail_threshold: float = 0.02


@dataclass(slots=True)
class GimpExecutionResult:
    used: bool
    exit_code: int
    stdout: str
    stderr: str
    version_text: str
    output_path: Path | None


def find_first_gimp_path() -> Path | None:
    for candidate in COMMON_GIMP_PATHS:
        if candidate.exists():
            return candidate
    return None


def validate_gimp_path(gimp_path: Path | None) -> None:
    if gimp_path is None or not str(gimp_path).strip():
        raise ValueError("GIMP の実行ファイルパスが未設定です。GUI で gimp-console.exe または gimp.exe を指定してください。")
    if not gimp_path.exists() or not gimp_path.is_file():
        raise ValueError(f"GIMP の実行ファイルが見つかりません: {gimp_path}")


def detect_gimp_info(gimp_path: Path) -> GimpInfo:
    validate_gimp_path(gimp_path)
    completed = subprocess.run(
        [str(gimp_path), "--version"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=20,
    )
    text = (completed.stdout or completed.stderr or "").strip()
    version_match = re.search(r"version\s+(\d+)\.(\d+)\.(\d+)", text, re.IGNORECASE)
    if not version_match:
        name_match = re.search(r"gimp[-\s]?(\d+)", gimp_path.name.lower())
        if name_match:
            major_version = int(name_match.group(1))
        else:
            raise RuntimeError(f"GIMP のバージョンを判定できませんでした: {gimp_path}")
    else:
        major_version = int(version_match.group(1))
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
    return True, f"GIMP {info.version_text}"


def normalize_noise_preset(value: str) -> str:
    preset = (value or "off").strip().lower()
    if preset not in NOISE_PRESET_ORDER:
        raise ValueError(f"未対応のノイズ除去設定です: {value}")
    return preset


def normalize_unsharp_preset(value: str) -> str:
    preset = (value or "off").strip().lower()
    if preset not in UNSHARP_PRESET_ORDER:
        raise ValueError(f"未対応のアンシャープマスク設定です: {value}")
    return preset


def normalize_post_unsharp_preset(value: str) -> str:
    preset = (value or "off").strip().lower()
    if preset not in POST_UNSHARP_PRESET_ORDER:
        raise ValueError(f"未対応の後処理シャープ設定です: {value}")
    return preset


def _noise_parameters_for_gimp2(settings: NoiseReductionSettings) -> tuple[float, int, int] | None:
    preset = normalize_noise_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return (2.0, 1, 248)
    if preset == "medium":
        return (4.0, 2, 246)
    if preset == "strong":
        return (6.0, 3, 244)
    return (
        max(float(settings.detail_radius), 0.0),
        max(min(int(settings.detail_black_level), 255), 0),
        max(min(int(settings.detail_white_level), 255), 0),
    )


def _noise_iterations_for_gimp3(settings: NoiseReductionSettings) -> int | None:
    preset = normalize_noise_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return 1
    if preset == "medium":
        return 2
    if preset == "strong":
        return 4
    return max(int(settings.detail_iterations), 0)


def _unsharp_parameters(settings: UnsharpMaskSettings, *, allow_detail: bool = True) -> tuple[float, float, float] | None:
    preset = normalize_unsharp_preset(settings.preset) if allow_detail else normalize_post_unsharp_preset(settings.preset)
    if preset == "off":
        return None
    if preset == "weak":
        return (1.2, 0.35, 0.02)
    if preset == "medium":
        return (1.8, 0.5, 0.02)
    if preset == "strong":
        return (2.6, 0.75, 0.01)
    return (
        max(float(settings.detail_radius), 0.0),
        max(float(settings.detail_amount), 0.0),
        min(max(float(settings.detail_threshold), 0.0), 1.0),
    )


def describe_noise_settings(settings: NoiseReductionSettings) -> str:
    preset = normalize_noise_preset(settings.preset)
    if preset != "detail":
        return preset
    return (
        "detail"
        f"(radius={settings.detail_radius:.2f},black={int(settings.detail_black_level)},"
        f"white={int(settings.detail_white_level)},iterations={int(settings.detail_iterations)})"
    )


def describe_unsharp_settings(settings: UnsharpMaskSettings, *, allow_detail: bool = True) -> str:
    preset = normalize_unsharp_preset(settings.preset) if allow_detail else normalize_post_unsharp_preset(settings.preset)
    if preset != "detail" or not allow_detail:
        return preset
    return (
        "detail"
        f"(radius={settings.detail_radius:.3f},amount={settings.detail_amount:.3f},"
        f"threshold={settings.detail_threshold:.3f})"
    )


def _scheme_string(path_or_text: str) -> str:
    return path_or_text.replace("\\", "/").replace("\"", "\\\"")


def _build_gimp2_script(
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


def _build_gimp3_script(
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


def run_gimp_processing(
    *,
    gimp_path: Path,
    input_path: Path,
    output_path: Path,
    noise_settings: NoiseReductionSettings,
    unsharp_settings: UnsharpMaskSettings,
) -> GimpExecutionResult:
    info = detect_gimp_info(gimp_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    script = (
        _build_gimp3_script(
            input_path=input_path,
            output_path=output_path,
            noise_settings=noise_settings,
            unsharp_settings=unsharp_settings,
        )
        if info.major_version >= 3
        else _build_gimp2_script(
            input_path=input_path,
            output_path=output_path,
            noise_settings=noise_settings,
            unsharp_settings=unsharp_settings,
        )
    )

    command = [
        str(info.executable),
        "-i",
        "-d",
        "-f",
        "-s",
        "-c",
        "--batch-interpreter=plug-in-script-fu-eval",
        "--batch",
        script,
    ]
    if info.major_version >= 3:
        command.append("--quit")
    else:
        command.extend(["--batch", "(gimp-quit 0)"])

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        timeout=180,
    )
    result = GimpExecutionResult(
        used=True,
        exit_code=int(completed.returncode),
        stdout=(completed.stdout or "").strip(),
        stderr=(completed.stderr or "").strip(),
        version_text=info.version_text,
        output_path=output_path if output_path.exists() else None,
    )
    if result.exit_code != 0 or result.output_path is None:
        error_parts = [f"GIMP 処理に失敗しました。exit_code={result.exit_code}"]
        if result.stdout:
            error_parts.append(f"stdout={result.stdout}")
        if result.stderr:
            error_parts.append(f"stderr={result.stderr}")
        raise RuntimeError(" | ".join(error_parts))
    return result
