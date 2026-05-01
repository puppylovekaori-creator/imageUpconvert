from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import tempfile
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .alpha_utils import bleed_transparent_rgb, resize_alpha
from .crop_utils import apply_preview_crop_mode
from .cutout_utils import apply_person_cutout
from .gimp_runner import (
    GimpExecutionResult,
    NoiseReductionSettings,
    UnsharpMaskSettings,
    describe_noise_settings,
    describe_unsharp_settings,
    normalize_post_unsharp_preset,
    run_gimp_processing,
    validate_gimp_path,
)
from .image_io import (
    build_flat_output_path,
    build_output_path,
    collect_input_files,
    copy_failed_file,
    load_image,
    save_png,
)
from .log_utils import CsvLogger

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local runtime
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


APP_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODELS_BY_SCALE = {
    2: APP_ROOT / "models" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x2.pth",
    4: APP_ROOT / "models" / "swinir" / "001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
}
VENDOR_NETWORK_PATH = APP_ROOT / "vendor" / "SwinIR" / "models" / "network_swinir.py"
PREVIEW_ROOT = APP_ROOT / "temp" / "preview"
WHO_VALUE = "imageUpconvert"
SUPPORTED_PREVIEW_RANGES = {"full", "center", "face_near"}
PROCESSING_MODE_GIMP_ONLY = "gimp_only"
PROCESSING_MODE_SWINIR_ONLY = "swinir_only"
PROCESSING_MODE_GIMP_PRE_SWINIR = "gimp_pre_swinir"
PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST = "gimp_pre_swinir_gimp_post"
SUPPORTED_PROCESSING_MODES = {
    PROCESSING_MODE_GIMP_ONLY,
    PROCESSING_MODE_SWINIR_ONLY,
    PROCESSING_MODE_GIMP_PRE_SWINIR,
    PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST,
}
PROCESSING_MODE_LABELS = {
    PROCESSING_MODE_GIMP_ONLY: "GIMPのみ",
    PROCESSING_MODE_SWINIR_ONLY: "SwinIRのみ",
    PROCESSING_MODE_GIMP_PRE_SWINIR: "GIMP前処理 + SwinIR",
    PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST: "GIMP前処理 + SwinIR + GIMP後処理",
}


class UserCancelledError(RuntimeError):
    """Raised when the user requests an immediate cancel."""


@dataclass(slots=True)
class PipelineOptions:
    scale: int
    tile_size: int
    tile_overlap: int
    processing_mode: str = PROCESSING_MODE_GIMP_PRE_SWINIR
    gimp_path: Path | None = None
    use_gimp_pre: bool = True
    noise_settings: NoiseReductionSettings = field(default_factory=NoiseReductionSettings)
    pre_unsharp_settings: UnsharpMaskSettings = field(default_factory=UnsharpMaskSettings)
    use_gimp_post: bool = False
    post_unsharp_settings: UnsharpMaskSettings = field(
        default_factory=lambda: UnsharpMaskSettings(preset="off")
    )
    use_cutout: bool = False


@dataclass(slots=True)
class BatchOptions:
    input_dir: Path
    output_dir: Path
    recursive: bool
    skip_existing: bool
    collision_policy: str
    pipeline: PipelineOptions
    test_mode: bool = False
    test_limit: int = 5


@dataclass(slots=True)
class PreviewOptions:
    input_file: Path
    preview_range: str
    preview_ratio: str
    pipeline: PipelineOptions
    preview_dir: Path = PREVIEW_ROOT


@dataclass(slots=True)
class ComparisonOptions:
    input_file: Path
    output_dir: Path
    skip_existing: bool
    collision_policy: str
    pipeline: PipelineOptions


@dataclass(slots=True)
class ModelDescriptor:
    task: str
    scale: int
    training_patch_size: int
    large_model: bool
    window_size: int
    label: str


@dataclass(slots=True)
class RunSummary:
    kind: str
    total_files: int
    processed: int
    skipped: int
    failed: int
    warnings: int
    stopped: bool
    cancelled: bool
    output_dir: str


@dataclass(slots=True)
class PreviewSummary:
    kind: str
    input_file: str
    preview_range: str
    scale: int
    original_preview_path: str
    gimp_pre_preview_path: str
    swinir_preview_path: str
    post_preview_path: str
    message: str


@dataclass(slots=True)
class PreparedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    alpha_present: str
    preview_range: str
    preview_crop_range: str


@dataclass(slots=True)
class ImageState:
    rgb: np.ndarray
    alpha: np.ndarray | None


@dataclass(slots=True)
class StageLog:
    used: bool = False
    exit_code: int = 0
    stdout: str = ""
    stderr: str = ""
    version_text: str = ""


@dataclass(slots=True)
class PipelineExecutionResult:
    prepared: PreparedImage
    final_state: ImageState
    actual_scale: str
    warning_message: str
    gimp_pre_log: StageLog
    gimp_post_log: StageLog


def normalize_processing_mode(value: str) -> str:
    mode = (value or PROCESSING_MODE_GIMP_PRE_SWINIR).strip().lower()
    if mode not in SUPPORTED_PROCESSING_MODES:
        raise ValueError(f"未対応の処理モードです: {value}")
    return mode


def processing_mode_uses_swinir(mode: str) -> bool:
    normalized = normalize_processing_mode(mode)
    return normalized in {
        PROCESSING_MODE_SWINIR_ONLY,
        PROCESSING_MODE_GIMP_PRE_SWINIR,
        PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST,
    }


def processing_mode_uses_gimp_pre(mode: str) -> bool:
    normalized = normalize_processing_mode(mode)
    return normalized in {
        PROCESSING_MODE_GIMP_ONLY,
        PROCESSING_MODE_GIMP_PRE_SWINIR,
        PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST,
    }


def processing_mode_uses_gimp_post(mode: str) -> bool:
    normalized = normalize_processing_mode(mode)
    return normalized == PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST


def pipeline_uses_swinir(pipeline: PipelineOptions) -> bool:
    return processing_mode_uses_swinir(pipeline.processing_mode)


def pipeline_uses_gimp_pre(pipeline: PipelineOptions) -> bool:
    return processing_mode_uses_gimp_pre(pipeline.processing_mode)


def pipeline_uses_gimp_post(pipeline: PipelineOptions) -> bool:
    return processing_mode_uses_gimp_post(pipeline.processing_mode)


def describe_processing_mode(mode: str) -> str:
    normalized = normalize_processing_mode(mode)
    return PROCESSING_MODE_LABELS[normalized]


class InterruptController:
    def __init__(
        self,
        is_stop_requested: Callable[[], bool] | None = None,
        is_cancel_requested: Callable[[], bool] | None = None,
    ) -> None:
        self._is_stop_requested = is_stop_requested or (lambda: False)
        self._is_cancel_requested = is_cancel_requested or (lambda: False)

    def raise_if_cancelled(self) -> None:
        if self._is_cancel_requested():
            raise UserCancelledError("ユーザー操作によりキャンセルされました。")

    def should_stop(self) -> bool:
        return self._is_stop_requested()


class ModelRuntimeCache:
    def __init__(self, device: Any, message_callback: Callable[[str], None]) -> None:
        self._device = device
        self._message_callback = message_callback
        self._cache: dict[tuple[str, int], tuple[ModelDescriptor, Any]] = {}

    @property
    def device(self) -> Any:
        return self._device

    def get(self, scale: int) -> tuple[ModelDescriptor, Any]:
        model_path = get_default_model_path(scale)
        key = (str(model_path.resolve()), scale)
        if key not in self._cache:
            descriptor = infer_model_descriptor(model_path, scale)
            model = define_model(descriptor, model_path, self._device)
            self._cache[key] = (descriptor, model)
            self._message_callback(
                f"内部モデル読込: {model_path.name} / {descriptor.label} / x{descriptor.scale}"
            )
        return self._cache[key]


@lru_cache(maxsize=1)
def load_swinir_network_class():
    spec = importlib.util.spec_from_file_location("vendor_swinir_network", VENDOR_NETWORK_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load SwinIR network from {VENDOR_NETWORK_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SwinIR


def detect_model_scale_from_filename(model_path: Path) -> int | None:
    scale_match = re.search(r"_x(\d+)(?:[_\.]|$)", model_path.name.lower())
    if not scale_match:
        return None
    return int(scale_match.group(1))


def infer_model_descriptor(model_path: Path, requested_scale: int) -> ModelDescriptor:
    name = model_path.name.lower()

    if "classicalsr" in name or name.startswith("001_"):
        task = "classical_sr"
        training_patch_size = (
            int(re.search(r"_s(\d+)w", name).group(1)) if re.search(r"_s(\d+)w", name) else 64
        )
        large_model = False
        label = "classical_sr"
    elif "lightweightsr" in name or name.startswith("002_"):
        task = "lightweight_sr"
        training_patch_size = 64
        large_model = False
        label = "lightweight_sr"
    elif "realsr" in name or name.startswith("003_"):
        task = "real_sr"
        training_patch_size = 64
        large_model = "swinir-l" in name or "_l_" in name or "dfowmfc" in name
        label = "real_sr"
    else:
        raise ValueError(
            "ファイル名から公式 SwinIR モデル種別を判定できませんでした。"
            " 001_classicalSR_... や 003_realSR_... のような公式ファイル名を使ってください。"
        )

    detected_scale = detect_model_scale_from_filename(model_path)
    if detected_scale is None:
        detected_scale = 4 if task == "real_sr" else requested_scale
    if requested_scale != detected_scale:
        raise ValueError(
            f"選択した倍率 x{requested_scale} と、モデル名から判定した倍率 x{detected_scale} が一致しません。"
        )
    if task == "real_sr" and requested_scale != 4:
        raise ValueError("公式の real_sr モデルは x4 のみ対応です。")

    return ModelDescriptor(
        task=task,
        scale=requested_scale,
        training_patch_size=training_patch_size,
        large_model=large_model,
        window_size=8,
        label=label,
    )


def ensure_torch_available() -> None:
    if TORCH_IMPORT_ERROR is not None or torch is None:
        raise RuntimeError(
            "PyTorch を利用できません。setup.bat を先に実行するか、venv に torch をインストールしてください。"
        ) from TORCH_IMPORT_ERROR


def get_default_model_path(scale: int) -> Path:
    model_path = DEFAULT_MODELS_BY_SCALE.get(scale)
    if model_path is None:
        raise ValueError(f"未対応の倍率です: x{scale}")
    return model_path


def list_available_internal_scales() -> list[int]:
    return [scale for scale, path in DEFAULT_MODELS_BY_SCALE.items() if path.exists()]


def validate_pipeline_options(options: PipelineOptions) -> None:
    mode = normalize_processing_mode(options.processing_mode)
    if options.tile_size < 0 or options.tile_overlap < 0:
        raise ValueError("tile size と tile overlap は 0 以上で指定してください。")
    if processing_mode_uses_swinir(mode):
        if options.scale not in {2, 4}:
            raise ValueError("倍率は 2x または 4x のみ対応です。")
        model_path = get_default_model_path(options.scale)
        if not model_path.exists() or not model_path.is_file():
            raise ValueError(f"内部モデルが見つかりません: {model_path}")
    if processing_mode_uses_gimp_pre(mode) or processing_mode_uses_gimp_post(mode):
        validate_gimp_path(options.gimp_path)
    if processing_mode_uses_gimp_post(mode):
        normalize_post_unsharp_preset(options.post_unsharp_settings.preset)


def validate_batch_options(options: BatchOptions) -> None:
    if not options.input_dir.exists() or not options.input_dir.is_dir():
        raise ValueError("入力フォルダが存在しません。")
    validate_pipeline_options(options.pipeline)

    input_resolved = options.input_dir.resolve()
    output_resolved = options.output_dir.resolve()
    if input_resolved == output_resolved or input_resolved in output_resolved.parents:
        raise ValueError(
            "出力フォルダは入力フォルダと別にしてください。入力フォルダの内側も指定できません。"
        )


def validate_preview_options(options: PreviewOptions) -> None:
    if not options.input_file.exists() or not options.input_file.is_file():
        raise ValueError("プレビュー対象画像が存在しません。")
    if options.preview_range not in SUPPORTED_PREVIEW_RANGES:
        raise ValueError("プレビュー範囲は 全体 / 中央crop / 顔付近crop のいずれかを選んでください。")
    validate_pipeline_options(options.pipeline)


def validate_comparison_options(options: ComparisonOptions) -> None:
    if not options.input_file.exists() or not options.input_file.is_file():
        raise ValueError("比較対象画像が存在しません。")
    if options.pipeline.scale not in {2, 4}:
        raise ValueError("比較処理の倍率は 2x または 4x のみ対応です。")
    model_path = get_default_model_path(options.pipeline.scale)
    if not model_path.exists() or not model_path.is_file():
        raise ValueError(f"比較処理用の内部モデルが見つかりません: {model_path}")
    validate_gimp_path(options.pipeline.gimp_path)
    normalize_post_unsharp_preset(options.pipeline.post_unsharp_settings.preset)


def get_effective_output_dir(options: BatchOptions) -> Path:
    if not options.test_mode:
        return options.output_dir
    return options.output_dir.with_name(f"{options.output_dir.name}_test")


def get_comparison_output_dir(output_dir: Path) -> Path:
    return output_dir.parent / "comparison"


def numpy_to_tensor(rgb: np.ndarray, device: Any):
    chw = np.transpose(rgb, (2, 0, 1))
    return torch.from_numpy(chw).float().unsqueeze(0).to(device)


def tensor_to_numpy(output: Any) -> np.ndarray:
    array = output.squeeze(0).clamp(0, 1).cpu().numpy()
    return np.transpose(array, (1, 2, 0))


def pad_to_window(img: Any, window_size: int):
    _, _, h_old, w_old = img.size()
    h_pad = ((h_old + window_size - 1) // window_size) * window_size - h_old
    w_pad = ((w_old + window_size - 1) // window_size) * window_size - w_old

    if h_pad > 0:
        img = torch.cat([img, torch.flip(img, [2])], dim=2)[:, :, : h_old + h_pad, :]
    if w_pad > 0:
        img = torch.cat([img, torch.flip(img, [3])], dim=3)[:, :, :, : w_old + w_pad]
    return img, h_old, w_old


def run_model_tiles(
    img: Any,
    model: Any,
    descriptor: ModelDescriptor,
    tile_size: int,
    tile_overlap: int,
    controller: InterruptController,
):
    if tile_size <= 0:
        controller.raise_if_cancelled()
        return model(img)

    b, c, h, w = img.size()
    tile = min(tile_size, h, w)
    tile = (tile // descriptor.window_size) * descriptor.window_size
    if tile < descriptor.window_size:
        controller.raise_if_cancelled()
        return model(img)

    overlap = min(tile_overlap, max(tile - 1, 0))
    stride = max(tile - overlap, 1)
    scale = descriptor.scale

    h_idx_list = list(range(0, max(h - tile, 0), stride)) + [h - tile]
    w_idx_list = list(range(0, max(w - tile, 0), stride)) + [w - tile]

    output_accumulator = torch.zeros(b, c, h * scale, w * scale).type_as(img)
    weight_accumulator = torch.zeros_like(output_accumulator)

    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            controller.raise_if_cancelled()
            in_patch = img[..., h_idx : h_idx + tile, w_idx : w_idx + tile]
            out_patch = model(in_patch)
            out_patch_mask = torch.ones_like(out_patch)

            output_accumulator[
                ..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale
            ].add_(out_patch)
            weight_accumulator[
                ..., h_idx * scale : (h_idx + tile) * scale, w_idx * scale : (w_idx + tile) * scale
            ].add_(out_patch_mask)

    return output_accumulator.div_(weight_accumulator)


def load_checkpoint(model_path: Path, device: Any) -> dict[str, Any]:
    try:
        return torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:  # pragma: no cover - older torch fallback
        return torch.load(model_path, map_location=device)


def define_model(descriptor: ModelDescriptor, model_path: Path, device: Any):
    SwinIR = load_swinir_network_class()

    if descriptor.task == "classical_sr":
        model = SwinIR(
            upscale=descriptor.scale,
            in_chans=3,
            img_size=descriptor.training_patch_size,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180,
            num_heads=[6, 6, 6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )
        param_key = "params"
    elif descriptor.task == "lightweight_sr":
        model = SwinIR(
            upscale=descriptor.scale,
            in_chans=3,
            img_size=64,
            window_size=8,
            img_range=1.0,
            depths=[6, 6, 6, 6],
            embed_dim=60,
            num_heads=[6, 6, 6, 6],
            mlp_ratio=2,
            upsampler="pixelshuffledirect",
            resi_connection="1conv",
        )
        param_key = "params"
    elif descriptor.task == "real_sr":
        if descriptor.large_model:
            model = SwinIR(
                upscale=descriptor.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6, 6, 6, 6],
                embed_dim=240,
                num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="3conv",
            )
        else:
            model = SwinIR(
                upscale=descriptor.scale,
                in_chans=3,
                img_size=64,
                window_size=8,
                img_range=1.0,
                depths=[6, 6, 6, 6, 6, 6],
                embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6],
                mlp_ratio=2,
                upsampler="nearest+conv",
                resi_connection="1conv",
            )
        param_key = "params_ema"
    else:
        raise ValueError(f"未対応のモデル種別です: {descriptor.task}")

    checkpoint = load_checkpoint(model_path, device)
    state_dict = checkpoint[param_key] if param_key in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model.to(device)


def upscale_rgb(
    rgb: np.ndarray,
    model: Any,
    descriptor: ModelDescriptor,
    device: Any,
    tile_size: int,
    tile_overlap: int,
    controller: InterruptController,
) -> np.ndarray:
    image_tensor = numpy_to_tensor(rgb, device)
    image_tensor, h_old, w_old = pad_to_window(image_tensor, descriptor.window_size)
    with torch.inference_mode():
        output = run_model_tiles(
            image_tensor,
            model,
            descriptor,
            tile_size=tile_size,
            tile_overlap=tile_overlap,
            controller=controller,
        )
        output = output[..., : h_old * descriptor.scale, : w_old * descriptor.scale]
    return tensor_to_numpy(output)


def format_size(width: int, height: int) -> str:
    return f"{width}x{height}"


def format_actual_scale(
    *,
    processing_width: int,
    processing_height: int,
    actual_width: int,
    actual_height: int,
    requested_scale: int,
) -> tuple[str, str]:
    scale_x = actual_width / processing_width if processing_width else 0.0
    scale_y = actual_height / processing_height if processing_height else 0.0
    actual_scale_text = f"{scale_x:.4f}x{scale_y:.4f}"
    expected_width = processing_width * requested_scale
    expected_height = processing_height * requested_scale

    warning_message = ""
    if actual_width != expected_width or actual_height != expected_height:
        warning_message = (
            f"指定倍率 x{requested_scale} に対して出力サイズが一致しません。"
            f" 期待={expected_width}x{expected_height}, 実際={actual_width}x{actual_height}"
        )
    return actual_scale_text, warning_message


def _log_from_gimp_result(result: GimpExecutionResult | None) -> StageLog:
    if result is None:
        return StageLog()
    return StageLog(
        used=result.used,
        exit_code=result.exit_code,
        stdout=result.stdout,
        stderr=result.stderr,
        version_text=result.version_text,
    )


def _save_state(path: Path, state: ImageState) -> None:
    save_png(path, state.rgb, state.alpha)


def prepare_image(
    input_path: Path,
    *,
    preview_range: str,
    preview_ratio: str,
) -> PreparedImage:
    loaded = load_image(input_path)
    crop_result = apply_preview_crop_mode(
        loaded.rgb,
        loaded.alpha,
        mode=preview_range,
        ratio_text=preview_ratio,
    )
    height, width = crop_result.rgb.shape[:2]
    return PreparedImage(
        rgb=crop_result.rgb,
        alpha=crop_result.alpha,
        original_width=loaded.width,
        original_height=loaded.height,
        processing_width=width,
        processing_height=height,
        alpha_present="yes" if crop_result.alpha is not None else "no",
        preview_range=crop_result.mode,
        preview_crop_range=crop_result.format_range(),
    )


def _apply_gimp_stage(
    *,
    state: ImageState,
    gimp_path: Path,
    noise_settings: NoiseReductionSettings,
    unsharp_settings: UnsharpMaskSettings,
    temp_dir: Path,
    prefix: str,
    controller: InterruptController,
) -> tuple[ImageState, StageLog]:
    controller.raise_if_cancelled()
    input_path = temp_dir / f"{prefix}_input.png"
    output_path = temp_dir / f"{prefix}_output.png"
    _save_state(input_path, state)
    result = run_gimp_processing(
        gimp_path=gimp_path,
        input_path=input_path,
        output_path=output_path,
        noise_settings=noise_settings,
        unsharp_settings=unsharp_settings,
    )
    controller.raise_if_cancelled()
    loaded = load_image(output_path)
    return ImageState(rgb=loaded.rgb, alpha=loaded.alpha), _log_from_gimp_result(result)


def _upscale_state(
    *,
    state: ImageState,
    scale: int,
    tile_size: int,
    tile_overlap: int,
    model_cache: ModelRuntimeCache,
    controller: InterruptController,
) -> tuple[ImageState, str, str]:
    descriptor, model = model_cache.get(scale)
    prepared_rgb = bleed_transparent_rgb(state.rgb, state.alpha)
    output_rgb = upscale_rgb(
        prepared_rgb,
        model=model,
        descriptor=descriptor,
        device=model_cache.device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        controller=controller,
    )
    output_alpha = resize_alpha(state.alpha, scale) if state.alpha is not None else None
    actual_height, actual_width = output_rgb.shape[:2]
    processing_height, processing_width = state.rgb.shape[:2]
    actual_scale, warning_message = format_actual_scale(
        processing_width=processing_width,
        processing_height=processing_height,
        actual_width=actual_width,
        actual_height=actual_height,
        requested_scale=scale,
    )
    return ImageState(output_rgb, output_alpha), actual_scale, warning_message


def execute_pipeline(
    *,
    input_path: Path,
    pipeline: PipelineOptions,
    model_cache: ModelRuntimeCache | None,
    controller: InterruptController,
    preview_range: str = "full",
    preview_ratio: str = "4:5",
) -> PipelineExecutionResult:
    mode = normalize_processing_mode(pipeline.processing_mode)
    prepared = prepare_image(
        input_path,
        preview_range=preview_range,
        preview_ratio=preview_ratio,
    )
    original_state = ImageState(prepared.rgb, prepared.alpha)
    gimp_pre_log = StageLog()
    gimp_post_log = StageLog()

    temp_dir = _make_work_dir("imageupconvert_")
    try:
        current_state = original_state
        if processing_mode_uses_gimp_pre(mode):
            controller.raise_if_cancelled()
            current_state, gimp_pre_log = _apply_gimp_stage(
                state=original_state,
                gimp_path=Path(pipeline.gimp_path) if pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=pipeline.noise_settings,
                unsharp_settings=pipeline.pre_unsharp_settings,
                temp_dir=temp_dir,
                prefix="gimp_pre",
                controller=controller,
            )

        actual_scale = "1.0000x1.0000"
        warning_message = ""
        if processing_mode_uses_swinir(mode):
            if model_cache is None:
                raise RuntimeError("SwinIR を使う処理モードですが、モデル実行環境を初期化できませんでした。")
            current_state, actual_scale, warning_message = _upscale_state(
                state=current_state,
                scale=pipeline.scale,
                tile_size=pipeline.tile_size,
                tile_overlap=pipeline.tile_overlap,
                model_cache=model_cache,
                controller=controller,
            )
        else:
            current_height, current_width = current_state.rgb.shape[:2]
            actual_scale, warning_message = format_actual_scale(
                processing_width=prepared.processing_width,
                processing_height=prepared.processing_height,
                actual_width=current_width,
                actual_height=current_height,
                requested_scale=1,
            )

        if processing_mode_uses_gimp_post(mode):
            controller.raise_if_cancelled()
            current_state, gimp_post_log = _apply_gimp_stage(
                state=current_state,
                gimp_path=Path(pipeline.gimp_path) if pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=NoiseReductionSettings(preset="off"),
                unsharp_settings=pipeline.post_unsharp_settings,
                temp_dir=temp_dir,
                prefix="gimp_post",
                controller=controller,
            )

        if pipeline.use_cutout:
            controller.raise_if_cancelled()
            cutout_rgb, cutout_alpha = apply_person_cutout(current_state.rgb, current_state.alpha)
            current_state = ImageState(cutout_rgb, cutout_alpha)

        return PipelineExecutionResult(
            prepared=prepared,
            final_state=current_state,
            actual_scale=actual_scale,
            warning_message=warning_message,
            gimp_pre_log=gimp_pre_log,
            gimp_post_log=gimp_post_log,
        )
    finally:
        _cleanup_work_dir(temp_dir)


def make_log_row(
    *,
    task_kind: str,
    processing_mode: str,
    processed_at: str,
    input_path: Path,
    output_path: Path | None,
    original_size: str,
    processing_input_size: str,
    output_size: str,
    requested_scale: int | None,
    actual_scale: str,
    model_path: Path | None,
    pipeline: PipelineOptions,
    prepared: PreparedImage | None,
    swinir_used: bool,
    gimp_pre_used: bool,
    gimp_post_used: bool,
    cutout_used: bool,
    gimp_pre_log: StageLog,
    gimp_post_log: StageLog,
    result: str,
    warning_message: str,
    error_message: str,
    elapsed_seconds: float,
) -> dict[str, str]:
    return {
        "who": WHO_VALUE,
        "processed_at": processed_at,
        "task_kind": task_kind,
        "processing_mode": normalize_processing_mode(processing_mode),
        "input_file_path": str(input_path),
        "output_file_path": str(output_path) if output_path else "",
        "original_image_size": original_size,
        "processing_input_size": processing_input_size,
        "output_image_size": output_size,
        "requested_scale": str(requested_scale) if requested_scale is not None else "",
        "actual_scale": actual_scale,
        "swinir_enabled": "yes" if swinir_used else "no",
        "model": str(model_path) if model_path is not None else "",
        "tile_size": str(pipeline.tile_size),
        "tile_overlap": str(pipeline.tile_overlap),
        "alpha_present": prepared.alpha_present if prepared is not None else "",
        "preview_range": prepared.preview_range if prepared is not None else "",
        "preview_crop_range": prepared.preview_crop_range if prepared is not None else "",
        "gimp_path": str(pipeline.gimp_path) if pipeline.gimp_path else "",
        "gimp_version": gimp_post_log.version_text or gimp_pre_log.version_text,
        "gimp_pre_enabled": "yes" if gimp_pre_used else "no",
        "noise_reduction_setting": describe_noise_settings(pipeline.noise_settings) if gimp_pre_used else "",
        "unsharp_pre_setting": describe_unsharp_settings(pipeline.pre_unsharp_settings) if gimp_pre_used else "",
        "gimp_post_enabled": "yes" if gimp_post_used else "no",
        "unsharp_post_setting": describe_unsharp_settings(
            pipeline.post_unsharp_settings, allow_detail=False
        ) if gimp_post_used else "",
        "cutout_enabled": "yes" if cutout_used else "no",
        "gimp_pre_exit_code": str(gimp_pre_log.exit_code) if gimp_pre_log.used else "",
        "gimp_pre_stdout": gimp_pre_log.stdout.replace("\r", " ").replace("\n", " ").strip(),
        "gimp_pre_stderr": gimp_pre_log.stderr.replace("\r", " ").replace("\n", " ").strip(),
        "gimp_post_exit_code": str(gimp_post_log.exit_code) if gimp_post_log.used else "",
        "gimp_post_stdout": gimp_post_log.stdout.replace("\r", " ").replace("\n", " ").strip(),
        "gimp_post_stderr": gimp_post_log.stderr.replace("\r", " ").replace("\n", " ").strip(),
        "processing_result": result,
        "warning_message": warning_message,
        "error_message": error_message,
        "processing_time_seconds": f"{elapsed_seconds:.3f}",
    }


def _build_batch_output_suffix(pipeline: PipelineOptions) -> str:
    mode = normalize_processing_mode(pipeline.processing_mode)
    if mode == PROCESSING_MODE_GIMP_ONLY:
        suffix = "_gimp_only"
    elif mode == PROCESSING_MODE_SWINIR_ONLY:
        suffix = f"_swinir_x{pipeline.scale}"
    elif mode == PROCESSING_MODE_GIMP_PRE_SWINIR:
        suffix = f"_gimp_pre_swinir_x{pipeline.scale}"
    else:
        suffix = f"_gimp_pre_swinir_x{pipeline.scale}_gimp_post"
    if pipeline.use_cutout:
        suffix += "_cutout"
    return suffix


def _make_temp_output_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}.tmp{output_path.suffix}")


def _save_final_output(output_path: Path, state: ImageState) -> None:
    temp_output_path = _make_temp_output_path(output_path)
    if temp_output_path.exists():
        temp_output_path.unlink()
    save_png(temp_output_path, state.rgb, state.alpha)
    temp_output_path.replace(output_path)


def _make_work_dir(prefix: str) -> Path:
    temp_root = APP_ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=prefix, dir=temp_root))


def _cleanup_work_dir(path: Path) -> None:
    try:
        shutil.rmtree(path, ignore_errors=True)
    except Exception:
        return


def run_batch(
    options: BatchOptions,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    message_callback: Callable[[str], None] | None = None,
    controller: InterruptController | None = None,
) -> RunSummary:
    validate_batch_options(options)

    progress_callback = progress_callback or (lambda payload: None)
    message_callback = message_callback or (lambda message: None)
    controller = controller or InterruptController()
    mode = normalize_processing_mode(options.pipeline.processing_mode)
    if processing_mode_uses_swinir(mode):
        ensure_torch_available()

    output_dir = get_effective_output_dir(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_root = output_dir / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(options.input_dir, options.recursive)
    if options.test_mode:
        files = files[: options.test_limit]
    if not files:
        raise ValueError("入力フォルダ内に対応画像が見つかりませんでした。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if processing_mode_uses_swinir(mode) else None
    model_cache = ModelRuntimeCache(device, message_callback) if device is not None else None
    logger = CsvLogger(output_dir)
    summary = RunSummary(
        kind="run",
        total_files=len(files),
        processed=0,
        skipped=0,
        failed=0,
        warnings=0,
        stopped=False,
        cancelled=False,
        output_dir=str(output_dir),
    )

    model_path = get_default_model_path(options.pipeline.scale) if processing_mode_uses_swinir(mode) else None
    message_callback(f"出力先: {output_dir}")
    message_callback(f"処理モード: {describe_processing_mode(mode)}")
    if device is not None:
        message_callback(f"使用デバイス: {device}")
    if model_path is not None:
        message_callback(f"内部モデル: {model_path.name}")
    else:
        message_callback("SwinIR: この処理モードでは使いません。")

    try:
        for index, input_path in enumerate(files, start=1):
            controller.raise_if_cancelled()
            if controller.should_stop():
                summary.stopped = True
                message_callback("停止要求を受け付けました。現在のファイル完了後、新しいファイルは開始しません。")
                break

            progress_callback(
                {
                    "phase": "started",
                    "completed": index - 1,
                    "total": len(files),
                    "path": str(input_path),
                }
            )
            processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            started_at = time.perf_counter()
            output_path = build_output_path(
                input_path=input_path,
                input_root=options.input_dir,
                output_root=output_dir,
                suffix=_build_batch_output_suffix(options.pipeline),
                collision_policy=options.collision_policy,
                skip_existing=options.skip_existing,
            )

            if output_path is None:
                row = make_log_row(
                    task_kind="batch",
                    processing_mode=mode,
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=None,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=options.pipeline.scale if processing_mode_uses_swinir(mode) else None,
                    actual_scale="",
                    model_path=model_path,
                    pipeline=options.pipeline,
                    prepared=None,
                    swinir_used=processing_mode_uses_swinir(mode),
                    gimp_pre_used=processing_mode_uses_gimp_pre(mode),
                    gimp_post_used=processing_mode_uses_gimp_post(mode),
                    cutout_used=options.pipeline.use_cutout,
                    gimp_pre_log=StageLog(),
                    gimp_post_log=StageLog(),
                    result="skipped",
                    warning_message="",
                    error_message="同名の出力ファイルが既にあるためスキップしました。",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                summary.skipped += 1
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(files),
                        "path": str(input_path),
                        "result": "skipped",
                    }
                )
                continue

            try:
                result = execute_pipeline(
                    input_path=input_path,
                    pipeline=options.pipeline,
                    model_cache=model_cache,
                    controller=controller,
                )
                _save_final_output(output_path, result.final_state)
                output_height, output_width = result.final_state.rgb.shape[:2]
                row_result = "processed_warning" if result.warning_message else "processed"
                row = make_log_row(
                    task_kind="batch",
                    processing_mode=mode,
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=output_path,
                    original_size=format_size(result.prepared.original_width, result.prepared.original_height),
                    processing_input_size=format_size(
                        result.prepared.processing_width, result.prepared.processing_height
                    ),
                    output_size=format_size(output_width, output_height),
                    requested_scale=options.pipeline.scale if processing_mode_uses_swinir(mode) else None,
                    actual_scale=result.actual_scale,
                    model_path=model_path,
                    pipeline=options.pipeline,
                    prepared=result.prepared,
                    swinir_used=processing_mode_uses_swinir(mode),
                    gimp_pre_used=processing_mode_uses_gimp_pre(mode),
                    gimp_post_used=processing_mode_uses_gimp_post(mode),
                    cutout_used=options.pipeline.use_cutout,
                    gimp_pre_log=result.gimp_pre_log,
                    gimp_post_log=result.gimp_post_log,
                    result=row_result,
                    warning_message=result.warning_message,
                    error_message="",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                summary.processed += 1
                if result.warning_message:
                    summary.warnings += 1
                    message_callback(f"警告付き完了: {input_path.name} / {result.warning_message}")
                else:
                    message_callback(f"処理完了: {input_path.name}")
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(files),
                        "path": str(input_path),
                        "result": row_result,
                    }
                )
            except UserCancelledError:
                raise
            except Exception as exc:
                failed_copy_path = copy_failed_file(input_path, options.input_dir, failed_root)
                row = make_log_row(
                    task_kind="batch",
                    processing_mode=mode,
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=output_path,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=options.pipeline.scale if processing_mode_uses_swinir(mode) else None,
                    actual_scale="",
                    model_path=model_path,
                    pipeline=options.pipeline,
                    prepared=None,
                    swinir_used=processing_mode_uses_swinir(mode),
                    gimp_pre_used=processing_mode_uses_gimp_pre(mode),
                    gimp_post_used=processing_mode_uses_gimp_post(mode),
                    cutout_used=options.pipeline.use_cutout,
                    gimp_pre_log=StageLog(),
                    gimp_post_log=StageLog(),
                    result="failed",
                    warning_message="",
                    error_message=f"{exc} | failed_copy={failed_copy_path}",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                logger.log_failed(row)
                summary.failed += 1
                message_callback(f"失敗: {input_path.name} -> {exc}")
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(files),
                        "path": str(input_path),
                        "result": "failed",
                    }
                )
    except UserCancelledError:
        summary.cancelled = True
        message_callback("キャンセルされました。処理を中断しました。")
    finally:
        logger.close()

    return summary


def _save_preview_outputs(
    preview_dir: Path,
    input_file: Path,
    *,
    original_state: ImageState,
    gimp_pre_state: ImageState | None,
    swinir_state: ImageState | None,
    final_state: ImageState,
    scale: int | None,
    preview_range: str,
) -> tuple[str, str, str, str]:
    preview_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = input_file.stem
    range_suffix = preview_range
    original_path = preview_dir / f"{safe_stem}_{range_suffix}_original.png"
    _save_state(original_path, original_state)

    gimp_pre_path = ""
    if gimp_pre_state is not None:
        gimp_pre_target = preview_dir / f"{safe_stem}_{range_suffix}_gimp_pre.png"
        _save_state(gimp_pre_target, gimp_pre_state)
        gimp_pre_path = str(gimp_pre_target)

    swinir_path = ""
    if swinir_state is not None:
        scale_suffix = f"_x{scale}" if scale is not None else ""
        swinir_target = preview_dir / f"{safe_stem}_{range_suffix}_swinir{scale_suffix}.png"
        _save_state(swinir_target, swinir_state)
        swinir_path = str(swinir_target)

    final_target = preview_dir / f"{safe_stem}_{range_suffix}_final.png"
    _save_state(final_target, final_state)
    return str(original_path), gimp_pre_path, swinir_path, str(final_target)


def run_preview(
    options: PreviewOptions,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    message_callback: Callable[[str], None] | None = None,
    controller: InterruptController | None = None,
) -> PreviewSummary:
    validate_preview_options(options)

    progress_callback = progress_callback or (lambda payload: None)
    message_callback = message_callback or (lambda message: None)
    controller = controller or InterruptController()
    mode = normalize_processing_mode(options.pipeline.processing_mode)
    if processing_mode_uses_swinir(mode):
        ensure_torch_available()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if processing_mode_uses_swinir(mode) else None
    model_cache = ModelRuntimeCache(device, message_callback) if device is not None else None

    progress_callback({"phase": "started", "completed": 0, "total": 1, "path": str(options.input_file)})
    prepared = prepare_image(
        options.input_file,
        preview_range=options.preview_range,
        preview_ratio=options.preview_ratio,
    )
    original_state = ImageState(prepared.rgb, prepared.alpha)
    gimp_pre_state: ImageState | None = None
    swinir_reference_state: ImageState | None = None
    final_state = original_state

    temp_dir = _make_work_dir("imageupconvert_preview_")
    try:
        if processing_mode_uses_gimp_pre(mode):
            gimp_pre_state, _ = _apply_gimp_stage(
                state=original_state,
                gimp_path=Path(options.pipeline.gimp_path) if options.pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=options.pipeline.noise_settings,
                unsharp_settings=options.pipeline.pre_unsharp_settings,
                temp_dir=temp_dir,
                prefix="preview_pre",
                controller=controller,
            )
            final_state = gimp_pre_state

        if processing_mode_uses_swinir(mode):
            if model_cache is None:
                raise RuntimeError("SwinIR プレビュー環境を初期化できませんでした。")
            swinir_reference_state, _, _ = _upscale_state(
                state=original_state,
                scale=options.pipeline.scale,
                tile_size=options.pipeline.tile_size,
                tile_overlap=options.pipeline.tile_overlap,
                model_cache=model_cache,
                controller=controller,
            )
            final_input_state = gimp_pre_state if gimp_pre_state is not None else original_state
            final_state, _, _ = _upscale_state(
                state=final_input_state,
                scale=options.pipeline.scale,
                tile_size=options.pipeline.tile_size,
                tile_overlap=options.pipeline.tile_overlap,
                model_cache=model_cache,
                controller=controller,
            )
        if processing_mode_uses_gimp_post(mode):
            final_state, _ = _apply_gimp_stage(
                state=final_state,
                gimp_path=Path(options.pipeline.gimp_path) if options.pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=NoiseReductionSettings(preset="off"),
                unsharp_settings=options.pipeline.post_unsharp_settings,
                temp_dir=temp_dir,
                prefix="preview_post",
                controller=controller,
            )
        if options.pipeline.use_cutout:
            cutout_rgb, cutout_alpha = apply_person_cutout(final_state.rgb, final_state.alpha)
            final_state = ImageState(cutout_rgb, cutout_alpha)

        original_path, gimp_pre_path, swinir_path, post_path = _save_preview_outputs(
            options.preview_dir,
            options.input_file,
            original_state=original_state,
            gimp_pre_state=gimp_pre_state,
            swinir_state=swinir_reference_state,
            final_state=final_state,
            scale=options.pipeline.scale if processing_mode_uses_swinir(mode) else None,
            preview_range=options.preview_range,
        )
    finally:
        _cleanup_work_dir(temp_dir)

    progress_callback({"phase": "finished", "completed": 1, "total": 1, "path": str(options.input_file)})
    message = (
        f"プレビュー生成完了: {options.input_file.name}"
        f" / 範囲={options.preview_range}"
        f" / 処理モード={describe_processing_mode(mode)}"
    )
    if processing_mode_uses_swinir(mode):
        message += f" / x{options.pipeline.scale}"
    message_callback(message)
    return PreviewSummary(
        kind="preview",
        input_file=str(options.input_file),
        preview_range=options.preview_range,
        scale=options.pipeline.scale if processing_mode_uses_swinir(mode) else 1,
        original_preview_path=original_path,
        gimp_pre_preview_path=gimp_pre_path,
        swinir_preview_path=swinir_path,
        post_preview_path=post_path,
        message=message,
    )


def run_comparison(
    options: ComparisonOptions,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    message_callback: Callable[[str], None] | None = None,
    controller: InterruptController | None = None,
) -> RunSummary:
    ensure_torch_available()
    validate_comparison_options(options)

    progress_callback = progress_callback or (lambda payload: None)
    message_callback = message_callback or (lambda message: None)
    controller = controller or InterruptController()

    output_dir = get_comparison_output_dir(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_root = output_dir / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)
    logger = CsvLogger(output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_cache = ModelRuntimeCache(device, message_callback)
    model_path = get_default_model_path(options.pipeline.scale)

    prepared = prepare_image(
        options.input_file,
        preview_range="full",
        preview_ratio="4:5",
    )
    original_state = ImageState(prepared.rgb, prepared.alpha)
    gimp_pre_log = StageLog()
    gimp_post_log = StageLog()
    summary = RunSummary(
        kind="run",
        total_files=0,
        processed=0,
        skipped=0,
        failed=0,
        warnings=0,
        stopped=False,
        cancelled=False,
        output_dir=str(output_dir),
    )

    try:
        temp_dir = _make_work_dir("imageupconvert_compare_")
        try:
            gimp_only_state, gimp_pre_log = _apply_gimp_stage(
                state=original_state,
                gimp_path=Path(options.pipeline.gimp_path) if options.pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=options.pipeline.noise_settings,
                unsharp_settings=options.pipeline.pre_unsharp_settings,
                temp_dir=temp_dir,
                prefix="comparison_pre",
                controller=controller,
            )

            swinir_only_state, swinir_only_scale, swinir_only_warning = _upscale_state(
                state=original_state,
                scale=options.pipeline.scale,
                tile_size=options.pipeline.tile_size,
                tile_overlap=options.pipeline.tile_overlap,
                model_cache=model_cache,
                controller=controller,
            )
            gimp_pre_swinir_state, gimp_pre_swinir_scale, gimp_pre_swinir_warning = _upscale_state(
                state=gimp_only_state,
                scale=options.pipeline.scale,
                tile_size=options.pipeline.tile_size,
                tile_overlap=options.pipeline.tile_overlap,
                model_cache=model_cache,
                controller=controller,
            )
            gimp_pre_swinir_gimp_post_state, gimp_post_log = _apply_gimp_stage(
                state=gimp_pre_swinir_state,
                gimp_path=Path(options.pipeline.gimp_path) if options.pipeline.gimp_path is not None else None,  # type: ignore[arg-type]
                noise_settings=NoiseReductionSettings(preset="off"),
                unsharp_settings=options.pipeline.post_unsharp_settings,
                temp_dir=temp_dir,
                prefix="comparison_post",
                controller=controller,
            )

            variant_rows: list[dict[str, Any]] = [
                {
                    "name": "original",
                    "state": original_state,
                    "actual_scale": "1.0000x1.0000",
                    "warning_message": "",
                    "processing_mode": PROCESSING_MODE_GIMP_ONLY,
                    "swinir_used": False,
                    "gimp_pre_used": False,
                    "gimp_post_used": False,
                    "cutout_used": False,
                    "model_path": None,
                    "gimp_pre_log": StageLog(),
                    "gimp_post_log": StageLog(),
                },
                {
                    "name": "gimp_only",
                    "state": gimp_only_state,
                    "actual_scale": "1.0000x1.0000",
                    "warning_message": "",
                    "processing_mode": PROCESSING_MODE_GIMP_ONLY,
                    "swinir_used": False,
                    "gimp_pre_used": True,
                    "gimp_post_used": False,
                    "cutout_used": False,
                    "model_path": None,
                    "gimp_pre_log": gimp_pre_log,
                    "gimp_post_log": StageLog(),
                },
                {
                    "name": "swinir_only",
                    "state": swinir_only_state,
                    "actual_scale": swinir_only_scale,
                    "warning_message": swinir_only_warning,
                    "processing_mode": PROCESSING_MODE_SWINIR_ONLY,
                    "swinir_used": True,
                    "gimp_pre_used": False,
                    "gimp_post_used": False,
                    "cutout_used": False,
                    "model_path": model_path,
                    "gimp_pre_log": StageLog(),
                    "gimp_post_log": StageLog(),
                },
                {
                    "name": "gimp_pre_swinir",
                    "state": gimp_pre_swinir_state,
                    "actual_scale": gimp_pre_swinir_scale,
                    "warning_message": gimp_pre_swinir_warning,
                    "processing_mode": PROCESSING_MODE_GIMP_PRE_SWINIR,
                    "swinir_used": True,
                    "gimp_pre_used": True,
                    "gimp_post_used": False,
                    "cutout_used": False,
                    "model_path": model_path,
                    "gimp_pre_log": gimp_pre_log,
                    "gimp_post_log": StageLog(),
                },
                {
                    "name": "gimp_pre_swinir_gimp_post",
                    "state": gimp_pre_swinir_gimp_post_state,
                    "actual_scale": gimp_pre_swinir_scale,
                    "warning_message": gimp_pre_swinir_warning,
                    "processing_mode": PROCESSING_MODE_GIMP_PRE_SWINIR_GIMP_POST,
                    "swinir_used": True,
                    "gimp_pre_used": True,
                    "gimp_post_used": True,
                    "cutout_used": False,
                    "model_path": model_path,
                    "gimp_pre_log": gimp_pre_log,
                    "gimp_post_log": gimp_post_log,
                },
            ]
            if options.pipeline.use_cutout:
                gimp_only_cutout_rgb, gimp_only_cutout_alpha = apply_person_cutout(
                    gimp_only_state.rgb,
                    gimp_only_state.alpha,
                )
                gimp_pre_swinir_cutout_rgb, gimp_pre_swinir_cutout_alpha = apply_person_cutout(
                    gimp_pre_swinir_state.rgb,
                    gimp_pre_swinir_state.alpha,
                )
                variant_rows.extend(
                    [
                        {
                            "name": "gimp_only_cutout",
                            "state": ImageState(gimp_only_cutout_rgb, gimp_only_cutout_alpha),
                            "actual_scale": "1.0000x1.0000",
                            "warning_message": "",
                            "processing_mode": PROCESSING_MODE_GIMP_ONLY,
                            "swinir_used": False,
                            "gimp_pre_used": True,
                            "gimp_post_used": False,
                            "cutout_used": True,
                            "model_path": None,
                            "gimp_pre_log": gimp_pre_log,
                            "gimp_post_log": StageLog(),
                        },
                        {
                            "name": "gimp_pre_swinir_cutout",
                            "state": ImageState(
                                gimp_pre_swinir_cutout_rgb,
                                gimp_pre_swinir_cutout_alpha,
                            ),
                            "actual_scale": gimp_pre_swinir_scale,
                            "warning_message": gimp_pre_swinir_warning,
                            "processing_mode": PROCESSING_MODE_GIMP_PRE_SWINIR,
                            "swinir_used": True,
                            "gimp_pre_used": True,
                            "gimp_post_used": False,
                            "cutout_used": True,
                            "model_path": model_path,
                            "gimp_pre_log": gimp_pre_log,
                            "gimp_post_log": StageLog(),
                        },
                    ]
                )
            summary.total_files = len(variant_rows)

            for index, variant in enumerate(variant_rows, start=1):
                controller.raise_if_cancelled()
                if controller.should_stop():
                    summary.stopped = True
                    message_callback("停止要求を受け付けました。現在の比較パターン完了後に停止します。")
                    break

                progress_callback(
                    {
                        "phase": "started",
                        "completed": index - 1,
                        "total": len(variant_rows),
                        "path": f"{options.input_file.name} [{variant['name']}]",
                    }
                )
                processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                started_at = time.perf_counter()
                suffix = (
                    f"_{variant['name']}_x{options.pipeline.scale}"
                    if variant["swinir_used"]
                    else f"_{variant['name']}"
                )
                output_path = build_flat_output_path(
                    input_path=options.input_file,
                    output_root=output_dir,
                    suffix=suffix,
                    collision_policy=options.collision_policy,
                    skip_existing=options.skip_existing,
                )
                if output_path is None:
                    row = make_log_row(
                        task_kind="comparison",
                        processing_mode=variant["processing_mode"],
                        processed_at=processed_at,
                        input_path=options.input_file,
                        output_path=None,
                        original_size="",
                        processing_input_size="",
                        output_size="",
                        requested_scale=options.pipeline.scale if variant["swinir_used"] else None,
                        actual_scale="",
                        model_path=variant["model_path"],
                        pipeline=options.pipeline,
                        prepared=prepared,
                        swinir_used=bool(variant["swinir_used"]),
                        gimp_pre_used=bool(variant["gimp_pre_used"]),
                        gimp_post_used=bool(variant["gimp_post_used"]),
                        cutout_used=bool(variant["cutout_used"]),
                        gimp_pre_log=variant["gimp_pre_log"],
                        gimp_post_log=variant["gimp_post_log"],
                        result="skipped",
                        warning_message="",
                        error_message="同名の比較出力ファイルが既にあるためスキップしました。",
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                    logger.log_processing(row)
                    summary.skipped += 1
                    progress_callback(
                        {
                            "phase": "finished",
                            "completed": index,
                            "total": len(variant_rows),
                            "path": f"{options.input_file.name} [{variant['name']}]",
                            "result": "skipped",
                        }
                    )
                    continue

                try:
                    state = variant["state"]
                    variant_scale = str(variant["actual_scale"])
                    warning_message = str(variant["warning_message"])
                    _save_final_output(output_path, state)
                    output_height, output_width = state.rgb.shape[:2]
                    result_name = "processed_warning" if warning_message else "processed"
                    row = make_log_row(
                        task_kind="comparison",
                        processing_mode=variant["processing_mode"],
                        processed_at=processed_at,
                        input_path=options.input_file,
                        output_path=output_path,
                        original_size=format_size(prepared.original_width, prepared.original_height),
                        processing_input_size=format_size(
                            prepared.processing_width, prepared.processing_height
                        ),
                        output_size=format_size(output_width, output_height),
                        requested_scale=options.pipeline.scale if variant["swinir_used"] else None,
                        actual_scale=variant_scale,
                        model_path=variant["model_path"],
                        pipeline=options.pipeline,
                        prepared=prepared,
                        swinir_used=bool(variant["swinir_used"]),
                        gimp_pre_used=bool(variant["gimp_pre_used"]),
                        gimp_post_used=bool(variant["gimp_post_used"]),
                        cutout_used=bool(variant["cutout_used"]),
                        gimp_pre_log=variant["gimp_pre_log"],
                        gimp_post_log=variant["gimp_post_log"],
                        result=result_name,
                        warning_message=warning_message,
                        error_message="",
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                    logger.log_processing(row)
                    summary.processed += 1
                    if warning_message:
                        summary.warnings += 1
                    message_callback(f"比較出力完了: {variant['name']}")
                    progress_callback(
                        {
                            "phase": "finished",
                            "completed": index,
                            "total": len(variant_rows),
                            "path": f"{options.input_file.name} [{variant['name']}]",
                            "result": result_name,
                        }
                    )
                except Exception as exc:
                    failed_copy_path = copy_failed_file(options.input_file, options.input_file.parent, failed_root)
                    row = make_log_row(
                        task_kind="comparison",
                        processing_mode=variant["processing_mode"],
                        processed_at=processed_at,
                        input_path=options.input_file,
                        output_path=output_path,
                        original_size="",
                        processing_input_size="",
                        output_size="",
                        requested_scale=options.pipeline.scale if variant["swinir_used"] else None,
                        actual_scale="",
                        model_path=variant["model_path"],
                        pipeline=options.pipeline,
                        prepared=prepared,
                        swinir_used=bool(variant["swinir_used"]),
                        gimp_pre_used=bool(variant["gimp_pre_used"]),
                        gimp_post_used=bool(variant["gimp_post_used"]),
                        cutout_used=bool(variant["cutout_used"]),
                        gimp_pre_log=variant["gimp_pre_log"],
                        gimp_post_log=variant["gimp_post_log"],
                        result="failed",
                        warning_message="",
                        error_message=f"{exc} | failed_copy={failed_copy_path}",
                        elapsed_seconds=time.perf_counter() - started_at,
                    )
                    logger.log_processing(row)
                    logger.log_failed(row)
                    summary.failed += 1
                    message_callback(f"比較出力失敗: {variant['name']} -> {exc}")
                    progress_callback(
                        {
                            "phase": "finished",
                            "completed": index,
                            "total": len(variant_rows),
                            "path": f"{options.input_file.name} [{variant['name']}]",
                            "result": "failed",
                        }
                    )
        finally:
            _cleanup_work_dir(temp_dir)
    except UserCancelledError:
        summary.cancelled = True
        message_callback("キャンセルされました。比較出力を中断しました。")
    finally:
        logger.close()

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GUI を使わずに SwinIR の一括処理を実行します。")
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=int, choices=[2, 4])
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--collision-policy", choices=["skip", "serial"], default="skip")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-limit", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = BatchOptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        recursive=args.recursive,
        skip_existing=args.skip_existing,
        collision_policy=args.collision_policy,
        pipeline=PipelineOptions(
            scale=args.scale,
            tile_size=args.tile_size,
            tile_overlap=args.tile_overlap,
        ),
        test_mode=args.test_mode,
        test_limit=args.test_limit,
    )

    def on_progress(payload: dict[str, Any]) -> None:
        if payload["phase"] == "started":
            print(f"[{payload['completed']}/{payload['total']}] {payload['path']}")

    summary = run_batch(options, progress_callback=on_progress, message_callback=print)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
