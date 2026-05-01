from __future__ import annotations

import argparse
import importlib.util
import json
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .alpha_utils import bleed_transparent_rgb, resize_alpha
from .crop_utils import center_crop_by_ratio, parse_ratio_text
from .image_io import (
    build_flat_output_path,
    build_output_path,
    collect_input_files,
    copy_failed_file,
    load_image,
    save_png,
)
from .log_utils import CsvLogger
from .sharpen_utils import apply_sharpen, is_sharpen_enabled, normalize_sharpen_strength

try:
    import torch
except Exception as exc:  # pragma: no cover - depends on local runtime
    torch = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


VENDOR_NETWORK_PATH = (
    Path(__file__).resolve().parent.parent / "vendor" / "SwinIR" / "models" / "network_swinir.py"
)
DEFAULT_COMPARISON_SHARPEN_LEVELS = ("none", "weak", "medium")


class UserCancelledError(RuntimeError):
    """Raised when the user requests an immediate cancel."""


@dataclass(slots=True)
class CropOptions:
    enabled: bool = False
    ratio: str = "4:5"


@dataclass(slots=True)
class BatchOptions:
    input_dir: Path
    output_dir: Path
    model_path: Path
    scale: int
    tile_size: int
    tile_overlap: int
    skip_existing: bool
    recursive: bool
    collision_policy: str
    sharpen_strength: str = "weak"
    crop_options: CropOptions = field(default_factory=CropOptions)
    test_mode: bool = False
    test_limit: int = 5


@dataclass(slots=True)
class ComparisonOptions:
    input_file: Path
    output_dir: Path
    model_paths_by_scale: dict[int, Path]
    tile_size: int
    tile_overlap: int
    skip_existing: bool
    collision_policy: str
    sharpen_levels: tuple[str, ...] = DEFAULT_COMPARISON_SHARPEN_LEVELS
    crop_options: CropOptions = field(default_factory=CropOptions)


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
    total_files: int
    processed: int
    skipped: int
    failed: int
    warnings: int
    stopped: bool
    cancelled: bool
    output_dir: str


@dataclass(slots=True)
class PreparedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    original_width: int
    original_height: int
    processing_width: int
    processing_height: int
    alpha_present: str
    crop_enabled: str
    crop_range: str


@dataclass(frozen=True, slots=True)
class VariantPlan:
    label: str
    model_path: Path
    scale: int
    sharpen_strength: str
    suffix: str


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

    def get(self, model_path: Path, scale: int) -> tuple[ModelDescriptor, Any]:
        key = (str(model_path.resolve()), scale)
        if key not in self._cache:
            descriptor = infer_model_descriptor(model_path, scale)
            model = define_model(descriptor, model_path, self._device)
            self._cache[key] = (descriptor, model)
            self._message_callback(
                f"モデル読込: {model_path.name} / {descriptor.label} / x{descriptor.scale}"
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
        training_patch_size = int(re.search(r"_s(\d+)w", name).group(1)) if re.search(r"_s(\d+)w", name) else 64
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


def validate_options(options: BatchOptions) -> None:
    if not options.input_dir.exists() or not options.input_dir.is_dir():
        raise ValueError("入力フォルダが存在しません。")
    if not options.model_path.exists() or not options.model_path.is_file():
        raise ValueError("モデルファイルが存在しません。")
    if options.scale not in {2, 4}:
        raise ValueError("倍率は 2 または 4 を選んでください。")
    if options.tile_size < 0 or options.tile_overlap < 0:
        raise ValueError("tile size と tile overlap は 0 以上で指定してください。")

    input_resolved = options.input_dir.resolve()
    output_resolved = options.output_dir.resolve()
    if input_resolved == output_resolved or input_resolved in output_resolved.parents:
        raise ValueError(
            "出力フォルダは入力フォルダと別にしてください。入力フォルダの内側も指定できません。"
        )

    normalize_sharpen_strength(options.sharpen_strength)
    if options.crop_options.enabled:
        parse_ratio_text(options.crop_options.ratio)


def validate_comparison_options(options: ComparisonOptions) -> None:
    if not options.input_file.exists() or not options.input_file.is_file():
        raise ValueError("比較対象の画像ファイルが存在しません。")
    if options.input_file.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
        raise ValueError("比較対象は .jpg / .jpeg / .png / .webp のみ対応です。")
    if options.tile_size < 0 or options.tile_overlap < 0:
        raise ValueError("tile size と tile overlap は 0 以上で指定してください。")
    if 2 not in options.model_paths_by_scale or 4 not in options.model_paths_by_scale:
        raise ValueError("比較処理には x2 と x4 の両方のモデルが必要です。")
    for scale, model_path in options.model_paths_by_scale.items():
        if scale not in {2, 4}:
            raise ValueError("比較処理で使える倍率は 2x / 4x のみです。")
        if not model_path.exists() or not model_path.is_file():
            raise ValueError(f"x{scale} 用モデルファイルが存在しません。")
        infer_model_descriptor(model_path, scale)
    for sharpen_level in options.sharpen_levels:
        normalize_sharpen_strength(sharpen_level)
    if options.crop_options.enabled:
        parse_ratio_text(options.crop_options.ratio)


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


def make_log_row(
    *,
    processed_at: str,
    input_path: Path,
    output_path: Path | None,
    original_size: str,
    processing_input_size: str,
    output_size: str,
    requested_scale: int,
    actual_scale: str,
    model_path: Path,
    tile_size: int,
    tile_overlap: int,
    alpha_present: str,
    crop_enabled: str,
    crop_range: str,
    sharpen_enabled: str,
    sharpen_strength: str,
    result: str,
    warning_message: str,
    error_message: str,
    elapsed_seconds: float,
) -> dict[str, str]:
    return {
        "processed_at": processed_at,
        "input_file_path": str(input_path),
        "output_file_path": str(output_path) if output_path else "",
        "original_image_size": original_size,
        "processing_input_size": processing_input_size,
        "output_image_size": output_size,
        "requested_scale": str(requested_scale),
        "actual_scale": actual_scale,
        "model": str(model_path),
        "tile_size": str(tile_size),
        "tile_overlap": str(tile_overlap),
        "alpha_present": alpha_present,
        "crop_enabled": crop_enabled,
        "crop_range": crop_range,
        "sharpen_enabled": sharpen_enabled,
        "sharpen_strength": sharpen_strength,
        "processing_result": result,
        "warning_message": warning_message,
        "error_message": error_message,
        "processing_time_seconds": f"{elapsed_seconds:.3f}",
    }


def build_output_suffix(
    *,
    scale: int,
    crop_enabled: bool,
    sharpen_strength: str,
    include_sharpen_none: bool = False,
) -> str:
    normalized_sharpen = normalize_sharpen_strength(sharpen_strength)
    suffix = f"_swinir_x{scale}"
    if crop_enabled:
        suffix += "_crop"
    if normalized_sharpen != "none" or include_sharpen_none:
        suffix += f"_sharp_{normalized_sharpen}"
    return suffix


def format_size(width: int, height: int) -> str:
    return f"{width}x{height}"


def format_actual_scale(
    *,
    processing_width: int,
    processing_height: int,
    expected_width: int,
    expected_height: int,
    actual_width: int,
    actual_height: int,
    requested_scale: int,
) -> tuple[str, str]:
    scale_x = actual_width / processing_width if processing_width else 0.0
    scale_y = actual_height / processing_height if processing_height else 0.0
    actual_scale_text = f"{scale_x:.4f}x{scale_y:.4f}"

    warning_message = ""
    if actual_width != expected_width or actual_height != expected_height:
        warning_message = (
            f"指定倍率 x{requested_scale} に対して出力サイズが一致しません。"
            f" 期待={expected_width}x{expected_height}, 実際={actual_width}x{actual_height}"
        )
    return actual_scale_text, warning_message


def prepare_image(input_path: Path, crop_options: CropOptions) -> PreparedImage:
    loaded = load_image(input_path)
    rgb = loaded.rgb
    alpha = loaded.alpha
    crop_enabled_text = "no"
    crop_range = ""

    if crop_options.enabled:
        crop_result = center_crop_by_ratio(rgb, alpha, crop_options.ratio)
        rgb = crop_result.rgb
        alpha = crop_result.alpha
        crop_enabled_text = "yes"
        crop_range = crop_result.format_range()

    prepared_rgb = bleed_transparent_rgb(rgb, alpha)
    height, width = prepared_rgb.shape[:2]
    return PreparedImage(
        rgb=prepared_rgb,
        alpha=alpha,
        original_width=loaded.width,
        original_height=loaded.height,
        processing_width=width,
        processing_height=height,
        alpha_present="yes" if alpha is not None else "no",
        crop_enabled=crop_enabled_text,
        crop_range=crop_range,
    )


def execute_variant(
    *,
    prepared_image: PreparedImage,
    plan: VariantPlan,
    output_path: Path,
    tile_size: int,
    tile_overlap: int,
    model_cache: ModelRuntimeCache,
    controller: InterruptController,
) -> tuple[str, str, str]:
    descriptor, model = model_cache.get(plan.model_path, plan.scale)
    output_rgb = upscale_rgb(
        prepared_image.rgb,
        model=model,
        descriptor=descriptor,
        device=model_cache.device,
        tile_size=tile_size,
        tile_overlap=tile_overlap,
        controller=controller,
    )
    output_rgb = apply_sharpen(output_rgb, plan.sharpen_strength)
    output_alpha = resize_alpha(prepared_image.alpha, plan.scale) if prepared_image.alpha is not None else None

    actual_height, actual_width = output_rgb.shape[:2]
    expected_width = prepared_image.processing_width * plan.scale
    expected_height = prepared_image.processing_height * plan.scale
    actual_scale, warning_message = format_actual_scale(
        processing_width=prepared_image.processing_width,
        processing_height=prepared_image.processing_height,
        expected_width=expected_width,
        expected_height=expected_height,
        actual_width=actual_width,
        actual_height=actual_height,
        requested_scale=plan.scale,
    )

    if output_alpha is not None:
        alpha_height, alpha_width = output_alpha.shape[:2]
        if alpha_width != actual_width or alpha_height != actual_height:
            raise ValueError(
                f"alpha 出力サイズが RGB と一致しません。RGB={actual_width}x{actual_height}, alpha={alpha_width}x{alpha_height}"
            )

    save_png(output_path, output_rgb, output_alpha)
    return format_size(actual_width, actual_height), actual_scale, warning_message


def run_batch(
    options: BatchOptions,
    *,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
    message_callback: Callable[[str], None] | None = None,
    controller: InterruptController | None = None,
) -> RunSummary:
    ensure_torch_available()
    validate_options(options)

    progress_callback = progress_callback or (lambda payload: None)
    message_callback = message_callback or (lambda message: None)
    controller = controller or InterruptController()

    output_dir = get_effective_output_dir(options)
    output_dir.mkdir(parents=True, exist_ok=True)
    failed_root = output_dir / "failed"
    failed_root.mkdir(parents=True, exist_ok=True)

    files = collect_input_files(options.input_dir, options.recursive)
    if options.test_mode:
        files = files[: options.test_limit]
    if not files:
        raise ValueError("入力フォルダ内に対応画像が見つかりませんでした。")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    message_callback(f"出力先: {output_dir}")
    message_callback(f"使用デバイス: {device}")
    message_callback(
        "crop: "
        + (
            f"ON ({options.crop_options.ratio})"
            if options.crop_options.enabled
            else "OFF"
        )
    )
    message_callback(f"シャープ処理: {normalize_sharpen_strength(options.sharpen_strength)}")

    summary = RunSummary(
        total_files=len(files),
        processed=0,
        skipped=0,
        failed=0,
        warnings=0,
        stopped=False,
        cancelled=False,
        output_dir=str(output_dir),
    )
    logger = CsvLogger(output_dir)
    model_cache = ModelRuntimeCache(device, message_callback)
    plan = VariantPlan(
        label=f"x{options.scale} / sharp {normalize_sharpen_strength(options.sharpen_strength)}",
        model_path=options.model_path,
        scale=options.scale,
        sharpen_strength=normalize_sharpen_strength(options.sharpen_strength),
        suffix=build_output_suffix(
            scale=options.scale,
            crop_enabled=options.crop_options.enabled,
            sharpen_strength=options.sharpen_strength,
            include_sharpen_none=False,
        ),
    )

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
                suffix=plan.suffix,
                collision_policy=options.collision_policy,
                skip_existing=options.skip_existing,
            )

            if output_path is None:
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=None,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=options.scale,
                    actual_scale="",
                    model_path=options.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present="",
                    crop_enabled="yes" if options.crop_options.enabled else "no",
                    crop_range=options.crop_options.ratio if options.crop_options.enabled else "",
                    sharpen_enabled="yes" if is_sharpen_enabled(options.sharpen_strength) else "no",
                    sharpen_strength=normalize_sharpen_strength(options.sharpen_strength),
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
                prepared = prepare_image(input_path, options.crop_options)
                output_size, actual_scale, warning_message = execute_variant(
                    prepared_image=prepared,
                    plan=plan,
                    output_path=output_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    model_cache=model_cache,
                    controller=controller,
                )
                result = "processed_warning" if warning_message else "processed"
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=output_path,
                    original_size=format_size(prepared.original_width, prepared.original_height),
                    processing_input_size=format_size(prepared.processing_width, prepared.processing_height),
                    output_size=output_size,
                    requested_scale=options.scale,
                    actual_scale=actual_scale,
                    model_path=options.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present=prepared.alpha_present,
                    crop_enabled=prepared.crop_enabled,
                    crop_range=prepared.crop_range,
                    sharpen_enabled="yes" if is_sharpen_enabled(options.sharpen_strength) else "no",
                    sharpen_strength=normalize_sharpen_strength(options.sharpen_strength),
                    result=result,
                    warning_message=warning_message,
                    error_message="",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                summary.processed += 1
                if warning_message:
                    summary.warnings += 1
                    message_callback(f"警告付き完了: {input_path.name} / {warning_message}")
                else:
                    message_callback(f"処理完了: {input_path.name}")
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(files),
                        "path": str(input_path),
                        "result": result,
                    }
                )
            except UserCancelledError:
                raise
            except Exception as exc:
                failed_copy_path = copy_failed_file(input_path, options.input_dir, failed_root)
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=input_path,
                    output_path=output_path,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=options.scale,
                    actual_scale="",
                    model_path=options.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present="",
                    crop_enabled="yes" if options.crop_options.enabled else "no",
                    crop_range=options.crop_options.ratio if options.crop_options.enabled else "",
                    sharpen_enabled="yes" if is_sharpen_enabled(options.sharpen_strength) else "no",
                    sharpen_strength=normalize_sharpen_strength(options.sharpen_strength),
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


def build_comparison_plans(options: ComparisonOptions) -> list[VariantPlan]:
    plans: list[VariantPlan] = []
    for scale in (2, 4):
        model_path = options.model_paths_by_scale[scale]
        for sharpen_strength in options.sharpen_levels:
            normalized_sharpen = normalize_sharpen_strength(sharpen_strength)
            plans.append(
                VariantPlan(
                    label=f"x{scale} / sharp {normalized_sharpen}",
                    model_path=model_path,
                    scale=scale,
                    sharpen_strength=normalized_sharpen,
                    suffix=build_output_suffix(
                        scale=scale,
                        crop_enabled=options.crop_options.enabled,
                        sharpen_strength=normalized_sharpen,
                        include_sharpen_none=True,
                    ),
                )
            )
    return plans


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

    plans = build_comparison_plans(options)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger = CsvLogger(output_dir)
    model_cache = ModelRuntimeCache(device, message_callback)

    message_callback(f"比較出力先: {output_dir}")
    message_callback(f"比較対象: {options.input_file}")
    message_callback(f"使用デバイス: {device}")
    message_callback(
        "crop: "
        + (
            f"ON ({options.crop_options.ratio})"
            if options.crop_options.enabled
            else "OFF"
        )
    )

    summary = RunSummary(
        total_files=len(plans),
        processed=0,
        skipped=0,
        failed=0,
        warnings=0,
        stopped=False,
        cancelled=False,
        output_dir=str(output_dir),
    )

    try:
        for index, plan in enumerate(plans, start=1):
            controller.raise_if_cancelled()
            if controller.should_stop():
                summary.stopped = True
                message_callback("停止要求を受け付けました。現在の比較パターン完了後に停止します。")
                break

            progress_callback(
                {
                    "phase": "started",
                    "completed": index - 1,
                    "total": len(plans),
                    "path": f"{options.input_file.name} [{plan.label}]",
                }
            )

            processed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            started_at = time.perf_counter()
            output_path = build_flat_output_path(
                input_path=options.input_file,
                output_root=output_dir,
                suffix=plan.suffix,
                collision_policy=options.collision_policy,
                skip_existing=options.skip_existing,
            )

            if output_path is None:
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=options.input_file,
                    output_path=None,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=plan.scale,
                    actual_scale="",
                    model_path=plan.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present="",
                    crop_enabled="yes" if options.crop_options.enabled else "no",
                    crop_range=options.crop_options.ratio if options.crop_options.enabled else "",
                    sharpen_enabled="yes" if is_sharpen_enabled(plan.sharpen_strength) else "no",
                    sharpen_strength=plan.sharpen_strength,
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
                        "total": len(plans),
                        "path": f"{options.input_file.name} [{plan.label}]",
                        "result": "skipped",
                    }
                )
                continue

            try:
                prepared = prepare_image(options.input_file, options.crop_options)
                output_size, actual_scale, warning_message = execute_variant(
                    prepared_image=prepared,
                    plan=plan,
                    output_path=output_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    model_cache=model_cache,
                    controller=controller,
                )
                result = "processed_warning" if warning_message else "processed"
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=options.input_file,
                    output_path=output_path,
                    original_size=format_size(prepared.original_width, prepared.original_height),
                    processing_input_size=format_size(prepared.processing_width, prepared.processing_height),
                    output_size=output_size,
                    requested_scale=plan.scale,
                    actual_scale=actual_scale,
                    model_path=plan.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present=prepared.alpha_present,
                    crop_enabled=prepared.crop_enabled,
                    crop_range=prepared.crop_range,
                    sharpen_enabled="yes" if is_sharpen_enabled(plan.sharpen_strength) else "no",
                    sharpen_strength=plan.sharpen_strength,
                    result=result,
                    warning_message=warning_message,
                    error_message="",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                summary.processed += 1
                if warning_message:
                    summary.warnings += 1
                    message_callback(f"比較出力 警告付き完了: {plan.label} / {warning_message}")
                else:
                    message_callback(f"比較出力完了: {plan.label}")
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(plans),
                        "path": f"{options.input_file.name} [{plan.label}]",
                        "result": result,
                    }
                )
            except UserCancelledError:
                raise
            except Exception as exc:
                failed_copy_path = copy_failed_file(options.input_file, options.input_file.parent, failed_root)
                row = make_log_row(
                    processed_at=processed_at,
                    input_path=options.input_file,
                    output_path=output_path,
                    original_size="",
                    processing_input_size="",
                    output_size="",
                    requested_scale=plan.scale,
                    actual_scale="",
                    model_path=plan.model_path,
                    tile_size=options.tile_size,
                    tile_overlap=options.tile_overlap,
                    alpha_present="",
                    crop_enabled="yes" if options.crop_options.enabled else "no",
                    crop_range=options.crop_options.ratio if options.crop_options.enabled else "",
                    sharpen_enabled="yes" if is_sharpen_enabled(plan.sharpen_strength) else "no",
                    sharpen_strength=plan.sharpen_strength,
                    result="failed",
                    warning_message="",
                    error_message=f"{exc} | failed_copy={failed_copy_path}",
                    elapsed_seconds=time.perf_counter() - started_at,
                )
                logger.log_processing(row)
                logger.log_failed(row)
                summary.failed += 1
                message_callback(f"比較出力失敗: {plan.label} -> {exc}")
                progress_callback(
                    {
                        "phase": "finished",
                        "completed": index,
                        "total": len(plans),
                        "path": f"{options.input_file.name} [{plan.label}]",
                        "result": "failed",
                    }
                )
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
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--scale", required=True, type=int, choices=[2, 4])
    parser.add_argument("--tile-size", type=int, default=400)
    parser.add_argument("--tile-overlap", type=int, default=32)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--collision-policy", choices=["skip", "serial"], default="skip")
    parser.add_argument("--sharpen", choices=["none", "weak", "medium", "strong"], default="weak")
    parser.add_argument("--enable-crop", action="store_true")
    parser.add_argument("--crop-ratio", default="4:5")
    parser.add_argument("--test-mode", action="store_true")
    parser.add_argument("--test-limit", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    options = BatchOptions(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        model_path=args.model_path,
        scale=args.scale,
        tile_size=args.tile_size,
        tile_overlap=args.tile_overlap,
        skip_existing=args.skip_existing,
        recursive=args.recursive,
        collision_policy=args.collision_policy,
        sharpen_strength=args.sharpen,
        crop_options=CropOptions(enabled=args.enable_crop, ratio=args.crop_ratio),
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
