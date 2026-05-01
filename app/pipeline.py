from __future__ import annotations

import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .app_config import (
    TASK_KIND_BATCH,
    TASK_KIND_COMPARISON,
    TASK_KIND_LABELS,
    TASK_KIND_TEST_20,
    TASK_KIND_TEST_5,
    WHO_VALUE,
    AppSettings,
    get_mode_label,
    mode_uses_preprocess,
    mode_uses_upscale,
    model_token,
    preset_token,
    scale_token,
)
from .gimp_runner import (
    GimpInfo,
    NoiseReductionSettings,
    UnsharpMaskSettings,
    describe_noise_settings,
    describe_unsharp_settings,
    detect_gimp_info,
    run_gimp_preprocess,
    validate_gimp_path,
)
from .image_utils import (
    copy_to_failed,
    ensure_valid_output_file,
    iter_image_files,
    read_image_size,
    save_png_copy,
    validate_input_output_paths,
)
from .log_utils import CsvLogger
from .process_utils import StopRequestedError, ensure_not_stopped
from .upscale_runner import UpscalePlan, prepare_upscale_plan, run_upscale


@dataclass(slots=True)
class PipelineOptions:
    input_dir: Path
    output_dir: Path
    gimp_path: Path
    processing_mode: str
    model_name: str
    scale: int
    noise_reduction: str
    unsharp_strength: str
    include_subfolders: bool
    skip_existing: bool


@dataclass(slots=True)
class TaskSummary:
    task_kind: str
    output_dir: str
    total: int
    completed: int
    failed: int
    skipped: int
    stopped: bool
    elapsed_seconds: float


@dataclass(slots=True)
class FileProcessResult:
    status: str
    output_path: Path | None
    output_size: tuple[int, int] | None
    execution_method: str
    processing_seconds: float


def settings_to_pipeline_options(settings: AppSettings) -> PipelineOptions:
    return PipelineOptions(
        input_dir=Path(settings.last_input_dir).expanduser(),
        output_dir=Path(settings.last_output_dir).expanduser(),
        gimp_path=Path(settings.gimp_path).expanduser(),
        processing_mode=settings.processing_mode,
        model_name=settings.model_name,
        scale=settings.scale,
        noise_reduction=settings.noise_reduction,
        unsharp_strength=settings.unsharp_strength,
        include_subfolders=settings.include_subfolders,
        skip_existing=settings.skip_existing,
    )


def _variant_suffix(options: PipelineOptions, *, force_preprocess: bool, force_upscale: bool) -> str:
    suffix_parts: list[str] = []
    if force_preprocess:
        suffix_parts.append(
            "pre_"
            + preset_token("noise", options.noise_reduction)
            + "_"
            + preset_token("sharp", options.unsharp_strength)
        )
    if force_upscale:
        suffix_parts.append(f"upscale_{model_token(options.model_name)}{scale_token(options.scale)}")
    return "_" + "_".join(suffix_parts) if suffix_parts else ""


def _build_batch_output_path(
    source_path: Path,
    *,
    input_dir: Path,
    output_root: Path,
    options: PipelineOptions,
) -> Path:
    relative_input = source_path.relative_to(input_dir)
    destination_dir = output_root / relative_input.parent
    suffix = _variant_suffix(
        options,
        force_preprocess=mode_uses_preprocess(options.processing_mode),
        force_upscale=mode_uses_upscale(options.processing_mode),
    )
    return destination_dir / f"{source_path.stem}{suffix}.png"


def _build_comparison_output_path(source_path: Path, *, output_root: Path, suffix: str) -> Path:
    return output_root / f"{source_path.stem}_{suffix}.png"


def _log_row(
    *,
    task_kind: str,
    status: str,
    source_path: Path,
    relative_input: Path,
    output_path: Path | None,
    original_size: tuple[int, int] | None,
    output_size: tuple[int, int] | None,
    options: PipelineOptions,
    gimp_info: GimpInfo,
    execution_method: str,
    error_message: str,
    processing_seconds: float,
    gimp_pre_enabled: bool,
) -> dict[str, str]:
    return {
        "who": WHO_VALUE,
        "processed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "task_kind": task_kind,
        "status": status,
        "processing_mode": get_mode_label(options.processing_mode),
        "input_file": str(source_path),
        "output_file": str(output_path) if output_path is not None else "",
        "relative_input": str(relative_input),
        "original_image_size": f"{original_size[0]}x{original_size[1]}" if original_size else "",
        "output_image_size": f"{output_size[0]}x{output_size[1]}" if output_size else "",
        "gimp_path": str(options.gimp_path),
        "gimp_version": gimp_info.version_text,
        "execution_method": execution_method,
        "model": options.model_name,
        "scale": f"{int(options.scale)}x",
        "gimp_pre_enabled": "ON" if gimp_pre_enabled else "OFF",
        "noise_reduction_setting": describe_noise_settings(NoiseReductionSettings(options.noise_reduction)),
        "unsharp_setting": describe_unsharp_settings(UnsharpMaskSettings(options.unsharp_strength)),
        "error_message": error_message,
        "processing_time_seconds": f"{processing_seconds:.3f}",
    }


def _task_output_root(base_output_dir: Path, task_kind: str) -> Path:
    if task_kind == TASK_KIND_COMPARISON:
        return base_output_dir / "comparison"
    if task_kind == TASK_KIND_TEST_5:
        return base_output_dir / "test_5"
    if task_kind == TASK_KIND_TEST_20:
        return base_output_dir / "test_20"
    return base_output_dir


def _emit_progress(
    progress_callback: Callable[[dict], None] | None,
    *,
    total: int,
    completed: int,
    failed: int,
    skipped: int,
    current_file: str,
    started_at: float,
) -> None:
    if progress_callback is None:
        return
    done = completed + failed + skipped
    elapsed = max(time.monotonic() - started_at, 0.0001)
    rate_per_minute = (done / elapsed) * 60.0
    remaining = max(total - done, 0)
    eta_seconds = int(round((remaining / done) * elapsed)) if done > 0 else 0
    progress_callback(
        {
            "total": total,
            "completed": completed,
            "failed": failed,
            "skipped": skipped,
            "current_file": current_file,
            "rate_per_minute": rate_per_minute,
            "eta_seconds": eta_seconds,
            "elapsed_seconds": elapsed,
        }
    )


def _message(message_callback: Callable[[str], None] | None, text: str) -> None:
    if message_callback is not None:
        message_callback(text)


def _preprocess_to_temp(
    options: PipelineOptions,
    source_path: Path,
    temp_dir: Path,
    *,
    gimp_info: GimpInfo,
    stop_check: Callable[[], bool] | None,
) -> float:
    preprocessed_path = temp_dir / "gimp_pre.png"
    result = run_gimp_preprocess(
        gimp_path=options.gimp_path,
        input_path=source_path,
        output_path=preprocessed_path,
        noise_settings=NoiseReductionSettings(options.noise_reduction),
        unsharp_settings=UnsharpMaskSettings(options.unsharp_strength),
        gimp_info=gimp_info,
        stop_check=stop_check,
    )
    ensure_valid_output_file(preprocessed_path)
    return result.duration_seconds


def _execute_pipeline(
    options: PipelineOptions,
    source_path: Path,
    final_output_path: Path,
    *,
    gimp_info: GimpInfo,
    upscale_plan: UpscalePlan | None,
    force_preprocess: bool,
    force_upscale: bool,
    stop_check: Callable[[], bool] | None,
    allow_skip_existing: bool,
) -> FileProcessResult:
    started_at = time.monotonic()
    if allow_skip_existing and final_output_path.exists():
        try:
            size = read_image_size(final_output_path)
        except Exception:
            size = None
        return FileProcessResult(
            status="skipped",
            output_path=final_output_path,
            output_size=size,
            execution_method="other",
            processing_seconds=time.monotonic() - started_at,
        )

    final_output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = final_output_path.parent / "_temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="imageupconvert_", dir=temp_root) as temp_dir_text:
        temp_dir = Path(temp_dir_text)
        current_source = source_path
        execution_method = "other"
        total_duration = 0.0

        if force_preprocess:
            ensure_not_stopped(stop_check)
            total_duration += _preprocess_to_temp(
                options,
                current_source,
                temp_dir,
                gimp_info=gimp_info,
                stop_check=stop_check,
            )
            current_source = temp_dir / "gimp_pre.png"

        if force_upscale:
            if upscale_plan is None:
                raise RuntimeError("AI Upscale の実行計画が未準備です。")
            ensure_not_stopped(stop_check)
            upscaled_path = temp_dir / "upscaled.png"
            upscale_result = run_upscale(
                upscale_plan,
                gimp_path=options.gimp_path,
                input_path=current_source,
                output_path=upscaled_path,
                stop_check=stop_check,
            )
            current_source = upscale_result.output_path
            execution_method = upscale_result.execution_method
            total_duration += upscale_result.duration_seconds

        ensure_valid_output_file(current_source)
        shutil.move(str(current_source), str(final_output_path))
        ensure_valid_output_file(final_output_path)
        return FileProcessResult(
            status="success",
            output_path=final_output_path,
            output_size=read_image_size(final_output_path),
            execution_method=execution_method,
            processing_seconds=max(total_duration, time.monotonic() - started_at),
        )


def _process_single_batch_file(
    logger: CsvLogger,
    *,
    task_kind: str,
    source_path: Path,
    options: PipelineOptions,
    output_root: Path,
    gimp_info: GimpInfo,
    upscale_plan: UpscalePlan | None,
    stop_check: Callable[[], bool] | None,
) -> str:
    relative_input = source_path.relative_to(options.input_dir)
    original_size = read_image_size(source_path)
    output_path = _build_batch_output_path(
        source_path,
        input_dir=options.input_dir,
        output_root=output_root,
        options=options,
    )
    try:
        result = _execute_pipeline(
            options,
            source_path,
            output_path,
            gimp_info=gimp_info,
            upscale_plan=upscale_plan,
            force_preprocess=mode_uses_preprocess(options.processing_mode),
            force_upscale=mode_uses_upscale(options.processing_mode),
            stop_check=stop_check,
            allow_skip_existing=options.skip_existing,
        )
        logger.log_processing(
            _log_row(
                task_kind=task_kind,
                status=result.status,
                source_path=source_path,
                relative_input=relative_input,
                output_path=result.output_path,
                original_size=original_size,
                output_size=result.output_size,
                options=options,
                gimp_info=gimp_info,
                execution_method=result.execution_method,
                error_message="",
                processing_seconds=result.processing_seconds,
                gimp_pre_enabled=mode_uses_preprocess(options.processing_mode),
            )
        )
        return result.status
    except StopRequestedError:
        logger.log_processing(
            _log_row(
                task_kind=task_kind,
                status="stopped",
                source_path=source_path,
                relative_input=relative_input,
                output_path=None,
                original_size=original_size,
                output_size=None,
                options=options,
                gimp_info=gimp_info,
                execution_method="other",
                error_message="停止要求により中断しました。",
                processing_seconds=0.0,
                gimp_pre_enabled=mode_uses_preprocess(options.processing_mode),
            )
        )
        raise
    except Exception as exc:
        failed_root = output_root / "failed"
        copy_to_failed(source_path, failed_root, relative_input)
        row = _log_row(
            task_kind=task_kind,
            status="failed",
            source_path=source_path,
            relative_input=relative_input,
            output_path=output_path if output_path.exists() else None,
            original_size=original_size,
            output_size=None,
            options=options,
            gimp_info=gimp_info,
            execution_method="other",
            error_message=str(exc),
            processing_seconds=0.0,
            gimp_pre_enabled=mode_uses_preprocess(options.processing_mode),
        )
        logger.log_processing(row)
        logger.log_failed(row)
        return "failed"


def _run_comparison_task(
    options: PipelineOptions,
    *,
    files: list[Path],
    output_root: Path,
    gimp_info: GimpInfo,
    upscale_plan: UpscalePlan,
    progress_callback: Callable[[dict], None] | None,
    message_callback: Callable[[str], None] | None,
    stop_check: Callable[[], bool] | None,
) -> TaskSummary:
    source_path = files[0]
    original_size = read_image_size(source_path)
    relative_input = source_path.relative_to(options.input_dir)
    model_suffix = f"{model_token(options.model_name)}{scale_token(options.scale).replace('_', '')}"
    pre_suffix = f"{preset_token('noise', options.noise_reduction)}_{preset_token('sharp', options.unsharp_strength)}"
    variants = [
        ("original", False, False, _build_comparison_output_path(source_path, output_root=output_root, suffix="original")),
        ("gimp_pre", True, False, _build_comparison_output_path(source_path, output_root=output_root, suffix=f"gimp_pre_{pre_suffix}")),
        ("upscale_only", False, True, _build_comparison_output_path(source_path, output_root=output_root, suffix=f"upscale_{model_suffix}")),
        (
            "gimp_pre_upscale",
            True,
            True,
            _build_comparison_output_path(source_path, output_root=output_root, suffix=f"gimp_pre_{pre_suffix}_upscale_{model_suffix}"),
        ),
    ]

    logger = CsvLogger(output_root)
    started_at = time.monotonic()
    completed = 0
    failed = 0
    try:
        for label, uses_preprocess, uses_upscale, destination in variants:
            ensure_not_stopped(stop_check)
            _message(message_callback, f"比較処理: {label}")
            if label == "original":
                save_png_copy(source_path, destination)
                output_size = read_image_size(destination)
                logger.log_processing(
                    _log_row(
                        task_kind=TASK_KIND_COMPARISON,
                        status="success",
                        source_path=source_path,
                        relative_input=relative_input,
                        output_path=destination,
                        original_size=original_size,
                        output_size=output_size,
                        options=options,
                        gimp_info=gimp_info,
                        execution_method="other",
                        error_message="",
                        processing_seconds=0.0,
                        gimp_pre_enabled=False,
                    )
                )
                completed += 1
            else:
                try:
                    result = _execute_pipeline(
                        options,
                        source_path,
                        destination,
                        gimp_info=gimp_info,
                        upscale_plan=upscale_plan if uses_upscale else None,
                        force_preprocess=uses_preprocess,
                        force_upscale=uses_upscale,
                        stop_check=stop_check,
                        allow_skip_existing=False,
                    )
                    logger.log_processing(
                        _log_row(
                            task_kind=TASK_KIND_COMPARISON,
                            status=result.status,
                            source_path=source_path,
                            relative_input=relative_input,
                            output_path=result.output_path,
                            original_size=original_size,
                            output_size=result.output_size,
                            options=options,
                            gimp_info=gimp_info,
                            execution_method=result.execution_method,
                            error_message="",
                            processing_seconds=result.processing_seconds,
                            gimp_pre_enabled=uses_preprocess,
                        )
                    )
                    completed += 1
                except Exception as exc:
                    failed += 1
                    row = _log_row(
                        task_kind=TASK_KIND_COMPARISON,
                        status="failed",
                        source_path=source_path,
                        relative_input=relative_input,
                        output_path=destination if destination.exists() else None,
                        original_size=original_size,
                        output_size=None,
                        options=options,
                        gimp_info=gimp_info,
                        execution_method="other",
                        error_message=str(exc),
                        processing_seconds=0.0,
                        gimp_pre_enabled=uses_preprocess,
                    )
                    logger.log_processing(row)
                    logger.log_failed(row)
            _emit_progress(
                progress_callback,
                total=len(variants),
                completed=completed,
                failed=failed,
                skipped=0,
                current_file=source_path.name,
                started_at=started_at,
            )
        return TaskSummary(
            task_kind=TASK_KIND_COMPARISON,
            output_dir=str(output_root),
            total=len(variants),
            completed=completed,
            failed=failed,
            skipped=0,
            stopped=False,
            elapsed_seconds=time.monotonic() - started_at,
        )
    finally:
        logger.close()


def run_task(
    task_kind: str,
    options: PipelineOptions,
    *,
    progress_callback: Callable[[dict], None] | None = None,
    message_callback: Callable[[str], None] | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> TaskSummary:
    validate_gimp_path(options.gimp_path)
    if not options.input_dir.exists() or not options.input_dir.is_dir():
        raise ValueError(f"入力フォルダが見つかりません: {options.input_dir}")
    validate_input_output_paths(options.input_dir, options.output_dir)

    files = iter_image_files(options.input_dir, options.include_subfolders)
    if not files:
        raise ValueError("入力フォルダに対象画像がありません。")

    if task_kind == TASK_KIND_COMPARISON:
        target_files = files[:1]
    elif task_kind == TASK_KIND_TEST_5:
        target_files = files[:5]
    elif task_kind == TASK_KIND_TEST_20:
        target_files = files[:20]
    else:
        target_files = files

    gimp_info = detect_gimp_info(options.gimp_path)
    output_root = _task_output_root(options.output_dir, task_kind)
    output_root.mkdir(parents=True, exist_ok=True)
    upscale_plan: UpscalePlan | None = None
    if task_kind == TASK_KIND_COMPARISON or mode_uses_upscale(options.processing_mode):
        upscale_plan = prepare_upscale_plan(gimp_info, options.model_name, options.scale)

    _message(message_callback, f"{TASK_KIND_LABELS[task_kind]}を開始します。件数: {len(target_files)}")
    if task_kind == TASK_KIND_COMPARISON:
        return _run_comparison_task(
            options,
            files=target_files,
            output_root=output_root,
            gimp_info=gimp_info,
            upscale_plan=upscale_plan,
            progress_callback=progress_callback,
            message_callback=message_callback,
            stop_check=stop_check,
        )

    logger = CsvLogger(output_root)
    started_at = time.monotonic()
    completed = 0
    failed = 0
    skipped = 0
    stopped = False
    try:
        _emit_progress(
            progress_callback,
            total=len(target_files),
            completed=0,
            failed=0,
            skipped=0,
            current_file="",
            started_at=started_at,
        )
        for source_path in target_files:
            ensure_not_stopped(stop_check)
            _message(message_callback, f"処理中: {source_path.name}")
            status = _process_single_batch_file(
                logger,
                task_kind=task_kind,
                source_path=source_path,
                options=options,
                output_root=output_root,
                gimp_info=gimp_info,
                upscale_plan=upscale_plan,
                stop_check=stop_check,
            )
            if status == "success":
                completed += 1
            elif status == "failed":
                failed += 1
            elif status == "skipped":
                skipped += 1
            _emit_progress(
                progress_callback,
                total=len(target_files),
                completed=completed,
                failed=failed,
                skipped=skipped,
                current_file=source_path.name,
                started_at=started_at,
            )
    except StopRequestedError:
        stopped = True
        _message(message_callback, "停止要求を受け付けました。現在のファイル単位で安全停止しました。")
    finally:
        logger.close()

    return TaskSummary(
        task_kind=task_kind,
        output_dir=str(output_root),
        total=len(target_files),
        completed=completed,
        failed=failed,
        skipped=skipped,
        stopped=stopped,
        elapsed_seconds=time.monotonic() - started_at,
    )
