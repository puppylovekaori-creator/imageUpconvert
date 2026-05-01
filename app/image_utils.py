from __future__ import annotations

import shutil
from pathlib import Path

from PIL import Image

from .app_config import SUPPORTED_EXTENSIONS


def is_supported_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS


def iter_image_files(input_dir: Path, recursive: bool) -> list[Path]:
    globber = input_dir.rglob if recursive else input_dir.glob
    files = [path for path in globber("*") if is_supported_image(path)]
    return sorted(files, key=lambda item: str(item).lower())


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return int(image.width), int(image.height)


def _normalize_for_png(image: Image.Image) -> Image.Image:
    if image.mode in {"RGBA", "LA"}:
        return image.copy()
    if image.mode == "P":
        return image.convert("RGBA")
    if image.mode in {"RGB", "L"}:
        return image.copy()
    return image.convert("RGBA" if "A" in image.getbands() else "RGB")


def save_png_copy(source_path: Path, destination_path: Path) -> None:
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as image:
        normalized = _normalize_for_png(image)
        try:
            normalized.save(destination_path, format="PNG")
        finally:
            normalized.close()


def ensure_valid_output_file(path: Path) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"出力ファイルの作成に失敗しました: {path}")
    read_image_size(path)


def is_same_or_child(base_path: Path, candidate_path: Path) -> bool:
    try:
        candidate_path.relative_to(base_path)
        return True
    except ValueError:
        return False


def validate_input_output_paths(input_dir: Path, output_dir: Path) -> None:
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("入力フォルダと出力フォルダが同じです。元画像保護のため処理を開始できません。")
    if is_same_or_child(input_dir.resolve(), output_dir.resolve()):
        raise ValueError("出力フォルダが入力フォルダ配下です。再帰処理事故防止のため別フォルダを指定してください。")


def build_failed_copy_path(failed_root: Path, relative_input: Path) -> Path:
    destination = failed_root / relative_input
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        return destination
    stem = destination.stem
    suffix = destination.suffix
    counter = 1
    while True:
        candidate = destination.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def copy_to_failed(source_path: Path, failed_root: Path, relative_input: Path) -> Path:
    destination = build_failed_copy_path(failed_root, relative_input)
    shutil.copy2(source_path, destination)
    return destination
