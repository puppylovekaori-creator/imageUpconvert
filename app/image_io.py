from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(slots=True)
class LoadedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    width: int
    height: int
    original_mode: str


def collect_input_files(input_dir: Path, recursive: bool) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in input_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return sorted(files, key=lambda path: str(path).lower())


def load_image(path: Path) -> LoadedImage:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        original_mode = image.mode
        if "A" in image.getbands() or image.mode in {"LA", "PA"}:
            rgba = image.convert("RGBA")
            data = np.asarray(rgba, dtype=np.uint8)
            rgb = data[:, :, :3].astype(np.float32) / 255.0
            alpha = data[:, :, 3].astype(np.float32) / 255.0
            width, height = rgba.size
            return LoadedImage(rgb=rgb, alpha=alpha, width=width, height=height, original_mode=original_mode)

        rgb_image = image.convert("RGB")
        rgb = np.asarray(rgb_image, dtype=np.uint8).astype(np.float32) / 255.0
        width, height = rgb_image.size
        return LoadedImage(rgb=rgb, alpha=None, width=width, height=height, original_mode=original_mode)


def save_png(path: Path, rgb: np.ndarray, alpha: np.ndarray | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    rgb_uint8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        alpha_uint8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
        rgba = np.dstack((rgb_uint8, alpha_uint8))
        Image.fromarray(rgba, mode="RGBA").save(path, format="PNG")
        return

    Image.fromarray(rgb_uint8, mode="RGB").save(path, format="PNG")


def build_output_path(
    input_path: Path,
    input_root: Path,
    output_root: Path,
    suffix: str,
    collision_policy: str,
    skip_existing: bool,
) -> Path | None:
    relative_parent = input_path.parent.relative_to(input_root)
    output_dir = output_root / relative_parent
    candidate = output_dir / f"{input_path.stem}{suffix}.png"

    if not candidate.exists():
        return candidate

    if skip_existing or collision_policy == "skip":
        return None

    if collision_policy != "serial":
        raise ValueError(f"Unsupported collision policy: {collision_policy}")

    serial_index = 1
    while True:
        serial_candidate = output_dir / f"{input_path.stem}{suffix}_{serial_index:02d}.png"
        if not serial_candidate.exists():
            return serial_candidate
        serial_index += 1


def copy_failed_file(input_path: Path, input_root: Path, failed_root: Path) -> Path:
    relative_parent = input_path.parent.relative_to(input_root)
    target_dir = failed_root / relative_parent
    target_dir.mkdir(parents=True, exist_ok=True)

    candidate = target_dir / input_path.name
    if not candidate.exists():
        shutil.copy2(input_path, candidate)
        return candidate

    serial_index = 1
    while True:
        serial_candidate = target_dir / f"{input_path.stem}_{serial_index:02d}{input_path.suffix}"
        if not serial_candidate.exists():
            shutil.copy2(input_path, serial_candidate)
            return serial_candidate
        serial_index += 1
