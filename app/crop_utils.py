from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class CropResult:
    rgb: np.ndarray
    alpha: np.ndarray | None
    left: int
    top: int
    right: int
    bottom: int
    ratio_text: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def format_range(self) -> str:
        return (
            f"x={self.left},y={self.top},w={self.width},h={self.height},ratio={self.ratio_text}"
        )


def parse_ratio_text(ratio_text: str) -> tuple[int, int]:
    parts = ratio_text.strip().split(":")
    if len(parts) != 2:
        raise ValueError("crop 比率は 1:1 や 4:5 の形式で指定してください。")

    try:
        width_ratio = int(parts[0])
        height_ratio = int(parts[1])
    except ValueError as exc:
        raise ValueError("crop 比率は整数の組み合わせで指定してください。") from exc

    if width_ratio <= 0 or height_ratio <= 0:
        raise ValueError("crop 比率は 1 以上で指定してください。")

    return width_ratio, height_ratio


def center_crop_by_ratio(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    ratio_text: str,
) -> CropResult:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB 画像の形状が不正です。")

    source_height, source_width = rgb.shape[:2]
    ratio_width, ratio_height = parse_ratio_text(ratio_text)
    target_ratio = ratio_width / ratio_height
    source_ratio = source_width / source_height

    if source_ratio > target_ratio:
        crop_height = source_height
        crop_width = max(1, min(source_width, int(round(crop_height * target_ratio))))
    else:
        crop_width = source_width
        crop_height = max(1, min(source_height, int(round(crop_width / target_ratio))))

    left = max((source_width - crop_width) // 2, 0)
    top = max((source_height - crop_height) // 2, 0)
    right = left + crop_width
    bottom = top + crop_height

    cropped_rgb = rgb[top:bottom, left:right, :]
    cropped_alpha = alpha[top:bottom, left:right] if alpha is not None else None
    return CropResult(
        rgb=cropped_rgb,
        alpha=cropped_alpha,
        left=left,
        top=top,
        right=right,
        bottom=bottom,
        ratio_text=ratio_text,
    )
