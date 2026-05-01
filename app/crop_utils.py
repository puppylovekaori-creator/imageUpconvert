from __future__ import annotations

from dataclasses import dataclass

import numpy as np


PREVIEW_RANGE_CHOICES = ("full", "center", "face_near")


@dataclass(slots=True)
class CropResult:
    rgb: np.ndarray
    alpha: np.ndarray | None
    left: int
    top: int
    right: int
    bottom: int
    ratio_text: str
    mode: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def format_range(self) -> str:
        return (
            f"mode={self.mode},x={self.left},y={self.top},w={self.width},h={self.height},ratio={self.ratio_text}"
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


def _compute_crop_size(width: int, height: int, ratio_text: str) -> tuple[int, int]:
    ratio_width, ratio_height = parse_ratio_text(ratio_text)
    target_ratio = ratio_width / ratio_height
    source_ratio = width / height

    if source_ratio > target_ratio:
        crop_height = height
        crop_width = max(1, min(width, int(round(crop_height * target_ratio))))
    else:
        crop_width = width
        crop_height = max(1, min(height, int(round(crop_width / target_ratio))))

    return crop_width, crop_height


def _crop_with_anchor(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    *,
    ratio_text: str,
    mode: str,
    anchor_x: float,
    anchor_y: float,
) -> CropResult:
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("RGB 画像の形状が不正です。")

    source_height, source_width = rgb.shape[:2]
    crop_width, crop_height = _compute_crop_size(source_width, source_height, ratio_text)

    target_center_x = source_width * anchor_x
    target_center_y = source_height * anchor_y
    left = int(round(target_center_x - (crop_width / 2.0)))
    top = int(round(target_center_y - (crop_height / 2.0)))

    left = min(max(left, 0), max(source_width - crop_width, 0))
    top = min(max(top, 0), max(source_height - crop_height, 0))
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
        mode=mode,
    )


def center_crop_by_ratio(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    ratio_text: str,
) -> CropResult:
    return _crop_with_anchor(
        rgb,
        alpha,
        ratio_text=ratio_text,
        mode="center",
        anchor_x=0.5,
        anchor_y=0.5,
    )


def face_near_crop_by_ratio(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    ratio_text: str,
) -> CropResult:
    # 顔検出ではなく、人物写真で顔が上寄りにあることを想定した簡易アンカー。
    return _crop_with_anchor(
        rgb,
        alpha,
        ratio_text=ratio_text,
        mode="face_near",
        anchor_x=0.5,
        anchor_y=0.34,
    )


def apply_preview_crop_mode(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    *,
    mode: str,
    ratio_text: str,
) -> CropResult:
    normalized = (mode or "full").strip().lower()
    if normalized == "full":
        height, width = rgb.shape[:2]
        return CropResult(
            rgb=rgb,
            alpha=alpha,
            left=0,
            top=0,
            right=width,
            bottom=height,
            ratio_text=ratio_text,
            mode="full",
        )
    if normalized == "center":
        return center_crop_by_ratio(rgb, alpha, ratio_text)
    if normalized == "face_near":
        return face_near_crop_by_ratio(rgb, alpha, ratio_text)
    raise ValueError(f"未対応のプレビュー範囲です: {mode}")
