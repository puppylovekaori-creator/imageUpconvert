from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


@dataclass(frozen=True, slots=True)
class SharpenPreset:
    radius: float
    percent: int
    threshold: int


SHARPEN_PRESETS = {
    "none": SharpenPreset(radius=0.0, percent=0, threshold=0),
    "weak": SharpenPreset(radius=1.0, percent=70, threshold=2),
    "medium": SharpenPreset(radius=1.3, percent=110, threshold=2),
    "strong": SharpenPreset(radius=1.6, percent=150, threshold=3),
}


def normalize_sharpen_strength(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SHARPEN_PRESETS:
        raise ValueError("シャープ強度は none / weak / medium / strong のいずれかで指定してください。")
    return normalized


def is_sharpen_enabled(value: str) -> bool:
    return normalize_sharpen_strength(value) != "none"


def apply_sharpen(rgb: np.ndarray, strength: str) -> np.ndarray:
    normalized = normalize_sharpen_strength(strength)
    if normalized == "none":
        return rgb

    preset = SHARPEN_PRESETS[normalized]
    rgb_uint8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    sharpened = Image.fromarray(rgb_uint8, mode="RGB").filter(
        ImageFilter.UnsharpMask(
            radius=preset.radius,
            percent=preset.percent,
            threshold=preset.threshold,
        )
    )
    return np.asarray(sharpened, dtype=np.uint8).astype(np.float32) / 255.0
