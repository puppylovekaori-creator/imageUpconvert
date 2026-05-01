from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageFilter


SHARPEN_METHODS = ("unsharp", "highpass", "clarity", "edge")
SHARPEN_STRENGTHS = ("none", "weak", "medium", "strong")
SHARPEN_METHOD_LABELS = {
    "unsharp": "Unsharp Mask",
    "highpass": "High Pass Sharpen",
    "clarity": "Local Contrast / Clarity",
    "edge": "Edge-only Sharpen",
}
SHARPEN_METHOD_LABELS_JA = {
    "unsharp": "Unsharp Mask（標準）",
    "highpass": "High Pass Sharpen（高域強調）",
    "clarity": "Local Contrast / Clarity（局所コントラスト）",
    "edge": "Edge-only Sharpen（輪郭限定）",
}
SHARPEN_STRENGTH_LABELS_JA = {
    "none": "なし",
    "weak": "弱",
    "medium": "中",
    "strong": "強",
}


@dataclass(frozen=True, slots=True)
class UnsharpPreset:
    radius: float
    percent: int
    threshold: int


@dataclass(frozen=True, slots=True)
class FloatSharpenPreset:
    radius: float
    amount: float
    threshold: float
    clamp: float


@dataclass(frozen=True, slots=True)
class EdgeSharpenPreset:
    radius: float
    percent: int
    threshold: int
    mask_threshold: float
    mask_blur_radius: float
    blend: float


UNSHARP_PRESETS = {
    "weak": UnsharpPreset(radius=0.9, percent=55, threshold=3),
    "medium": UnsharpPreset(radius=1.2, percent=95, threshold=3),
    "strong": UnsharpPreset(radius=1.5, percent=130, threshold=4),
}

HIGHPASS_PRESETS = {
    "weak": FloatSharpenPreset(radius=1.2, amount=0.22, threshold=0.012, clamp=0.060),
    "medium": FloatSharpenPreset(radius=1.6, amount=0.31, threshold=0.009, clamp=0.085),
    "strong": FloatSharpenPreset(radius=2.0, amount=0.38, threshold=0.007, clamp=0.110),
}

CLARITY_PRESETS = {
    "weak": FloatSharpenPreset(radius=2.4, amount=0.14, threshold=0.010, clamp=0.045),
    "medium": FloatSharpenPreset(radius=3.5, amount=0.20, threshold=0.008, clamp=0.060),
    "strong": FloatSharpenPreset(radius=4.6, amount=0.26, threshold=0.006, clamp=0.080),
}

EDGE_PRESETS = {
    "weak": EdgeSharpenPreset(
        radius=0.9,
        percent=70,
        threshold=3,
        mask_threshold=0.060,
        mask_blur_radius=0.8,
        blend=0.65,
    ),
    "medium": EdgeSharpenPreset(
        radius=1.1,
        percent=105,
        threshold=3,
        mask_threshold=0.050,
        mask_blur_radius=1.0,
        blend=0.85,
    ),
    "strong": EdgeSharpenPreset(
        radius=1.4,
        percent=135,
        threshold=4,
        mask_threshold=0.042,
        mask_blur_radius=1.2,
        blend=1.00,
    ),
}


def normalize_sharpen_method(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SHARPEN_METHODS:
        raise ValueError(
            "シャープ方式は unsharp / highpass / clarity / edge のいずれかで指定してください。"
        )
    return normalized


def normalize_sharpen_strength(value: str) -> str:
    normalized = value.strip().lower()
    if normalized not in SHARPEN_STRENGTHS:
        raise ValueError("シャープ強度は none / weak / medium / strong のいずれかで指定してください。")
    return normalized


def get_sharpen_method_label(value: str) -> str:
    normalized = normalize_sharpen_method(value)
    return SHARPEN_METHOD_LABELS[normalized]


def get_sharpen_method_label_ja(value: str) -> str:
    normalized = normalize_sharpen_method(value)
    return SHARPEN_METHOD_LABELS_JA[normalized]


def get_sharpen_strength_label_ja(value: str) -> str:
    normalized = normalize_sharpen_strength(value)
    return SHARPEN_STRENGTH_LABELS_JA[normalized]


def list_sharpen_methods() -> tuple[str, ...]:
    return SHARPEN_METHODS


def list_sharpen_strengths() -> tuple[str, ...]:
    return SHARPEN_STRENGTHS


def is_sharpen_enabled(method: str, strength: str) -> bool:
    normalize_sharpen_method(method)
    return normalize_sharpen_strength(strength) != "none"


def get_effective_sharpen_method(method: str, strength: str) -> str:
    normalized_strength = normalize_sharpen_strength(strength)
    if normalized_strength == "none":
        return "none"
    return normalize_sharpen_method(method)


def build_sharpen_suffix(method: str, strength: str) -> str:
    effective_method = get_effective_sharpen_method(method, strength)
    normalized_strength = normalize_sharpen_strength(strength)
    if normalized_strength == "none":
        return "_sharp_none"
    return f"_sharp_{effective_method}_{normalized_strength}"


def pil_rgb_to_numpy(image: Image.Image) -> np.ndarray:
    return np.asarray(image, dtype=np.uint8).astype(np.float32) / 255.0


def numpy_rgb_to_pil(rgb: np.ndarray) -> Image.Image:
    rgb_uint8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    return Image.fromarray(rgb_uint8, mode="RGB")


def rgb_to_luma(rgb: np.ndarray) -> np.ndarray:
    return (
        (rgb[:, :, 0] * 0.299)
        + (rgb[:, :, 1] * 0.587)
        + (rgb[:, :, 2] * 0.114)
    ).astype(np.float32)


def blur_rgb(rgb: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return rgb
    return pil_rgb_to_numpy(numpy_rgb_to_pil(rgb).filter(ImageFilter.GaussianBlur(radius)))


def blur_gray(gray: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return gray
    gray_uint8 = np.clip(np.rint(gray * 255.0), 0, 255).astype(np.uint8)
    blurred = Image.fromarray(gray_uint8, mode="L").filter(ImageFilter.GaussianBlur(radius))
    return np.asarray(blurred, dtype=np.uint8).astype(np.float32) / 255.0


def smoothstep(x: np.ndarray, edge0: float, edge1: float) -> np.ndarray:
    if edge1 <= edge0:
        return (x >= edge0).astype(np.float32)
    t = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def apply_luma_delta(rgb: np.ndarray, delta: np.ndarray) -> np.ndarray:
    return np.clip(rgb + delta[:, :, None], 0.0, 1.0)


def compute_gradient_magnitude(gray: np.ndarray) -> np.ndarray:
    padded = np.pad(gray, 1, mode="edge")
    gx = (
        padded[:-2, 2:] + (2.0 * padded[1:-1, 2:]) + padded[2:, 2:]
        - padded[:-2, :-2] - (2.0 * padded[1:-1, :-2]) - padded[2:, :-2]
    )
    gy = (
        padded[2:, :-2] + (2.0 * padded[2:, 1:-1]) + padded[2:, 2:]
        - padded[:-2, :-2] - (2.0 * padded[:-2, 1:-1]) - padded[:-2, 2:]
    )
    return np.sqrt((gx * gx) + (gy * gy)).astype(np.float32)


def apply_unsharp(rgb: np.ndarray, strength: str) -> np.ndarray:
    normalized_strength = normalize_sharpen_strength(strength)
    if normalized_strength == "none":
        return rgb

    preset = UNSHARP_PRESETS[normalized_strength]
    sharpened = numpy_rgb_to_pil(rgb).filter(
        ImageFilter.UnsharpMask(
            radius=preset.radius,
            percent=preset.percent,
            threshold=preset.threshold,
        )
    )
    return pil_rgb_to_numpy(sharpened)


def apply_highpass(rgb: np.ndarray, strength: str) -> np.ndarray:
    preset = HIGHPASS_PRESETS[normalize_sharpen_strength(strength)]
    blurred = blur_rgb(rgb, preset.radius)
    detail = rgb - blurred
    detail_magnitude = np.mean(np.abs(detail), axis=2)
    mask = smoothstep(detail_magnitude, preset.threshold, preset.threshold * 4.5)
    delta = np.clip(detail * (preset.amount * mask[:, :, None]), -preset.clamp, preset.clamp)
    return np.clip(rgb + delta, 0.0, 1.0)


def apply_clarity(rgb: np.ndarray, strength: str) -> np.ndarray:
    preset = CLARITY_PRESETS[normalize_sharpen_strength(strength)]
    luma = rgb_to_luma(rgb)
    blurred_luma = blur_gray(luma, preset.radius)
    detail = luma - blurred_luma
    detail_mask = smoothstep(np.abs(detail), preset.threshold, preset.threshold * 5.0)
    midtone_mask = np.clip(1.0 - (np.abs(luma - 0.5) / 0.58), 0.0, 1.0)
    delta = np.clip(detail * preset.amount * detail_mask * midtone_mask, -preset.clamp, preset.clamp)
    return apply_luma_delta(rgb, delta.astype(np.float32))


def apply_edge_only(rgb: np.ndarray, strength: str) -> np.ndarray:
    preset = EDGE_PRESETS[normalize_sharpen_strength(strength)]
    sharpened = numpy_rgb_to_pil(rgb).filter(
        ImageFilter.UnsharpMask(
            radius=preset.radius,
            percent=preset.percent,
            threshold=preset.threshold,
        )
    )
    sharpened_rgb = pil_rgb_to_numpy(sharpened)
    luma = rgb_to_luma(rgb)
    edge_mask = compute_gradient_magnitude(luma)
    edge_mask = smoothstep(edge_mask, preset.mask_threshold, preset.mask_threshold * 4.5)
    edge_mask = blur_gray(edge_mask, preset.mask_blur_radius)
    delta = (sharpened_rgb - rgb) * (edge_mask[:, :, None] * preset.blend)
    return np.clip(rgb + delta, 0.0, 1.0)


def apply_sharpen(rgb: np.ndarray, method: str, strength: str) -> np.ndarray:
    effective_method = get_effective_sharpen_method(method, strength)
    normalized_strength = normalize_sharpen_strength(strength)

    if normalized_strength == "none" or effective_method == "none":
        return rgb

    if effective_method == "unsharp":
        return apply_unsharp(rgb, normalized_strength)
    if effective_method == "highpass":
        return apply_highpass(rgb, normalized_strength)
    if effective_method == "clarity":
        return apply_clarity(rgb, normalized_strength)
    if effective_method == "edge":
        return apply_edge_only(rgb, normalized_strength)

    raise ValueError(f"未対応のシャープ方式です: {method}")
