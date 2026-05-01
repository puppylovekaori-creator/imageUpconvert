from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = APP_ROOT / "config.json"
WHO_VALUE = "GIMP AI Upscale Batch GUI"

SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")

PROCESSING_MODE_UPSCALE_ONLY = "upscale_only"
PROCESSING_MODE_PREPROCESS_ONLY = "gimp_pre_only"
PROCESSING_MODE_PREPROCESS_AND_UPSCALE = "gimp_pre_upscale"

PROCESSING_MODE_CHOICES = (
    ("AI Upscaleのみ", PROCESSING_MODE_UPSCALE_ONLY),
    ("GIMP前処理のみ", PROCESSING_MODE_PREPROCESS_ONLY),
    ("GIMP前処理 + AI Upscale", PROCESSING_MODE_PREPROCESS_AND_UPSCALE),
)
PROCESSING_MODE_VALUES = {value for _, value in PROCESSING_MODE_CHOICES}

UPSCALE_MODELS = (
    "UltraSharp-4x",
    "RealESRGAN_General_x4_v3",
    "realesrgan-x4plus",
    "realesrgan-x4plus-anime",
    "AnimeSharp-4x",
    "realesr-animevideov3-x4",
)
UPSCALE_SCALES = (2, 3, 4)

NOISE_PRESETS = ("off", "weak", "medium", "strong")
UNSHARP_PRESETS = ("off", "weak", "medium", "strong")

PRESET_LABELS = {
    "off": "OFF",
    "weak": "弱",
    "medium": "中",
    "strong": "強",
}

TOKEN_LABELS = {
    "off": "Off",
    "weak": "Weak",
    "medium": "Medium",
    "strong": "Strong",
}

TASK_KIND_COMPARISON = "comparison"
TASK_KIND_TEST_5 = "test_5"
TASK_KIND_TEST_20 = "test_20"
TASK_KIND_BATCH = "batch"

TASK_KIND_LABELS = {
    TASK_KIND_COMPARISON: "1枚比較処理",
    TASK_KIND_TEST_5: "先頭5枚テスト",
    TASK_KIND_TEST_20: "先頭20枚テスト",
    TASK_KIND_BATCH: "一括処理",
}


def _coerce_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


@dataclass(slots=True)
class AppSettings:
    last_input_dir: str = str(APP_ROOT / "input")
    last_output_dir: str = str(APP_ROOT / "output")
    gimp_path: str = ""
    processing_mode: str = PROCESSING_MODE_UPSCALE_ONLY
    model_name: str = "UltraSharp-4x"
    scale: int = 4
    gimp_preprocess_enabled: bool = False
    noise_reduction: str = "weak"
    unsharp_strength: str = "weak"
    include_subfolders: bool = False
    skip_existing: bool = True

    def __post_init__(self) -> None:
        self.last_input_dir = str(self.last_input_dir or (APP_ROOT / "input"))
        self.last_output_dir = str(self.last_output_dir or (APP_ROOT / "output"))
        self.gimp_path = str(self.gimp_path or "")
        if self.processing_mode not in PROCESSING_MODE_VALUES:
            self.processing_mode = PROCESSING_MODE_UPSCALE_ONLY
        if self.model_name not in UPSCALE_MODELS:
            self.model_name = "UltraSharp-4x"
        try:
            parsed_scale = int(self.scale)
        except (TypeError, ValueError):
            parsed_scale = 4
        self.scale = parsed_scale if parsed_scale in UPSCALE_SCALES else 4
        if self.noise_reduction not in NOISE_PRESETS:
            self.noise_reduction = "weak"
        if self.unsharp_strength not in UNSHARP_PRESETS:
            self.unsharp_strength = "weak"
        self.include_subfolders = _coerce_bool(self.include_subfolders, False)
        self.skip_existing = _coerce_bool(self.skip_existing, True)
        self.gimp_preprocess_enabled = self.processing_mode != PROCESSING_MODE_UPSCALE_ONLY


def mode_uses_preprocess(mode: str) -> bool:
    return mode in {
        PROCESSING_MODE_PREPROCESS_ONLY,
        PROCESSING_MODE_PREPROCESS_AND_UPSCALE,
    }


def mode_uses_upscale(mode: str) -> bool:
    return mode in {
        PROCESSING_MODE_UPSCALE_ONLY,
        PROCESSING_MODE_PREPROCESS_AND_UPSCALE,
    }


def get_mode_label(mode: str) -> str:
    for label, value in PROCESSING_MODE_CHOICES:
        if value == mode:
            return label
    return mode


def preset_label(value: str) -> str:
    return PRESET_LABELS.get(value, value)


def preset_token(prefix: str, value: str) -> str:
    return f"{prefix}{TOKEN_LABELS.get(value, value.title())}"


def model_token(model_name: str) -> str:
    token = "".join(char.lower() for char in model_name if char.isalnum())
    return token or "model"


def scale_token(scale: int) -> str:
    return "" if int(scale) == 4 else f"_scale{int(scale)}x"
