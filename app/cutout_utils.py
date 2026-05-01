from __future__ import annotations

from functools import lru_cache

import numpy as np
from PIL import Image

try:
    from rembg import new_session, remove
except Exception as exc:  # pragma: no cover - depends on local runtime
    REMBG_IMPORT_ERROR = exc
    new_session = None
    remove = None
else:
    REMBG_IMPORT_ERROR = None


@lru_cache(maxsize=1)
def _get_human_session():
    if new_session is None:
        raise RuntimeError(
            "人物切り抜きには rembg が必要です。setup.bat を再実行するか requirements.txt を入れ直してください。"
        ) from REMBG_IMPORT_ERROR
    return new_session("u2net_human_seg")


def apply_person_cutout(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if remove is None:
        raise RuntimeError(
            "人物切り抜きには rembg が必要です。setup.bat を再実行するか requirements.txt を入れ直してください。"
        ) from REMBG_IMPORT_ERROR

    rgb_uint8 = np.clip(np.rint(rgb * 255.0), 0, 255).astype(np.uint8)
    if alpha is not None:
        alpha_uint8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
        source = Image.fromarray(np.dstack((rgb_uint8, alpha_uint8)), mode="RGBA")
    else:
        source = Image.fromarray(rgb_uint8, mode="RGB")

    cut = remove(source, session=_get_human_session())
    rgba = cut.convert("RGBA")
    data = np.asarray(rgba, dtype=np.uint8)

    cut_rgb = data[:, :, :3].astype(np.float32) / 255.0
    cut_alpha = data[:, :, 3].astype(np.float32) / 255.0
    if alpha is not None:
        cut_alpha = np.clip(cut_alpha * alpha, 0.0, 1.0)
    return cut_rgb, cut_alpha
