from __future__ import annotations

from collections import deque

import numpy as np
from PIL import Image


def bleed_transparent_rgb(rgb: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Fill fully transparent pixels with nearby visible colors to reduce edge fringes."""
    if alpha is None:
        return rgb

    opaque_mask = alpha > 0
    if opaque_mask.all() or not opaque_mask.any():
        return rgb

    height, width = alpha.shape
    filled = np.array(rgb, copy=True)
    known = opaque_mask.copy()

    queue: deque[tuple[int, int]] = deque()
    for y in range(height):
        for x in range(width):
            if not known[y, x]:
                continue
            if (
                (y > 0 and not known[y - 1, x])
                or (y + 1 < height and not known[y + 1, x])
                or (x > 0 and not known[y, x - 1])
                or (x + 1 < width and not known[y, x + 1])
            ):
                queue.append((y, x))

    if not queue:
        return rgb

    while queue:
        y, x = queue.popleft()
        color = filled[y, x]
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if ny < 0 or ny >= height or nx < 0 or nx >= width or known[ny, nx]:
                continue
            filled[ny, nx] = color
            known[ny, nx] = True
            queue.append((ny, nx))

    return filled


def resize_alpha(alpha: np.ndarray, scale: int) -> np.ndarray:
    alpha_uint8 = np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)
    alpha_image = Image.fromarray(alpha_uint8, mode="L")
    resized = alpha_image.resize(
        (alpha_image.width * scale, alpha_image.height * scale),
        resample=Image.Resampling.LANCZOS,
    )
    return np.asarray(resized, dtype=np.float32) / 255.0
