"""Frame geometry shared by every live-capture feed.

The fly retina expects one fixed portrait frame size (360x640). Both capture
backends start from an arbitrarily shaped source image -- Chromium's compositor
screenshot in ``shorts.py``, an ``adb screencap`` PNG in ``bumble.py`` -- so the
crop-then-resize step lives here once instead of once per backend.
"""

from __future__ import annotations

import numpy as np
from PIL import Image


def center_crop_resize(img: Image.Image, width: int, height: int) -> np.ndarray:
    """Center-crop ``img`` to the target aspect, then resize to ``width``x``height``.

    Cropping before resizing (rather than stretching) matters for the fly: a
    squashed frame hands the T4/T5 motion detectors horizontal or vertical
    energy that the source never contained.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    iw, ih = img.size
    target_aspect = width / height
    src_aspect = iw / max(ih, 1)
    if src_aspect > target_aspect:
        new_w = max(1, int(ih * target_aspect))
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    else:
        new_h = max(1, int(iw / target_aspect))
        top = (ih - new_h) // 2
        img = img.crop((0, top, iw, top + new_h))
    img = img.resize((width, height), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)
