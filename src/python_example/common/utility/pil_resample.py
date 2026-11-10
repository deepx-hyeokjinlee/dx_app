"""
Pillow's antialiased resize (Image.resize with BILINEAR / BICUBIC), byte for
byte, in numpy -- the Python twin of cpp_example/common/utility/pil_resample.hpp.

The model zoo's compile JSON gives most classifiers the torchvision evaluation
transform, Resize(shorter side S) + CenterCrop, and torchvision resizes a PIL
image with Pillow. Pillow is not a dependency of this tree, so its arithmetic
(src/libImaging/Resample.c) is reproduced: a separable convolution whose
kernel widens with the shrink factor, horizontal pass then vertical, 22-bit
fixed-point weights, a uint8 image in between.
"""

import math
from functools import lru_cache
from typing import Tuple

import numpy as np

_PRECISION_BITS = 32 - 8 - 2


def _bilinear(x: np.ndarray) -> np.ndarray:
    x = np.abs(x)
    return np.where(x < 1.0, 1.0 - x, 0.0)


def _bicubic(x: np.ndarray) -> np.ndarray:
    a = -0.5
    x = np.abs(x)
    return np.where(x < 1.0, ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0,
                    np.where(x < 2.0, (((x - 5.0) * x + 8.0) * x - 4.0) * a, 0.0))


@lru_cache(maxsize=64)  # a camera sends one size: the weights are computed once
def _coeffs(in_size: int, out_size: int, bicubic: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(xmin per output, count per output, int64 weights [out, ksize])."""
    support_1 = 2.0 if bicubic else 1.0
    scale = in_size / out_size
    filterscale = max(scale, 1.0)
    support = support_1 * filterscale
    ksize = int(math.ceil(support)) * 2 + 1
    xmins = np.zeros(out_size, dtype=np.int64)
    counts = np.zeros(out_size, dtype=np.int64)
    kk = np.zeros((out_size, ksize), dtype=np.int64)
    filt = _bicubic if bicubic else _bilinear
    for xx in range(out_size):
        center = (xx + 0.5) * scale
        xmin = max(int(center - support + 0.5), 0)
        xmax = min(int(center + support + 0.5), in_size) - xmin
        w = filt((np.arange(xmax) + xmin - center + 0.5) * (1.0 / filterscale))
        # Pillow sums the weights one by one in double; np.sum pairs them, so
        # sum sequentially to divide by the same value.
        ww = 0.0
        for v in w:
            ww += float(v)
        if ww != 0.0:
            w = w / ww
        scaled = w * (1 << _PRECISION_BITS)
        kk[xx, :xmax] = np.where(w < 0, (-0.5 + scaled), (0.5 + scaled)).astype(np.int64)
        xmins[xx] = xmin
        counts[xx] = xmax
    return xmins, counts, kk


def _pass(img: np.ndarray, xmins: np.ndarray, counts: np.ndarray, kk: np.ndarray, axis: int) -> np.ndarray:
    """One separable pass along `axis` (1: columns, 0: rows) of an HxWxC uint8 image."""
    src = np.moveaxis(img, axis, 0).astype(np.int64)          # [in, other, C]
    out_size, ksize = kk.shape
    acc = np.full((out_size,) + src.shape[1:], 1 << (_PRECISION_BITS - 1), dtype=np.int64)
    for t in range(ksize):
        live = counts > t
        if not np.any(live):
            break
        idx = np.where(live, xmins + t, 0)
        w = np.where(live, kk[:, t], 0)
        acc += src[idx] * w[:, None, None]
    out = np.clip(acc >> _PRECISION_BITS, 0, 255)
    out = np.where(acc >= (1 << _PRECISION_BITS << 8), 255, np.where(acc <= 0, 0, out))
    return np.moveaxis(out.astype(np.uint8), 0, axis)


def resize(img: np.ndarray, width: int, height: int, bicubic: bool = False) -> np.ndarray:
    """Same result as PIL.Image.fromarray(img).resize((width, height), BILINEAR | BICUBIC)."""
    squeeze = img.ndim == 2
    if squeeze:
        img = img[:, :, None]
    h, w = img.shape[:2]
    need_h, need_v = width != w, height != h
    vx = _coeffs(h, height, bicubic) if need_v else None
    out = img
    if need_h:
        if need_v:  # only the rows the vertical kernels read (Pillow's ybox)
            y_first = int(vx[0][0])
            y_last = int(vx[0][-1] + vx[1][-1])
            out = out[y_first:y_last]
            vx = (vx[0] - y_first, vx[1], vx[2])
        out = _pass(out, *_coeffs(w, width, bicubic), axis=1)
    if need_v:
        out = _pass(out, *vx, axis=0)
    out = np.ascontiguousarray(out)
    return out[:, :, 0] if squeeze else out


def resize_short_center_crop(img: np.ndarray, short_side: int, crop_w: int, crop_h: int,
                             bicubic: bool = False) -> np.ndarray:
    """torchvision Resize(short_side) + CenterCrop((crop_h, crop_w)) on an HWC uint8 image."""
    h, w = img.shape[:2]
    if w <= h:
        nw, nh = short_side, int(short_side * h / w)
    else:
        nh, nw = short_side, int(short_side * w / h)
    r = resize(img, nw, nh, bicubic)
    left, top = int(round((nw - crop_w) / 2.0)), int(round((nh - crop_h) / 2.0))
    out = np.zeros((crop_h, crop_w) + r.shape[2:], dtype=np.uint8)
    ys, xs = max(top, 0), max(left, 0)
    ye, xe = min(top + crop_h, nh), min(left + crop_w, nw)
    out[ys - top:ye - top, xs - left:xe - left] = r[ys:ye, xs:xe]
    return out
