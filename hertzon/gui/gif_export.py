"""GIF export of a field animation: the measured playback rate and the frame timing of the GIF file.

A GIF stores each frame's display time in hundredths of a second, so a rate like 7.9 frames/s
(12.66 cs per frame) cannot be stored exactly per frame. The delays alternate between the two nearest
values so that a full loop of the GIF lasts as long as a loop in the viewer."""
from __future__ import annotations

from collections import deque

import numpy as np

MIN_DELAY_CS = 2                 # browsers and viewers show shorter delays much slower (often 10 cs)
MAX_GIF_FPS = 100.0 / MIN_DELAY_CS


class FpsMeter:
    """Frames actually shown per second during playback, from the times the frames were drawn.

    tick() after each drawn frame, pause() when playback stops (the pause is not counted), reset() when
    the rate changes (other speed setting or other frames)."""

    def __init__(self, window: int = 120):
        self.dts: deque[float] = deque(maxlen=window)
        self.last: float | None = None

    def tick(self, t: float) -> None:
        if self.last is not None and t > self.last:
            self.dts.append(t - self.last)
        self.last = t

    def pause(self) -> None:
        self.last = None

    def reset(self) -> None:
        self.dts.clear()
        self.last = None

    def set_window(self, n: int) -> None:
        """Average over the last n frame intervals (one loop of the animation, at most 300)."""
        n = int(min(300, max(2, n)))
        if n != self.dts.maxlen:
            self.dts = deque(self.dts, maxlen=n)

    def count(self) -> int:
        return len(self.dts)

    def fps(self) -> float | None:
        s = float(sum(self.dts))
        return len(self.dts) / s if self.dts and s > 0 else None


def gif_delays(fps: float, n: int) -> list[int]:
    """Display time of each of n frames in hundredths of a second for a playback rate of fps, at least
    MIN_DELAY_CS each. The rounding error is carried to the next frame, so the loop has the right length."""
    per = max(float(MIN_DELAY_CS), 100.0 / fps)
    out, acc = [], 0.0
    for _ in range(n):
        acc += per
        d = int(round(acc))
        out.append(d)
        acc -= d
    return out


def _packed(im: np.ndarray) -> np.ndarray:
    im = np.asarray(im)[..., :3].astype(np.uint32)
    return (im[..., 0] << 16) | (im[..., 1] << 8) | im[..., 2]


def _unpacked(p: np.ndarray) -> np.ndarray:
    return np.stack([(p >> 16) & 255, (p >> 8) & 255, p & 255], -1).astype(np.uint8)


def gif_palette(images: list[np.ndarray], n: int = 256) -> np.ndarray:
    """One palette (n x 3) for all frames, so that a colour is the same in every frame (with one palette
    per frame, the colour bar and the field jump between frames). Every distinct colour counts with the
    logarithm of its pixel count: the big uniform background does not crowd out the colour scale and the
    small parts of the field."""
    u, c = np.zeros(0, np.uint32), np.zeros(0, np.int64)
    for im in images:
        fu, fc = np.unique(_packed(im), return_counts=True)
        u, inv = np.unique(np.concatenate([u, fu]), return_inverse=True)
        c = np.bincount(inv, weights=np.concatenate([c, fc]), minlength=len(u)).astype(np.int64)
    cols = _unpacked(u)
    if len(cols) <= n:
        return cols
    from PIL import Image
    w = 1 + np.floor(np.log2(c)).astype(int)
    sample = np.repeat(cols, w, axis=0).reshape(-1, 1, 3)
    pal = Image.fromarray(sample).quantize(n, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE,
                                                  kmeans=3)
    return np.array(pal.getpalette()[:3 * n], np.uint8).reshape(-1, 3)


def palette_indices(im: np.ndarray, pal: np.ndarray) -> np.ndarray:
    """The nearest palette colour of every pixel (exact, no dithering: still parts stay still)."""
    u, inv = np.unique(_packed(im), return_inverse=True)
    cols, P = _unpacked(u).astype(np.int32), pal.astype(np.int32)
    idx = np.empty(len(u), np.uint8)
    for k in range(0, len(u), 4096):
        idx[k:k + 4096] = ((cols[k:k + 4096, None, :] - P[None]) ** 2).sum(-1).argmin(1)
    return idx[inv.reshape(-1)].reshape(im.shape[:2])


def write_gif(path: str, images: list[np.ndarray], fps: float) -> list[int]:
    """Save RGB(A) frames (all of one size) as a looping GIF at fps frames per second; returns the
    delays in hundredths of a second."""
    from PIL import Image
    if not images:
        raise ValueError("no frames")
    if len({im.shape[:2] for im in images}) > 1:
        raise ValueError("the frames differ in size")
    delays = gif_delays(fps, len(images))
    pal = gif_palette(images)
    flat = pal.reshape(-1).tolist()
    pics = []
    for im in images:
        pic = Image.frombytes("P", im.shape[1::-1], palette_indices(im, pal).tobytes())
        pic.putpalette(flat)
        pics.append(pic)
    pics[0].save(path, format="GIF", save_all=True, append_images=pics[1:], duration=[10 * d for d in delays],
                 loop=0, disposal=1, optimize=False)
    return delays
