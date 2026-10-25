"""GIF export of field animations: frame timing from the measured playback rate, and the GIF has the
pixel size of the view and runs as fast as the playback on screen."""
from __future__ import annotations

import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from hertzon.gui.gif_export import MIN_DELAY_CS, FpsMeter, gif_delays, gif_palette, write_gif


def test_delays_keep_the_loop_length():
    d = gif_delays(7.9, 79)                       # 12.66 cs per frame cannot be stored per frame
    assert set(d) <= {12, 13}
    assert sum(d) == 1000                         # 79 frames at 7.9 frames/s last 10 s
    assert sum(gif_delays(3.0, 18)) == 600
    assert gif_delays(60.0, 5) == [MIN_DELAY_CS] * 5   # faster than a GIF can show


def test_meter_counts_only_playback():
    m = FpsMeter()
    for t in (0.0, 0.1, 0.2, 0.3):
        m.tick(t)
    m.pause()
    m.tick(5.0)                                   # after a pause: the pause is not a frame interval
    m.tick(5.1)
    assert m.count() == 4
    assert m.fps() == pytest.approx(10.0)
    m.set_window(2)                               # average over the last loop only
    m.tick(5.15)
    assert m.fps() == pytest.approx(2 / 0.15)
    m.reset()
    assert m.fps() is None


def test_gif_frames_size_and_timing(tmp_path):
    from PIL import Image
    imgs = [np.full((37, 53, 3), (20 * i, 255 - 20 * i, 90), np.uint8) for i in range(10)]
    path = tmp_path / "a.gif"
    delays = write_gif(str(path), imgs, 7.9)
    with Image.open(path) as g:
        assert g.size == (53, 37)
        assert g.n_frames == 10
        got = []
        for k in range(g.n_frames):
            g.seek(k)
            got.append(g.info["duration"])
            assert g.convert("RGB").getpixel((5, 5)) == tuple(int(v) for v in imgs[k][5, 5])
        assert got == [10 * d for d in delays]
        assert g.info.get("loop") == 0
    with pytest.raises(ValueError):
        write_gif(str(tmp_path / "b.gif"), [imgs[0], imgs[0][:10]], 5.0)


def test_colours_stay_the_same_in_every_frame(tmp_path):
    """More than 256 colours: one palette for all frames, so a part that does not change (the colour bar)
    looks the same in every frame, and a small coloured part is not swallowed by a big gradient."""
    from PIL import Image
    h, w = 120, 200
    ramp = np.linspace(0, 1, 100)[:, None]
    bar = (np.concatenate([ramp, 1 - ramp, 0.5 * ramp], 1) * 255).astype(np.uint8)      # 100 colours
    imgs = []
    for i in range(8):
        im = np.zeros((h, w, 3), np.uint8)
        x = np.linspace(0, 1, w - 20)
        im[:100, :w - 20] = (np.stack([x, (x + i / 8) % 1, 1 - x], 1) * 255).astype(np.uint8)[None]
        im[100:, :w - 20] = (180, 190, 200)
        im[110:114, 5:9] = (255, 0, 255)                                                # 16 rare pixels
        im[:100, w - 20:] = bar[:, None]
        imgs.append(im)
    assert len(gif_palette(imgs)) == 256
    path = tmp_path / "c.gif"
    write_gif(str(path), imgs, 10.0)
    with Image.open(path) as g:
        out = []
        for k in range(g.n_frames):
            g.seek(k)
            out.append(np.asarray(g.convert("RGB")).astype(int))
    assert len(out) == 8
    for o in out[1:]:
        assert np.array_equal(o[:, w - 20:], out[0][:, w - 20:])                    # colour bar unchanged
        assert np.array_equal(o[100:, :w - 20], out[0][100:, :w - 20])
    for o, im in zip(out, imgs):
        assert np.abs(o - im).sum(-1).mean() < 12
        assert np.abs(o[110:114, 5:9] - (255, 0, 255)).sum(-1).max() < 40
    few = [np.full((4, 4, 3), (10 * k, 0, 0), np.uint8) for k in range(5)]
    assert len(gif_palette(few)) == 5                                                # exact when it fits


def test_export_uses_the_measured_rate_and_the_view_size(tmp_path):
    """Drawing a frame takes 40 ms, so at a setting of 60 frames/s only about 20-25 are shown; the GIF
    has to run at the measured rate, not at the setting, and keep the size of the view."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PIL import Image
    from PySide6.QtWidgets import QApplication, QWidget
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from hertzon.gui.animation import AnimationBar

    shown = []

    def render():
        time.sleep(0.04)

    def screenshot(transparent_background=False, return_img=True):
        img = np.zeros((48, 64, 3), np.uint8)
        img[..., 0] = 25 * shown[-1]
        return img

    win = QWidget()
    win.view = SimpleNamespace(set_overlay=lambda name, meshes, render=True: shown.append(meshes),
                               plotter=SimpleNamespace(render=render, screenshot=screenshot))
    bar = AnimationBar(win)
    bar.load(["Mag_E1"], [(f"Phase = {10 * i} deg", {"Mag_E1": i}) for i in range(10)])
    bar.speed.setValue(60)
    assert bar.meter.fps() is None
    path = bar.export_gif(str(tmp_path / "anim.gif"))      # measures first, the player was not running
    fps = bar.meter.fps()
    assert path and fps is not None
    assert fps < 30                                         # slower than the setting: what was really shown
    assert not bar.is_playing()
    with Image.open(path) as g:
        assert g.size == (64, 48)
        assert g.n_frames == 10
        total = 0
        for k in range(g.n_frames):
            g.seek(k)
            total += g.info["duration"]
            assert g.convert("RGB").getpixel((0, 0))[0] == 25 * k
    assert abs(total - 1000 * 10 / fps) <= 10              # one loop lasts as long as in the viewer
    assert "Actual" in bar.actual.text()
