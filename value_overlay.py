from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ValueOverlayConfig:
    history_len: int = 240
    panel_size: Tuple[int, int] = (360, 160)
    panel_xy: Tuple[int, int] = (12, 12)
    padding: int = 10
    line_width: int = 2
    fps_smoothing: float = 0.9


class ValueGraphOverlay:
    def __init__(self, config: ValueOverlayConfig = ValueOverlayConfig()):
        self.config = config
        self._values: Deque[float] = deque(maxlen=config.history_len)
        self._font = ImageFont.load_default()
        self._fps_ema: Optional[float] = None

    def reset(self) -> None:
        self._values.clear()
        self._fps_ema = None

    def update(self, value: float, dt_sec: Optional[float] = None) -> None:
        self._values.append(float(value))
        self.update_fps(dt_sec)

    def update_fps(self, dt_sec: Optional[float]) -> None:
        if dt_sec is None or dt_sec <= 0:
            return
        fps = 1.0 / dt_sec
        if self._fps_ema is None:
            self._fps_ema = fps
        else:
            a = self.config.fps_smoothing
            self._fps_ema = a * self._fps_ema + (1 - a) * fps

    def draw(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb is None:
            return frame_rgb
        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3:
            return frame_rgb

        img = Image.fromarray(frame_rgb).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        x0, y0 = self.config.panel_xy
        w, h = self.config.panel_size
        x1, y1 = x0 + w, y0 + h

        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=10,
            fill=(10, 10, 10, 170),
            outline=(255, 255, 255, 80),
            width=2,
        )

        pad = self.config.padding
        gx0, gy0 = x0 + pad, y0 + pad + 14
        gx1, gy1 = x1 - pad, y1 - pad
        gh = max(1, gy1 - gy0)
        gw = max(1, gx1 - gx0)

        vals = list(self._values)
        if len(vals) >= 2:
            vmin, vmax = float(min(vals)), float(max(vals))
        elif len(vals) == 1:
            vmin, vmax = vals[0] - 1.0, vals[0] + 1.0
        else:
            vmin, vmax = -1.0, 1.0

        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0

        for frac in (0.0, 0.5, 1.0):
            y = int(gy0 + frac * gh)
            draw.line((gx0, y, gx1, y), fill=(255, 255, 255, 35), width=1)

        def xy(i: int, v: float) -> Tuple[int, int]:
            t = i / max(1, (len(vals) - 1))
            px = int(gx0 + t * gw)
            ny = 1.0 - (v - vmin) / (vmax - vmin)
            py = int(gy0 + ny * gh)
            return px, py

        if len(vals) >= 2:
            pts = [xy(i, v) for i, v in enumerate(vals)]
            draw.line(pts, fill=(80, 220, 255, 230), width=self.config.line_width)
            cx, cy = pts[-1]
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255, 230))

        cur = vals[-1] if vals else 0.0
        title = f"V(s): {cur:+.3f}   min/max: {vmin:+.2f}/{vmax:+.2f}"
        if self._fps_ema is not None:
            title += f"   fps: {self._fps_ema:.1f}"
        draw.text((x0 + pad, y0 + pad - 2), title, fill=(255, 255, 255, 230), font=self._font)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        return np.asarray(out)
