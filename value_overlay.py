from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class ValueOverlayConfig:
    history_len: int = 240
    panel_size: Tuple[int, int] = (340, 110)
    panel_xy: Tuple[int, int] = (12, 12)
    padding: int = 10
    line_width: int = 2
    fps_smoothing: float = 0.9


class _StickyGraph:
    def __init__(self, history_len: int):
        self._values: Deque[float] = deque(maxlen=history_len)
        self._vmin: Optional[float] = None
        self._vmax: Optional[float] = None

    @property
    def values(self) -> list[float]:
        return list(self._values)

    @property
    def vmin(self) -> Optional[float]:
        return self._vmin

    @property
    def vmax(self) -> Optional[float]:
        return self._vmax

    def reset(self) -> None:
        self._values.clear()
        self._vmin = None
        self._vmax = None

    def update(self, value: float) -> None:
        v = float(value)
        self._values.append(v)
        if self._vmin is None or v < self._vmin:
            self._vmin = v
        if self._vmax is None or v > self._vmax:
            self._vmax = v


class ValueRewardOverlay:
    def __init__(self, config: ValueOverlayConfig = ValueOverlayConfig()):
        self.config = config
        # Default bitmap font is quite small; prefer a truetype font when available.
        self._font = self._load_font(size=14)
        self._fps_ema: Optional[float] = None
        self._value = _StickyGraph(history_len=config.history_len)
        self._reward = _StickyGraph(history_len=config.history_len)

    @staticmethod
    def _load_font(size: int) -> ImageFont.ImageFont:
        try:
            return ImageFont.truetype("DejaVuSans.ttf", size=size)
        except Exception:
            return ImageFont.load_default()

    def reset(self) -> None:
        self._fps_ema = None
        self._value.reset()
        self._reward.reset()

    def update_value(self, value: float) -> None:
        self._value.update(value)

    def update_reward(self, reward: float) -> None:
        self._reward.update(reward)

    def update(self, value: float, reward: float, dt_sec: Optional[float] = None) -> None:
        self.update_value(value)
        self.update_reward(reward)
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

    def _draw_panel(
        self,
        draw: ImageDraw.ImageDraw,
        *,
        x0: int,
        y0: int,
        w: int,
        h: int,
        title: str,
        graph: _StickyGraph,
    ) -> None:
        x1, y1 = x0 + w, y0 + h
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=10,
            fill=(10, 10, 10, 170),
            outline=(255, 255, 255, 80),
            width=2,
        )

        pad = self.config.padding
        header_h = 14
        try:
            bbox = self._font.getbbox("Ag")
            header_h = max(12, int(bbox[3] - bbox[1] + 2))
        except Exception:
            pass

        gx0, gy0 = x0 + pad, y0 + pad + header_h
        gx1, gy1 = x1 - pad, y1 - pad
        gh = max(1, gy1 - gy0)
        gw = max(1, gx1 - gx0)

        vals = graph.values
        if graph.vmin is None or graph.vmax is None:
            if len(vals) == 1:
                vmin, vmax = vals[0] - 1.0, vals[0] + 1.0
            else:
                vmin, vmax = -1.0, 1.0
        else:
            vmin, vmax = float(graph.vmin), float(graph.vmax)

        if abs(vmax - vmin) < 1e-6:
            vmax = vmin + 1.0

        for frac in (0.0, 0.5, 1.0):
            y = int(gy0 + frac * gh)
            draw.line((gx0, y, gx1, y), fill=(255, 255, 255, 35), width=1)

        dx = gw / max(1, (self.config.history_len - 1))

        def xy(i: int, v: float) -> Tuple[int, int]:
            px = int(gx0 + i * dx)
            ny = 1.0 - (v - vmin) / (vmax - vmin)
            py = int(gy0 + ny * gh)
            return px, py

        if len(vals) >= 2:
            pts = [xy(i, v) for i, v in enumerate(vals)]
            draw.line(pts, fill=(80, 220, 255, 230), width=self.config.line_width)
            cx, cy = pts[-1]
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255, 230))
        elif len(vals) == 1:
            cx, cy = xy(0, vals[0])
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 255, 255, 230))

        cur = vals[-1] if vals else 0.0
        draw.text((x0 + pad, y0 + pad - 2), f"{title}: {cur:+.3f}", fill=(255, 255, 255, 230), font=self._font)

    def draw(self, frame_rgb: np.ndarray) -> np.ndarray:
        if frame_rgb is None:
            return frame_rgb
        if not isinstance(frame_rgb, np.ndarray) or frame_rgb.ndim != 3:
            return frame_rgb

        img = Image.fromarray(frame_rgb).convert("RGBA")
        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        img_w, img_h = img.size
        margin = 8

        panel_w, panel_h = self.config.panel_size
        panel_w = min(int(panel_w), max(1, img_w - margin * 2))
        panel_h = min(int(panel_h), max(1, (img_h - margin * 3) // 2))
        gap = 8

        x0 = margin
        y_reward = img_h - margin - panel_h
        y_value = y_reward - gap - panel_h
        if y_value < margin:
            y_value = margin
            y_reward = min(img_h - margin - panel_h, y_value + gap + panel_h)

        self._draw_panel(draw, x0=x0, y0=y_value, w=panel_w, h=panel_h, title="Value", graph=self._value)
        self._draw_panel(draw, x0=x0, y0=y_reward, w=panel_w, h=panel_h, title="Reward", graph=self._reward)

        out = Image.alpha_composite(img, overlay).convert("RGB")
        return np.asarray(out)
