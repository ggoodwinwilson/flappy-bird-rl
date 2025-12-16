from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ViewerInput:
    quit: bool = False
    flap: bool = False
    reset: bool = False
    toggle_pause: bool = False


class PygameViewer:
    def __init__(
        self,
        caption: str = "Flappy Bird (Value Overlay)",
        window_scale: int = 1,
        target_fps: int = 60,
    ):
        self.caption = caption
        self.window_scale = max(1, int(window_scale))
        self.target_fps = int(target_fps)
        self._initialized = False
        self._screen = None
        self._clock = None
        self._paused = False

    def _init(self, frame_rgb: np.ndarray) -> None:
        import pygame

        pygame.init()
        pygame.display.set_caption(self.caption)
        pygame.key.set_repeat(0)

        h, w = int(frame_rgb.shape[0]), int(frame_rgb.shape[1])
        self._screen = pygame.display.set_mode((w * self.window_scale, h * self.window_scale))
        self._clock = pygame.time.Clock()
        self._initialized = True

    @property
    def paused(self) -> bool:
        return self._paused

    def close(self) -> None:
        if not self._initialized:
            return
        import pygame

        pygame.quit()
        self._initialized = False
        self._screen = None
        self._clock = None

    def tick(self) -> float:
        if not self._initialized or self._clock is None or self.target_fps <= 0:
            return 0.0
        dt_ms = self._clock.tick(self.target_fps)
        return float(dt_ms) / 1000.0

    def poll(self) -> ViewerInput:
        import pygame

        out = ViewerInput()
        # Avoid crashing if called before the first `show()` (which initializes the window).
        if not pygame.display.get_init():
            return out
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                out.quit = True
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    out.quit = True
                elif event.key in (pygame.K_SPACE, pygame.K_UP):
                    out.flap = True
                elif event.key == pygame.K_r:
                    out.reset = True
                elif event.key == pygame.K_p:
                    out.toggle_pause = True

        if out.toggle_pause:
            self._paused = not self._paused
        return out

    def show(self, frame_rgb: np.ndarray, *, tick: bool = True) -> float:
        if frame_rgb is None:
            return 0.0
        if not self._initialized:
            self._init(frame_rgb)

        import pygame

        # Pygame expects (W, H, 3) for surfarray.make_surface
        surface = pygame.surfarray.make_surface(np.transpose(frame_rgb, (1, 0, 2)))
        if self.window_scale != 1:
            w, h = surface.get_size()
            surface = pygame.transform.scale(surface, (w * self.window_scale, h * self.window_scale))

        self._screen.blit(surface, (0, 0))
        pygame.display.flip()

        return self.tick() if tick else 0.0
