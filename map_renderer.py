"""Framework-agnostic pygame renderer for gridworld environments.

This module owns all presentation logic so that the agent loop stays free of
rendering concerns. It knows nothing about your agent: it only needs the
environment's grid geometry (nrow / ncol / desc) and, optionally, per-cell
values injected via ``set_values``.
"""

import pygame
import numpy as np


class GridMapRenderer:
    def __init__(self, env, cell=110, caption="Grid Map", marker_color=(255, 210, 60)):
        self.nrow = env.nrow
        self.ncol = env.ncol
        self.desc = getattr(env, "desc", None)
        self.cell = cell
        self.marker_color = marker_color
        self.values = None
        self._vmin = 0.0
        self._vmax = 1.0

        pygame.init()
        self.screen = pygame.display.set_mode((self.ncol * cell, self.nrow * cell))
        pygame.display.set_caption(caption)
        self._font = pygame.font.SysFont("monospace", int(cell * 0.3))

    def set_values(self, values):
        """Inject per-cell values (any iterable of length nrow*ncol)."""
        if values is None:
            self.values = None
            return
        arr = np.asarray(values, dtype=float)
        self.values = arr.reshape(self.nrow, self.ncol)
        finite = self.values[np.isfinite(self.values)]
        if finite.size:
            self._vmin = float(finite.min())
            self._vmax = float(finite.max())

    def _tile_color(self, value, is_valid):
        if not is_valid:
            return (25, 25, 120)
        if self.values is None or not np.isfinite(value):
            return (70, 120, 200)
        span = self._vmax - self._vmin
        t = 0.0 if span == 0 else (value - self._vmin) / span
        return (
            int(30 + t * 205),
            int(30 + (1 - t) * 205),
            40,
        )

    def render(self, obs):
        """Redraw the grid with the agent located at cell index ``obs``."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self.screen.fill((10, 10, 10))
        pr, pc = divmod(int(obs), self.ncol)

        for r in range(self.nrow):
            for c in range(self.ncol):
                rect = pygame.Rect(c * self.cell, r * self.cell, self.cell, self.cell)
                ch = None
                if self.desc is not None:
                    ch = self.desc[r][c].decode() if not isinstance(self.desc[r][c], str) else self.desc[r][c]
                is_valid = ch is None or ch != "H"
                value = self.values[r, c] if self.values is not None else None

                if ch in ("H", "G"):
                    color = (30, 130, 30) if ch == "G" else (25, 25, 120)
                else:
                    color = self._tile_color(value, is_valid)
                pygame.draw.rect(self.screen, color, rect)
                pygame.draw.rect(self.screen, (50, 50, 50), rect, 2)

                if ch == "G":
                    text = "GOAL"
                elif ch == "H":
                    text = "HOLE"
                elif self.values is not None and np.isfinite(value):
                    text = f"{value:.2f}"
                else:
                    text = ""
                if text:
                    label = self._font.render(text, True, (255, 255, 255))
                    self.screen.blit(label, label.get_rect(center=rect.center))

        if self.marker_color is not None:
            center = (pc * self.cell + self.cell // 2, pr * self.cell + self.cell // 2)
            pygame.draw.circle(self.screen, self.marker_color, center, self.cell // 3, width=5)

        pygame.display.flip()

    def close(self):
        pygame.quit()