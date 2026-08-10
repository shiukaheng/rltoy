"""Pygame renderer for GraphWorld environments."""

import math
from typing import Sequence

import numpy as np
import pygame


class GraphRenderer:
    def __init__(self, env, cell=100, caption="GraphWorld"):
        spec = env.spec
        self._state_names = list(spec["states"])
        self._action_names = list(spec["actions"])
        self._state_index = {name: i for i, name in enumerate(self._state_names)}
        self._action_index = {name: i for i, name in enumerate(self._action_names)}
        self._terminal = env.terminal_states()
        self._cell = cell
        self._values: np.ndarray | None = None
        self._policy: np.ndarray | None = None
        self._vmin = 0.0
        self._vmax = 1.0
        self._node_radius = int(cell * 0.28)

        self._nodes = {}
        for name, state in spec["states"].items():
            pos = state["position"]
            label = state.get("label", name)
            self._nodes[name] = {
                "pos": (pos[0], pos[1]),
                "label": label,
                "terminal": state.get("terminal", False),
            }

        self._edges = []
        for s_name, state in spec["states"].items():
            if state.get("terminal", False):
                continue
            for a_name, outcomes in state["actions"].items():
                for i, outcome in enumerate(outcomes):
                    ns = outcome["next_state"]
                    is_absorbing = (
                        s_name == ns
                        and spec["states"][s_name].get("terminal", False)
                    )
                    if is_absorbing:
                        continue
                    self._edges.append({
                        "src": s_name,
                        "dst": ns,
                        "action": a_name,
                        "prob": outcome["probability"],
                        "reward": outcome.get("reward", 0.0),
                        "outcome_idx": i,
                    })

        margin = cell * 0.8
        w = int(self._cell * 14 + 2 * margin)
        h = int(self._cell * 8 + 2 * margin)
        pygame.init()
        self._screen = pygame.display.set_mode((w, h))
        pygame.display.set_caption(caption)
        self._font_small = pygame.font.SysFont("monospace", int(cell * 0.14))
        self._font_label = pygame.font.SysFont("monospace", int(cell * 0.18), bold=True)
        self._font_value = pygame.font.SysFont("monospace", int(cell * 0.16))
        self._margin = margin

    def set_values(self, values: Sequence[float] | None):
        if values is None:
            self._values = None
            return
        arr = np.asarray(values, dtype=float)
        self._values = arr.ravel()
        finite = self._values[np.isfinite(self._values)]
        if finite.size:
            self._vmin = float(finite.min())
            self._vmax = float(finite.max())

    def set_policy(self, actions: Sequence[int] | None):
        if actions is None:
            self._policy = None
            return
        self._policy = np.asarray(actions, dtype=int).ravel()

    def _to_screen(self, nx: float, ny: float) -> tuple[float, float]:
        pad = 0.12
        usable = 1.0 - 2 * pad
        x = self._margin + (nx - pad) / usable * self._screen.get_width() * 0.85
        y = self._margin + (ny - pad) / usable * self._screen.get_height() * 0.85
        return x, y

    def _node_color(self, name: str) -> tuple[int, int, int]:
        si = self._state_index[name]
        if self._nodes[name]["terminal"]:
            return (60, 200, 60) if "goal" in name.lower() else (220, 60, 60)
        if self._values is not None and si < len(self._values) and np.isfinite(self._values[si]):
            span = self._vmax - self._vmin
            t = 0.0 if span == 0 else (self._values[si] - self._vmin) / span
            t = max(0.0, min(1.0, t))
            hue = 240.0 * (1.0 - t)
            c = pygame.Color(0)
            c.hsva = (hue, 100, 100, 100)
            return tuple(c)[:3]
        return (60, 100, 180)

    def _draw_arrowhead(self, tip_x: float, tip_y: float, angle: float, color, size: int = 10):
        left = angle + math.radians(150)
        right = angle - math.radians(150)
        pts = [
            (int(tip_x), int(tip_y)),
            (
                int(tip_x + size * math.cos(left)),
                int(tip_y + size * math.sin(left)),
            ),
            (
                int(tip_x + size * math.cos(right)),
                int(tip_y + size * math.sin(right)),
            ),
        ]
        pygame.draw.polygon(self._screen, color, pts)

    def _draw_curved_arrow(self, src_pos, dst_pos, offset, color):
        sx, sy = src_pos
        dx, dy = dst_pos
        mid_x = (sx + dx) / 2
        mid_y = (sy + dy) / 2
        angle = math.atan2(dy - sy, dx - sx)
        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)
        cp_x = mid_x + perp_x * offset
        cp_y = mid_y + perp_y * offset

        steps = 30
        pts = []
        for i in range(steps + 1):
            t = i / steps
            x = (1 - t) ** 2 * sx + 2 * (1 - t) * t * cp_x + t**2 * dx
            y = (1 - t) ** 2 * sy + 2 * (1 - t) * t * cp_y + t**2 * dy
            pts.append((int(x), int(y)))
        if len(pts) >= 2:
            pygame.draw.lines(self._screen, color, False, pts, 2)

        tip_angle = math.atan2(pts[-1][1] - pts[-2][1], pts[-1][0] - pts[-2][0])
        self._draw_arrowhead(pts[-1][0], pts[-1][1], tip_angle, color)

    def _count_parallel(self, src: str, dst: str) -> int:
        return sum(1 for e in self._edges if e["src"] == src and e["dst"] == dst)

    def _edge_offset(self, src: str, dst: str, idx: int, total: int) -> float:
        if total <= 1:
            return 0.0
        step = self._node_radius * 0.9
        base = -step * (total - 1) / 2.0
        return base + idx * step

    def _label_pos(self, src_pos, dst_pos, offset):
        sx, sy = src_pos
        dx, dy = dst_pos
        angle = math.atan2(dy - sy, dx - sx)
        mid_x = (sx + dx) / 2
        mid_y = (sy + dy) / 2
        perp_x = -math.sin(angle)
        perp_y = math.cos(angle)
        return mid_x + perp_x * offset * 1.6, mid_y + perp_y * offset * 1.6

    def _shorten_line(self, sx, sy, dx, dy, amount):
        angle = math.atan2(dy - sy, dx - sx)
        sx2 = sx + amount * math.cos(angle)
        sy2 = sy + amount * math.sin(angle)
        dx2 = dx - amount * math.cos(angle)
        dy2 = dy - amount * math.sin(angle)
        return sx2, sy2, dx2, dy2

    def render(self, obs: int):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise KeyboardInterrupt

        self._screen.fill((15, 15, 15))

        edge_groups: dict[tuple[str, str], list] = {}
        for e in self._edges:
            key = (e["src"], e["dst"])
            edge_groups.setdefault(key, []).append(e)

        for e in self._edges:
            key = (e["src"], e["dst"])
            group = edge_groups[key]
            idx = group.index(e)
            total = len(group)
            offset = self._edge_offset(*key, idx, total)

            src_pos = self._to_screen(*self._nodes[e["src"]]["pos"])
            dst_pos = self._to_screen(*self._nodes[e["dst"]]["pos"])
            sx2, sy2, dx2, dy2 = self._shorten_line(*src_pos, *dst_pos, self._node_radius * 1.02)

            if offset == 0.0:
                color = (160, 160, 160)
                pygame.draw.line(self._screen, color, (sx2, sy2), (dx2, dy2), 2)
                angle = math.atan2(dy2 - sy2, dx2 - sx2)
                self._draw_arrowhead(dx2, dy2, angle, color)
            else:
                color = (160, 160, 160)
                self._draw_curved_arrow((sx2, sy2), (dx2, dy2), offset, color)

            prob_str = f"p={e['prob']:.2f}" if e["prob"] != 1.0 else ""
            reward_str = f"r={e['reward']:+.1f}" if e["reward"] != 0.0 else ""
            parts = [e["action"]]
            if prob_str:
                parts.append(prob_str)
            if reward_str:
                parts.append(reward_str)
            label_text = " | ".join(parts)
            lx, ly = self._label_pos(src_pos, dst_pos, offset)
            lbl = self._font_small.render(label_text, True, (200, 200, 200))
            self._screen.blit(lbl, lbl.get_rect(center=(lx, ly)))

        for name, node in self._nodes.items():
            cx, cy = self._to_screen(*node["pos"])
            color = self._node_color(name)
            pygame.draw.circle(self._screen, color, (int(cx), int(cy)), self._node_radius)
            pygame.draw.circle(self._screen, (50, 50, 50), (int(cx), int(cy)), self._node_radius, 2)

            si = self._state_index[name]
            lbl = self._font_label.render(node["label"], True, (255, 255, 255))
            self._screen.blit(lbl, lbl.get_rect(center=(cx, cy - self._node_radius * 0.55)))

            if (
                not node["terminal"]
                and self._values is not None
                and si < len(self._values)
                and np.isfinite(self._values[si])
            ):
                v_text = f"V={self._values[si]:.2f}"
                v_lbl = self._font_value.render(v_text, True, (255, 255, 255))
                self._screen.blit(v_lbl, v_lbl.get_rect(center=(cx, cy + self._node_radius * 0.55)))

            if (
                not node["terminal"]
                and self._policy is not None
                and si < len(self._policy)
            ):
                a_idx = self._policy[si]
                if 0 <= a_idx < len(self._action_names):
                    pi_text = f"pi={self._action_names[a_idx]}"
                    pi_lbl = self._font_value.render(pi_text, True, (255, 210, 60))
                    y_off = (
                        self._node_radius * 1.05
                        if (self._values is not None
                            and si < len(self._values)
                            and np.isfinite(self._values[si]))
                        else self._node_radius * 0.55
                    )
                    self._screen.blit(
                        pi_lbl, pi_lbl.get_rect(center=(cx, cy + y_off))
                    )

        c_name = self._state_names[obs]
        cx, cy = self._to_screen(*self._nodes[c_name]["pos"])
        pygame.draw.circle(
            self._screen,
            (255, 210, 60),
            (int(cx), int(cy)),
            self._node_radius + 4,
            width=3,
        )

        pygame.display.flip()

    def close(self):
        pygame.quit()