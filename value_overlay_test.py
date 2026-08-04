import pygame
import gymnasium as gym
from bettermdptools.algorithms.planner import Planner

env = gym.make("FrozenLake-v1", is_slippery=False).unwrapped
nrow, ncol = env.nrow, env.ncol

V, V_track, pi = Planner(env.P).value_iteration()
V = V.reshape(nrow, ncol)

CELL = 110
W, H = ncol * CELL, nrow * CELL
pygame.init()
screen = pygame.display.set_mode((W, H))
pygame.display.set_caption("FrozenLake Value Map - use arrows")

FONT = pygame.font.SysFont("monospace", int(CELL * 0.3))


def cell_color(value, is_valid):
    if not is_valid:
        return (25, 25, 120)
    v = max(0.0, min(vmax, value))
    t = 0 if vmax == 0 else v / vmax
    return (int(30 + t * 205), int(30 + (1 - t) * 205), 40)


vmin, vmax = V.min(), V.max()

def draw():
    screen.fill((10, 10, 10))
    for r in range(nrow):
        for c in range(ncol):
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            is_valid = env.desc[r][c] != b"H"
            ch = env.desc[r][c].decode()
            if ch in "HG":
                pygame.draw.rect(screen, (30, 130, 30), rect)
            else:
                pygame.draw.rect(screen, cell_color(V[r, c], is_valid), rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 2)
            if is_valid:
                label = FONT.render(f"{V[r, c]:.2f}", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=rect.center))
            else:
                label = FONT.render("HOLE", True, (255, 255, 255))
                screen.blit(label, label.get_rect(center=rect.center))
    pygame.display.flip()


obs, info = env.reset()
player_r, player_c = divmod(obs, ncol)
draw()

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            elif event.key == pygame.K_DOWN:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            else:
                continue
            obs, reward, terminated, truncated, info = env.step(action)
            player_r, player_c = divmod(obs, ncol)
            draw()
            if terminated or truncated:
                obs, info = env.reset()
                player_r, player_c = divmod(obs, ncol)
                draw()

pygame.quit()
env.close()