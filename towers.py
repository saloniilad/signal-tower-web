"""
Cellular Network Simulator - Day 2: Tower Placement & Signal Model
------------------------------------------------------------------
Needs day1_city_grid.py in the same folder (we reuse its map code).

Signal model (log-distance, urban macro):
    PathLoss(dB) = 128.1 + 37.6 * log10(d_km)
    Rx(dBm)      = Tx_power(dBm) - PathLoss(dB)
A cell is "covered" when Rx >= RX_THRESHOLD.

Controls:
    Left click   place tower (uses next-tower channel/power)
    Right click  remove tower under cursor
    1-7          channel  (hovering a tower changes that tower, else the next one)
    UP / DOWN    power +/- 1 dBm (same hover rule)
    H            toggle signal heatmap      O  toggle coverage circles
    G            toggle grid lines          R  new random city
    C            clear all towers           S  save screenshot
    ESC          quit
"""

import numpy as np
import pygame

from city_grid import (
    GRID_W, GRID_H, CELL, SIDEBAR, N_ZONES, SEED, MANUAL_ZONES,
    BG, PANEL, TEXT, MUTED, generate_population, render_map, draw_grid_lines,
)

# ----------------------------- CONFIG ---------------------------------
CELL_KM = 0.1                 # each grid cell = 100 m
N_CHANNELS = 7                # channel set used for frequency reuse later
CHANNEL_COLORS = [
    (239, 68, 68), (249, 115, 22), (234, 179, 8), (34, 197, 94),
    (6, 182, 212), (99, 102, 241), (217, 70, 239),
]

PL_REF, PL_EXP = 128.1, 37.6  # path-loss intercept (dB at 1 km) and slope
MIN_DIST_KM = 0.05            # avoid log(0) right at the tower
RX_THRESHOLD = -95.0          # dBm: minimum usable signal
RX_STRONG = -55.0             # dBm: heatmap saturates here
POWER_MIN, POWER_MAX, POWER_DEFAULT = 10, 43, 30

YS, XS = np.mgrid[0:GRID_H, 0:GRID_W]


# ----------------------------- SIGNAL MODEL ---------------------------
def path_loss_db(d_km):
    return PL_REF + PL_EXP * np.log10(np.maximum(d_km, MIN_DIST_KM))


class Tower:
    def __init__(self, col, row, power_dbm, channel):
        self.col, self.row = col, row
        self.power_dbm = power_dbm
        self.channel = channel          # 0..N_CHANNELS-1
        self.signal = None
        self.update()

    def update(self):
        """Recompute Rx power (dBm) at every grid cell. Reused for interference on Day 3."""
        d_km = np.hypot(XS - self.col, YS - self.row) * CELL_KM
        self.signal = self.power_dbm - path_loss_db(d_km)

    @property
    def radius_cells(self):
        """Distance (in cells) where Rx drops to the threshold."""
        d_km = 10 ** ((self.power_dbm - RX_THRESHOLD - PL_REF) / PL_EXP)
        return d_km / CELL_KM

    @property
    def center_px(self):
        return int((self.col + 0.5) * CELL), int((self.row + 0.5) * CELL)


def radius_for_power(p):
    return 10 ** ((p - RX_THRESHOLD - PL_REF) / PL_EXP) / CELL_KM


def tower_at(towers, cell):
    return next((t for t in towers if (t.col, t.row) == cell), None)


def best_signal(towers):
    return np.max([t.signal for t in towers], axis=0) if towers else None


# ----------------------------- RENDERING ------------------------------
def rgba_to_surface(rgba):
    """(GRID_H, GRID_W, 4) uint8 array -> per-pixel-alpha Surface scaled to the map size.

    Uses pygame.image.frombuffer instead of pygame.surfarray, which is more
    reliable in the browser (Pygbag) build.
    """
    buf = np.ascontiguousarray(rgba, dtype=np.uint8).tobytes()   # keep alive until scaled
    small = pygame.image.frombuffer(buf, (GRID_W, GRID_H), "RGBA")
    return pygame.transform.scale(small, (GRID_W * CELL, GRID_H * CELL))


def build_overlay(best):
    """Semi-transparent heat layer: brighter = stronger signal, clear = uncovered."""
    s = np.clip((best - RX_THRESHOLD) / (RX_STRONG - RX_THRESHOLD), 0, 1)
    alpha = (s * 170).astype(np.uint8)
    alpha[best < RX_THRESHOLD] = 0
    rgba = np.empty((GRID_H, GRID_W, 4), dtype=np.uint8)
    rgba[..., 0] = (255 * s).astype(np.uint8)
    rgba[..., 1] = (200 + 55 * s).astype(np.uint8)
    rgba[..., 2] = 255
    rgba[..., 3] = alpha
    return rgba_to_surface(rgba)


def draw_towers(screen, towers, font, show_circles, hover_tower):
    screen.set_clip(pygame.Rect(0, 0, GRID_W * CELL, GRID_H * CELL))
    for t in towers:
        color = CHANNEL_COLORS[t.channel]
        if show_circles:
            pygame.draw.circle(screen, color, t.center_px, int(t.radius_cells * CELL), 2)
    for t in towers:
        color = CHANNEL_COLORS[t.channel]
        pygame.draw.circle(screen, (255, 255, 255), t.center_px, 9)
        pygame.draw.circle(screen, color, t.center_px, 7)
        if t is hover_tower:
            pygame.draw.circle(screen, (255, 255, 255), t.center_px, 13, 2)
        label = font.render(str(t.channel + 1), True, (0, 0, 0))
        screen.blit(label, label.get_rect(center=t.center_px))
    screen.set_clip(None)


def draw_sidebar(screen, fonts, towers, best, hover, hover_tower, next_ch, next_pw, pop):
    big, small = fonts
    x0, H = GRID_W * CELL, GRID_H * CELL
    pygame.draw.rect(screen, PANEL, (x0, 0, SIDEBAR, H))
    px, y = x0 + 16, 14

    def line(text, color=TEXT, dy=20):
        nonlocal y
        screen.blit(small.render(text, True, color), (px, y))
        y += dy

    screen.blit(big.render("Cell Network Sim", True, TEXT), (px, y)); y += 26
    line("Day 2 - Towers & signal", MUTED, 28)

    # Channel picker
    target = hover_tower
    ch = target.channel if target else next_ch
    pw = target.power_dbm if target else next_pw
    line("Editing hovered tower" if target else "Next tower to place", TEXT, 22)
    for i, c in enumerate(CHANNEL_COLORS):
        r = pygame.Rect(px + i * 31, y, 27, 22)
        pygame.draw.rect(screen, c, r, border_radius=4)
        if i == ch:
            pygame.draw.rect(screen, (255, 255, 255), r, 3, border_radius=4)
        n = small.render(str(i + 1), True, (0, 0, 0))
        screen.blit(n, n.get_rect(center=r.center))
    y += 30
    line(f"Power: {pw} dBm")
    line(f"Radius: {radius_for_power(pw):.1f} cells ({radius_for_power(pw) * CELL_KM:.2f} km)", TEXT, 28)

    # Stats
    covered = float((best >= RX_THRESHOLD).mean() * 100) if best is not None else 0.0
    line(f"Towers: {len(towers)}")
    line(f"Area covered: {covered:.1f}%", TEXT, 28)

    # Hover
    if hover:
        hx, hy = hover
        line(f"Cell ({hx}, {hy})  density {pop[hy, hx]:.2f}")
        if best is not None:
            line(f"Best signal: {best[hy, hx]:.1f} dBm")
        else:
            line("Best signal: no towers", MUTED)

    y = H - 150
    for txt in ["L-click: place tower", "R-click: remove tower", "1-7: channel",
                "UP/DOWN: power", "H: heatmap   O: circles", "G: grid   R: new map",
                "C: clear   S: screenshot"]:
        line(txt, MUTED, 20)


# ------------------------------- MAIN ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL + SIDEBAR, GRID_H * CELL))
    pygame.display.set_caption("Cellular Network Simulator - Day 2")
    fonts = (pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 20, bold=True),
             pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 15))
    tower_font = pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 13, bold=True)
    clock = pygame.time.Clock()

    rng = np.random.default_rng(SEED)
    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
    map_surface = render_map(pop)

    towers = []
    next_ch, next_pw = 0, POWER_DEFAULT
    show_grid = show_heat = show_circles = True
    overlay, best, dirty = None, None, True

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        hover = (mx // CELL, my // CELL) if mx < GRID_W * CELL and my < GRID_H * CELL else None
        hover_tower = tower_at(towers, hover) if hover else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and hover:
                if event.button == 1 and not hover_tower:
                    towers.append(Tower(hover[0], hover[1], next_pw, next_ch))
                    dirty = True
                elif event.button == 3 and hover_tower:
                    towers.remove(hover_tower)
                    dirty = True

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif pygame.K_1 <= k < pygame.K_1 + N_CHANNELS:
                    if hover_tower:
                        hover_tower.channel = k - pygame.K_1
                    else:
                        next_ch = k - pygame.K_1
                elif k in (pygame.K_UP, pygame.K_DOWN):
                    d = 1 if k == pygame.K_UP else -1
                    if hover_tower:
                        hover_tower.power_dbm = int(np.clip(hover_tower.power_dbm + d, POWER_MIN, POWER_MAX))
                        hover_tower.update()
                        dirty = True
                    else:
                        next_pw = int(np.clip(next_pw + d, POWER_MIN, POWER_MAX))
                elif k == pygame.K_h:
                    show_heat = not show_heat
                elif k == pygame.K_o:
                    show_circles = not show_circles
                elif k == pygame.K_g:
                    show_grid = not show_grid
                elif k == pygame.K_c:
                    towers.clear(); dirty = True
                elif k == pygame.K_r:
                    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
                    map_surface = render_map(pop)
                elif k == pygame.K_s:
                    pygame.image.save(screen, "screenshot_day2.png")
                    print("Saved screenshot_day2.png")

        if dirty:
            best = best_signal(towers)
            overlay = build_overlay(best) if best is not None else None
            dirty = False

        screen.fill(BG)
        screen.blit(map_surface, (0, 0))
        if show_heat and overlay:
            screen.blit(overlay, (0, 0))
        if show_grid:
            draw_grid_lines(screen)
        if hover and not hover_tower:
            pygame.draw.rect(screen, (255, 255, 255), (hover[0] * CELL, hover[1] * CELL, CELL, CELL), 2)
        draw_towers(screen, towers, tower_font, show_circles, hover_tower)
        draw_sidebar(screen, fonts, towers, best, hover, hover_tower, next_ch, next_pw, pop)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()