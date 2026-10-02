"""
Cellular Network Simulator - Interference Calculation
------------------------------------------------------------
Needs day1_city_grid.py and day2_towers.py in the same folder.

For every cell:
    serving tower  = the tower with the strongest signal there
    S              = signal from the serving tower
    I              = co-channel signals from other towers  (weight 1.0)
                   + adjacent-channel signals              (weight ADJ_WEIGHT)
    SIR (dB)       = S - 10*log10(I + noise)
A covered cell is flagged RED when SIR < SIR_MIN.

Controls (new ones marked *):
    Left/Right click  place / remove tower
    1-7               channel      UP/DOWN  power   (hovered tower, else next tower)
  * [ and ]           lower / raise the SIR threshold
  * I                 toggle interference overlay
  * A                 set ALL towers to channel 1 (worst-case demo)
    H heat, O circles, G grid, R new map, C clear, S screenshot, ESC quit
"""

import numpy as np
import pygame

from city_grid import (
    GRID_W, GRID_H, CELL, SIDEBAR, N_ZONES, SEED, MANUAL_ZONES,
    BG, PANEL, TEXT, MUTED, generate_population, render_map, draw_grid_lines,
)
from towers import (
    Tower, N_CHANNELS, CHANNEL_COLORS, RX_THRESHOLD, POWER_MIN, POWER_MAX,
    POWER_DEFAULT, radius_for_power, tower_at, build_overlay, draw_towers, rgba_to_surface,
)

# ----------------------------- CONFIG ---------------------------------
ADJ_WEIGHT = 0.1        # adjacent-channel leakage vs co-channel (0.1 = -10 dB)
NOISE_DBM = -110.0      # thermal noise floor
SIR_MIN_DEFAULT = 12.0  # dB: below this a covered cell is "interfered"


# ----------------------------- INTERFERENCE ---------------------------
def compute_interference(towers):
    """Return (best_dbm, sir_db, serving_index), each an (H, W) array."""
    sig = np.stack([t.signal for t in towers])          # (n, H, W) in dBm
    n = len(towers)
    serving = np.argmax(sig, axis=0)                    # strongest tower per cell
    best = np.max(sig, axis=0)

    ch = np.array([t.channel for t in towers])
    diff = np.abs(ch[:, None, None] - ch[serving][None])  # channel gap to serving tower
    weight = np.where(diff == 0, 1.0, np.where(diff == 1, ADJ_WEIGHT, 0.0))
    weight *= (np.arange(n)[:, None, None] != serving[None])  # exclude the serving tower

    interference_mw = (10 ** (sig / 10) * weight).sum(axis=0)
    sir = best - 10 * np.log10(interference_mw + 10 ** (NOISE_DBM / 10))
    return best, sir, serving


def build_interference_overlay(sir, bad, sir_min):
    """Red layer on flagged cells; deeper red = worse SIR."""
    severity = np.clip((sir_min - sir) / 15, 0, 1)
    alpha = np.where(bad, 90 + 130 * severity, 0).astype(np.uint8)
    rgba = np.empty((GRID_H, GRID_W, 4), dtype=np.uint8)
    rgba[..., 0], rgba[..., 1], rgba[..., 2] = 255, 30, 30
    rgba[..., 3] = alpha
    return rgba_to_surface(rgba)


# ----------------------------- SIDEBAR --------------------------------
def draw_sidebar(screen, fonts, towers, hover, hover_tower, next_ch, next_pw, sir_min, info):
    big, small = fonts
    x0, H = GRID_W * CELL, GRID_H * CELL
    pygame.draw.rect(screen, PANEL, (x0, 0, SIDEBAR, H))
    px, y = x0 + 16, 14

    def line(text, color=TEXT, dy=20):
        nonlocal y
        screen.blit(small.render(text, True, color), (px, y))
        y += dy

    screen.blit(big.render("Cell Network Sim", True, TEXT), (px, y)); y += 26
    line("Day 3 - Interference", MUTED, 28)

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
    line(f"Radius: {radius_for_power(pw):.1f} cells", TEXT, 28)

    line(f"Towers: {len(towers)}")
    line(f"Area covered: {info['covered_pct']:.1f}%")
    line(f"Interfered: {info['bad_pct']:.1f}% of cov.", (255, 120, 120) if info['bad_pct'] > 0 else TEXT)
    pygame.draw.rect(screen, (255, 30, 30), (px, y + 3, 14, 14))
    screen.blit(small.render(f"SIR < {sir_min:.0f} dB", True, TEXT), (px + 22, y)); y += 28

    if hover:
        hx, hy = hover
        line(f"Cell ({hx}, {hy})")
        if info["best"] is not None and info["best"][hy, hx] >= RX_THRESHOLD:
            line(f"Signal: {info['best'][hy, hx]:.1f} dBm")
            line(f"SIR: {info['sir'][hy, hx]:.1f} dB")
            line(f"Served by ch {towers[info['serving'][hy, hx]].channel + 1}")
        else:
            line("No service", MUTED)

    y = H - 170
    for txt in ["L/R-click: add/remove", "1-7: channel  UP/DN: power",
                "[ / ]: SIR threshold", "H: heat   O: circles",
                "I: interference  G: grid", "A: all ch 1 (worst case)",
                "R: new map  C: clear", "S: screenshot"]:
        line(txt, MUTED, 20)


# ------------------------------- MAIN ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL + SIDEBAR, GRID_H * CELL))
    pygame.display.set_caption("Cellular Network Simulator - Day 3")
    fonts = (pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 20, bold=True),
             pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 15))
    tower_font = pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 13, bold=True)
    clock = pygame.time.Clock()

    rng = np.random.default_rng(SEED)
    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
    map_surface = render_map(pop)

    towers = []
    next_ch, next_pw, sir_min = 0, POWER_DEFAULT, SIR_MIN_DEFAULT
    show_grid = show_heat = show_circles = show_interf = True
    heat_overlay = interf_overlay = None
    info = {"best": None, "sir": None, "serving": None, "covered_pct": 0.0, "bad_pct": 0.0}
    dirty = True

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
                    towers.append(Tower(hover[0], hover[1], next_pw, next_ch)); dirty = True
                elif event.button == 3 and hover_tower:
                    towers.remove(hover_tower); dirty = True

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if k == pygame.K_ESCAPE:
                    running = False
                elif pygame.K_1 <= k < pygame.K_1 + N_CHANNELS:
                    if hover_tower:
                        hover_tower.channel = k - pygame.K_1; dirty = True
                    else:
                        next_ch = k - pygame.K_1
                elif k in (pygame.K_UP, pygame.K_DOWN):
                    d = 1 if k == pygame.K_UP else -1
                    if hover_tower:
                        hover_tower.power_dbm = int(np.clip(hover_tower.power_dbm + d, POWER_MIN, POWER_MAX))
                        hover_tower.update(); dirty = True
                    else:
                        next_pw = int(np.clip(next_pw + d, POWER_MIN, POWER_MAX))
                elif k == pygame.K_LEFTBRACKET:
                    sir_min = max(0.0, sir_min - 1); dirty = True
                elif k == pygame.K_RIGHTBRACKET:
                    sir_min = min(30.0, sir_min + 1); dirty = True
                elif k == pygame.K_a:
                    for t in towers:
                        t.channel = 0
                    dirty = True
                elif k == pygame.K_h:
                    show_heat = not show_heat
                elif k == pygame.K_o:
                    show_circles = not show_circles
                elif k == pygame.K_i:
                    show_interf = not show_interf
                elif k == pygame.K_g:
                    show_grid = not show_grid
                elif k == pygame.K_c:
                    towers.clear(); dirty = True
                elif k == pygame.K_r:
                    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
                    map_surface = render_map(pop)
                elif k == pygame.K_s:
                    pygame.image.save(screen, "screenshot_day3.png")
                    print("Saved screenshot_day3.png")

        if dirty:
            if towers:
                best, sir, serving = compute_interference(towers)
                covered = best >= RX_THRESHOLD
                bad = covered & (sir < sir_min)
                heat_overlay = build_overlay(best)
                interf_overlay = build_interference_overlay(sir, bad, sir_min)
                info = {"best": best, "sir": sir, "serving": serving,
                        "covered_pct": covered.mean() * 100,
                        "bad_pct": bad.sum() / covered.sum() * 100 if covered.any() else 0.0}
            else:
                heat_overlay = interf_overlay = None
                info = {"best": None, "sir": None, "serving": None, "covered_pct": 0.0, "bad_pct": 0.0}
            dirty = False

        screen.fill(BG)
        screen.blit(map_surface, (0, 0))
        if show_heat and heat_overlay:
            screen.blit(heat_overlay, (0, 0))
        if show_interf and interf_overlay:
            screen.blit(interf_overlay, (0, 0))
        if show_grid:
            draw_grid_lines(screen)
        if hover and not hover_tower:
            pygame.draw.rect(screen, (255, 255, 255), (hover[0] * CELL, hover[1] * CELL, CELL, CELL), 2)
        draw_towers(screen, towers, tower_font, show_circles, hover_tower)
        draw_sidebar(screen, fonts, towers, hover, hover_tower, next_ch, next_pw, sir_min, info)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()