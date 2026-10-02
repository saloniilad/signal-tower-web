"""
Cellular Network Simulator - Scoring System & Budget
-----------------------------------------------------------
Needs city_grid.py, towers.py, interference.py in the same folder.

Cost of a tower  = BASE_COST + COST_PER_CELL2 * radius_cells^2   (bigger area = pricier)
Budget           = BUDGET total; remaining = BUDGET - sum(tower costs)
                   (removing a tower refunds it in full)

Score = coverage%  -  interference penalty  -  cost penalty
    coverage%            population-weighted % of the city with signal >= threshold
    interference penalty INT_WEIGHT  * population-weighted % of the city that is red
    cost penalty         COST_WEIGHT * % of the budget spent

Controls: same as Day 3. Placement/upgrades are refused if they exceed the budget.
"""

import numpy as np
import pygame

from city_grid import (
    GRID_W, GRID_H, CELL, SIDEBAR, N_ZONES, SEED, MANUAL_ZONES,
    BG, PANEL, TEXT, MUTED, generate_population, render_map, draw_grid_lines,
)
from towers import (
    Tower, N_CHANNELS, CHANNEL_COLORS, RX_THRESHOLD, POWER_MIN, POWER_MAX,
    POWER_DEFAULT, radius_for_power, tower_at, build_overlay, draw_towers,
)
from interference import (
    compute_interference, build_interference_overlay, SIR_MIN_DEFAULT,
)

# ----------------------------- CONFIG ---------------------------------
BUDGET = 1000
BASE_COST = 40
COST_PER_CELL2 = 1.5      # cost per (cell radius)^2
INT_WEIGHT = 1.0          # 1.0 = interfered coverage counts for nothing
COST_WEIGHT = 0.2         # spending the whole budget costs 20 score points


def tower_cost(power_dbm):
    return int(round(BASE_COST + COST_PER_CELL2 * radius_for_power(power_dbm) ** 2))


def total_spent(towers):
    return sum(tower_cost(t.power_dbm) for t in towers)


def compute_score(pop, covered, bad, spent):
    total = pop.sum()
    cov = pop[covered].sum() / total * 100
    intf = INT_WEIGHT * pop[bad].sum() / total * 100
    cost_pen = COST_WEIGHT * spent / BUDGET * 100
    return cov, intf, cost_pen, cov - intf - cost_pen


EMPTY = {"best": None, "sir": None, "serving": None,
         "cov": 0.0, "intf": 0.0, "score_pts": 0.0}


# ----------------------------- SIDEBAR --------------------------------
def draw_sidebar(screen, fonts, towers, hover, hover_tower, next_ch, next_pw, sir_min, info):
    big, small, huge = fonts
    x0, H = GRID_W * CELL, GRID_H * CELL
    pygame.draw.rect(screen, PANEL, (x0, 0, SIDEBAR, H))
    px, y = x0 + 16, 12
    spent = total_spent(towers)

    def line(text, color=TEXT, dy=20):
        nonlocal y
        screen.blit(small.render(text, True, color), (px, y))
        y += dy

    screen.blit(big.render("Cell Network Sim", True, TEXT), (px, y)); y += 26
    line("Day 4 - Score & budget", MUTED, 26)

    # Tower being placed / edited
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
    line(f"Power: {pw} dBm (r {radius_for_power(pw):.1f})")
    line(f"Cost: {tower_cost(pw)}", (250, 204, 21), 26)

    # Score HUD
    score = info["score_pts"]
    color = (74, 222, 128) if score >= 50 else (250, 204, 21) if score >= 0 else (248, 113, 113)
    line("SCORE", MUTED, 18)
    screen.blit(huge.render(f"{score:.1f}", True, color), (px, y)); y += 40
    line(f"Coverage:      +{info['cov']:.1f}", (74, 222, 128))
    line(f"Interference:  -{info['intf']:.1f}", (248, 113, 113))
    line(f"Cost penalty:  -{COST_WEIGHT * spent / BUDGET * 100:.1f}", (248, 113, 113), 24)

    # Budget
    line(f"Budget: {BUDGET - spent} / {BUDGET}", TEXT, 22)
    bar = pygame.Rect(px, y, SIDEBAR - 32, 12)
    pygame.draw.rect(screen, (55, 65, 81), bar, border_radius=6)
    frac = min(spent / BUDGET, 1)
    if frac > 0:
        fill = bar.copy(); fill.width = max(int(bar.width * frac), 8)
        pygame.draw.rect(screen, (250, 204, 21) if frac < 0.8 else (248, 113, 113), fill, border_radius=6)
    y += 22
    line(f"Towers: {len(towers)}   SIR>={sir_min:.0f} dB", TEXT, 24)

    # Hover info
    if hover:
        hx, hy = hover
        line(f"Cell ({hx}, {hy})")
        if info["best"] is not None and info["best"][hy, hx] >= RX_THRESHOLD:
            line(f"Signal: {info['best'][hy, hx]:.1f} dBm")
            line(f"SIR: {info['sir'][hy, hx]:.1f} dB")
        else:
            line("No service", MUTED)

    y = H - 130
    for txt in ["L/R-click: add/remove", "1-7: channel  UP/DN: power",
                "[ ]: SIR thr.  A: all ch1", "H heat  O circles  I red",
                "G grid  R map  C clear", "S: screenshot"]:
        line(txt, MUTED, 20)


# ------------------------------- MAIN ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL + SIDEBAR, GRID_H * CELL))
    pygame.display.set_caption("Cellular Network Simulator - Day 4")
    face = "consolas,menlo,dejavusansmono,arial"
    fonts = (pygame.font.SysFont(face, 20, bold=True),
             pygame.font.SysFont(face, 15),
             pygame.font.SysFont(face, 34, bold=True))
    tower_font = pygame.font.SysFont(face, 13, bold=True)
    clock = pygame.time.Clock()

    rng = np.random.default_rng(SEED)
    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
    map_surface = render_map(pop)

    towers = []
    next_ch, next_pw, sir_min = 0, POWER_DEFAULT, SIR_MIN_DEFAULT
    show_grid = show_heat = show_circles = show_interf = True
    heat_overlay = interf_overlay = None
    info = dict(EMPTY)
    dirty = True
    flash_msg, flash_until = "", 0

    def flash(msg):
        nonlocal flash_msg, flash_until
        flash_msg, flash_until = msg, pygame.time.get_ticks() + 2000

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
                    cost = tower_cost(next_pw)
                    if total_spent(towers) + cost <= BUDGET:
                        towers.append(Tower(hover[0], hover[1], next_pw, next_ch)); dirty = True
                    else:
                        flash(f"Not enough budget! (need {cost})")
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
                        new_pw = int(np.clip(hover_tower.power_dbm + d, POWER_MIN, POWER_MAX))
                        delta = tower_cost(new_pw) - tower_cost(hover_tower.power_dbm)
                        if total_spent(towers) + delta <= BUDGET:
                            hover_tower.power_dbm = new_pw
                            hover_tower.update(); dirty = True
                        else:
                            flash("Not enough budget to upgrade!")
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
                    map_surface = render_map(pop); dirty = True
                elif k == pygame.K_s:
                    pygame.image.save(screen, "screenshot_day4.png")
                    print("Saved screenshot_day4.png")

        if dirty:
            if towers:
                best, sir, serving = compute_interference(towers)
                covered = best >= RX_THRESHOLD
                bad = covered & (sir < sir_min)
                heat_overlay = build_overlay(best)
                interf_overlay = build_interference_overlay(sir, bad, sir_min)
                cov, intf, _, pts = compute_score(pop, covered, bad, total_spent(towers))
                info = {"best": best, "sir": sir, "serving": serving,
                        "cov": cov, "intf": intf, "score_pts": pts}
            else:
                heat_overlay = interf_overlay = None
                info = dict(EMPTY)
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

        if pygame.time.get_ticks() < flash_until:
            txt = fonts[0].render(flash_msg, True, (255, 255, 255))
            box = txt.get_rect(center=(GRID_W * CELL // 2, GRID_H * CELL - 30)).inflate(24, 14)
            pygame.draw.rect(screen, (185, 28, 28), box, border_radius=8)
            screen.blit(txt, txt.get_rect(center=box.center))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()