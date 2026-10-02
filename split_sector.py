"""
Cellular Network Simulator - Day 5: Cell Splitting & Sectoring
--------------------------------------------------------------
Needs day1..day4 files in the same folder.

SECTORING (T): hovered omni tower -> 3 directional sectors (120 deg each) at the same
    site, each with its own channel. Sector signal = path loss + antenna pattern
    (+6 dB boresight gain, 3GPP-style -12*(angle/65deg)^2 dB, capped at -20 dB), so a sector radiates
    mostly forward and hears/causes far less interference sideways and behind.
CELL SPLITTING (X = 2 cells, V = 3 cells): hovered omni tower is replaced by smaller,
    lower-power towers (-11 dB ~ half the radius) placed around the old site on
    different channels.
LOAD: each tower serves the population of the cells it wins. Capacity per tower/sector
    is LOAD_CAP; a tower above it gets an orange ring ("overloaded").

Hover: an omni tower's cell, or within 2 cells of a sectored site (the wedge under the
mouse is the selected sector). 1-7 / UP / DOWN edit only the selected tower/sector.
Right-click removes the whole site.
"""

import math
import numpy as np
import pygame

from city_grid import (
    GRID_W, GRID_H, CELL, SIDEBAR, N_ZONES, SEED, MANUAL_ZONES,
    BG, PANEL, TEXT, MUTED, generate_population, render_map, draw_grid_lines,
)
from towers import (
    Tower, XS, YS, N_CHANNELS, CHANNEL_COLORS, RX_THRESHOLD, POWER_MIN, POWER_MAX,
    POWER_DEFAULT, radius_for_power, build_overlay,
)
from interference import compute_interference, build_interference_overlay, SIR_MIN_DEFAULT
from scoring import BUDGET, COST_WEIGHT, tower_cost, compute_score, EMPTY

# ----------------------------- CONFIG ---------------------------------
BEAMWIDTH = 65.0          # 3 dB beamwidth of a sector antenna (deg)
SIDE_LOBE_DB = 20.0       # max attenuation off-boresight
SECTOR_GAIN_DB = 6.0      # boresight gain of a sector antenna vs an omni antenna
SECTOR_AZIMUTHS = (90, 210, 330)   # screen angles: 0 = right, 90 = down
SECTOR_COST_FACTOR = 0.5  # each sector costs 0.5x an omni -> 3 sectors = 1.5x
SPLIT_DROP_DB = 11.0      # ~ halves the radius (37.6*log10(2) = 11.3 dB)
LOAD_CAP = 30.0           # population weight one tower/sector can serve
PICK_RADIUS = 2           # cells: hover distance to select a sector


class Tower5(Tower):
    """Tower that may be a directional sector (azimuth in degrees) or omni (None)."""

    def __init__(self, col, row, power_dbm, channel, azimuth=None):
        self.azimuth = azimuth
        self.cost_factor = SECTOR_COST_FACTOR if azimuth is not None else 1.0
        super().__init__(col, row, power_dbm, channel)

    def update(self):
        super().update()
        if self.azimuth is not None:
            theta = np.degrees(np.arctan2(YS - self.row, XS - self.col))
            diff = (theta - self.azimuth + 180) % 360 - 180
            diff[self.row, self.col] = 0
            self.signal = self.signal + SECTOR_GAIN_DB - np.minimum(12 * (diff / BEAMWIDTH) ** 2, SIDE_LOBE_DB)


def _sector_radius(self):
    gain = SECTOR_GAIN_DB if self.azimuth is not None else 0.0
    return radius_for_power(self.power_dbm + gain)


Tower5.radius_cells = property(_sector_radius)


def cost_of(t):
    return int(round(tower_cost(t.power_dbm) * t.cost_factor))


def total_spent(towers):
    return sum(cost_of(t) for t in towers)


def spread_channels(c, n):
    """n channels starting at c, spaced by 2 so none are adjacent."""
    return [(c + 2 * j) % N_CHANNELS for j in range(n)]


def pick(towers, cell, mouse):
    for t in towers:
        if t.azimuth is None and (t.col, t.row) == cell:
            return t
    for t in towers:
        if t.azimuth is not None:
            cx, cy = t.center_px
            dx, dy = mouse[0] - cx, mouse[1] - cy
            if dx * dx + dy * dy <= (PICK_RADIUS * CELL) ** 2:
                diff = (math.degrees(math.atan2(dy, dx)) - t.azimuth + 180) % 360 - 180
                if abs(diff) <= 60:
                    return t
    return None


def try_sector(parent):
    if parent.azimuth is not None:
        return None, "Already sectored"
    chans = spread_channels(parent.channel, 3)
    return [Tower5(parent.col, parent.row, parent.power_dbm, c, az)
            for az, c in zip(SECTOR_AZIMUTHS, chans)], None


def try_split(towers, parent, n):
    r = parent.radius_cells
    if parent.azimuth is not None:
        return None, "Sectored sites can't be split"
    if r < 4:
        return None, "Cell too small to split"
    power = max(POWER_MIN, parent.power_dbm - SPLIT_DROP_DB)
    angles = [0, 180] if n == 2 else [90, 210, 330]
    occupied = {(t.col, t.row) for t in towers if t is not parent}
    kids = []
    for a, c in zip(angles, spread_channels(parent.channel, n)):
        col = int(round(parent.col + 0.5 * r * math.cos(math.radians(a))))
        row = int(round(parent.row + 0.5 * r * math.sin(math.radians(a))))
        if not (0 <= col < GRID_W and 0 <= row < GRID_H) or (col, row) in occupied:
            return None, "No room to split here"
        occupied.add((col, row))
        kids.append(Tower5(col, row, power, c))
    return kids, None


# ----------------------------- RENDERING ------------------------------
def draw_towers5(screen, towers, font, show_circles, hover_tower, load):
    screen.set_clip(pygame.Rect(0, 0, GRID_W * CELL, GRID_H * CELL))
    if show_circles:
        for t in towers:
            col, R, c = CHANNEL_COLORS[t.channel], int(t.radius_cells * CELL), t.center_px
            if t.azimuth is None:
                pygame.draw.circle(screen, col, c, R, 2)
            else:
                pts = [c] + [(c[0] + R * math.cos(math.radians(t.azimuth - 60 + 10 * k)),
                              c[1] + R * math.sin(math.radians(t.azimuth - 60 + 10 * k)))
                             for k in range(13)]
                pygame.draw.polygon(screen, col, pts, 4 if t is hover_tower else 2)
    for i, t in enumerate(towers):
        col, c = CHANNEL_COLORS[t.channel], t.center_px
        if load is not None and load[i] > LOAD_CAP:
            pygame.draw.circle(screen, (251, 146, 60), c, 16 if t.azimuth is None else 21, 3)
        if t.azimuth is None:
            p = c
            pygame.draw.circle(screen, (255, 255, 255), p, 9)
            pygame.draw.circle(screen, col, p, 7)
        else:
            a = math.radians(t.azimuth)
            p = (int(c[0] + 12 * math.cos(a)), int(c[1] + 12 * math.sin(a)))
            pygame.draw.circle(screen, (255, 255, 255), c, 3)
            pygame.draw.circle(screen, (255, 255, 255), p, 9)
            pygame.draw.circle(screen, col, p, 7)
        if t is hover_tower:
            pygame.draw.circle(screen, (255, 255, 255), p, 13, 2)
        label = font.render(str(t.channel + 1), True, (0, 0, 0))
        screen.blit(label, label.get_rect(center=p))
    screen.set_clip(None)


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
    line("Day 5 - Split & sector", MUTED, 26)

    tgt = hover_tower
    ch = tgt.channel if tgt else next_ch
    pw = tgt.power_dbm if tgt else next_pw
    line(("Editing hovered sector" if tgt and tgt.azimuth is not None else
          "Editing hovered tower" if tgt else "Next tower to place"), TEXT, 22)
    for i, c in enumerate(CHANNEL_COLORS):
        r = pygame.Rect(px + i * 31, y, 27, 22)
        pygame.draw.rect(screen, c, r, border_radius=4)
        if i == ch:
            pygame.draw.rect(screen, (255, 255, 255), r, 3, border_radius=4)
        n = small.render(str(i + 1), True, (0, 0, 0))
        screen.blit(n, n.get_rect(center=r.center))
    y += 30
    line(f"Power: {pw} dBm (r {radius_for_power(pw):.1f})")
    line(f"Cost: {cost_of(tgt) if tgt else tower_cost(next_pw)}", (250, 204, 21), 26)

    score = info["score_pts"]
    color = (74, 222, 128) if score >= 50 else (250, 204, 21) if score >= 0 else (248, 113, 113)
    line("SCORE", MUTED, 18)
    screen.blit(huge.render(f"{score:.1f}", True, color), (px, y)); y += 40
    line(f"Coverage:      +{info['cov']:.1f}", (74, 222, 128))
    line(f"Interference:  {0.0 - info['intf']:.1f}", (248, 113, 113))
    line(f"Cost penalty:  {0.0 - COST_WEIGHT * spent / BUDGET * 100:.1f}", (248, 113, 113), 24)

    line(f"Budget: {BUDGET - spent} / {BUDGET}", TEXT, 22)
    bar = pygame.Rect(px, y, SIDEBAR - 32, 12)
    pygame.draw.rect(screen, (55, 65, 81), bar, border_radius=6)
    frac = min(spent / BUDGET, 1)
    if frac > 0:
        fill = bar.copy(); fill.width = max(int(bar.width * frac), 8)
        pygame.draw.rect(screen, (250, 204, 21) if frac < 0.8 else (248, 113, 113), fill, border_radius=6)
    y += 22
    line(f"Towers: {len(towers)}  Overloaded: {info['over']}",
         (251, 146, 60) if info["over"] else TEXT, 24)

    if hover:
        hx, hy = hover
        line(f"Cell ({hx}, {hy})")
        if tgt:
            line(f"{'Sector' if tgt.azimuth is not None else 'Omni'} tower, ch {tgt.channel + 1}")
            ld = info["load"][towers.index(tgt)]
            over = ld > LOAD_CAP
            line(f"Load: {ld:.0f} / {LOAD_CAP:.0f}" + (" OVER" if over else ""),
                 (251, 146, 60) if over else TEXT)
        elif info["best"] is not None and info["best"][hy, hx] >= RX_THRESHOLD:
            line(f"Signal: {info['best'][hy, hx]:.1f} dBm")
            line(f"SIR: {info['sir'][hy, hx]:.1f} dB")
        else:
            line("No service", MUTED)

    y = H - 150
    for txt in ["L/R-click: add/remove", "1-7: channel  UP/DN: power",
                "T: sectors  X/V: split 2/3", f"[ ]: SIR thr ({sir_min:.0f} dB)  A: ch1",
                "H heat  O circles  I red", "G grid  R map  C clear", "S: screenshot"]:
        line(txt, MUTED, 20)


# ------------------------------- MAIN ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL + SIDEBAR, GRID_H * CELL))
    pygame.display.set_caption("Cellular Network Simulator - Day 5")
    face = "consolas,menlo,dejavusansmono,arial"
    fonts = (pygame.font.SysFont(face, 20, bold=True), pygame.font.SysFont(face, 15),
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
    info = {**EMPTY, "load": None, "over": 0}
    dirty = True
    flash_msg, flash_until = "", 0

    def flash(msg):
        nonlocal flash_msg, flash_until
        flash_msg, flash_until = msg, pygame.time.get_ticks() + 2000

    def replace(parent, kids):
        """Swap parent for kids if the budget allows."""
        delta = sum(cost_of(k) for k in kids) - cost_of(parent)
        if total_spent(towers) + delta > BUDGET:
            flash(f"Not enough budget! (need +{delta})")
            return False
        towers.remove(parent)
        towers.extend(kids)
        return True

    running = True
    while running:
        mx, my = pygame.mouse.get_pos()
        hover = (mx // CELL, my // CELL) if mx < GRID_W * CELL and my < GRID_H * CELL else None
        hover_tower = pick(towers, hover, (mx, my)) if hover else None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN and hover:
                if event.button == 1 and not hover_tower:
                    cost = tower_cost(next_pw)
                    if total_spent(towers) + cost <= BUDGET:
                        towers.append(Tower5(hover[0], hover[1], next_pw, next_ch)); dirty = True
                    else:
                        flash(f"Not enough budget! (need {cost})")
                elif event.button == 3 and hover_tower:
                    site = (hover_tower.col, hover_tower.row)
                    towers[:] = [t for t in towers if (t.col, t.row) != site]; dirty = True

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
                        delta = int(round(tower_cost(new_pw) * hover_tower.cost_factor)) - cost_of(hover_tower)
                        if total_spent(towers) + delta <= BUDGET:
                            hover_tower.power_dbm = new_pw
                            hover_tower.update(); dirty = True
                        else:
                            flash("Not enough budget to upgrade!")
                    else:
                        next_pw = int(np.clip(next_pw + d, POWER_MIN, POWER_MAX))
                elif k in (pygame.K_t, pygame.K_x, pygame.K_v):
                    if not hover_tower:
                        flash("Hover a tower first")
                    else:
                        if k == pygame.K_t:
                            kids, err = try_sector(hover_tower)
                        else:
                            kids, err = try_split(towers, hover_tower, 2 if k == pygame.K_x else 3)
                        if err:
                            flash(err)
                        elif replace(hover_tower, kids):
                            dirty = True
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
                    pygame.image.save(screen, "screenshot_day5.png")
                    print("Saved screenshot_day5.png")

        if hover_tower not in towers:
            hover_tower = None

        if dirty:
            if towers:
                best, sir, serving = compute_interference(towers)
                covered = best >= RX_THRESHOLD
                bad = covered & (sir < sir_min)
                heat_overlay = build_overlay(best)
                interf_overlay = build_interference_overlay(sir, bad, sir_min)
                cov, intf, _, pts = compute_score(pop, covered, bad, total_spent(towers))
                load = np.bincount(serving[covered], weights=pop[covered], minlength=len(towers))
                info = {"best": best, "sir": sir, "serving": serving, "cov": cov, "intf": intf,
                        "score_pts": pts, "load": load, "over": int((load > LOAD_CAP).sum())}
            else:
                heat_overlay = interf_overlay = None
                info = {**EMPTY, "load": None, "over": 0}
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
        draw_towers5(screen, towers, tower_font, show_circles, hover_tower, info["load"])
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