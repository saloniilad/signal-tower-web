"""
Cellular Network Simulator - Setup & Core Grid
-----------------------------------------------------
- 2D NumPy grid representing the city
- Population "zones" (Gaussian hotspots) = where coverage matters most
- Pygame rendering with density shading + legend + hover info
- Matplotlib export of the density map (for your report)

Controls:
    R  - regenerate a new random city
    G  - toggle grid lines
    P  - save density map with Matplotlib (population_map.png)
    S  - save screenshot of the Pygame window (screenshot.png)
    ESC / close window - quit
"""

import numpy as np
import pygame

# Matplotlib is only used for the optional PNG export. It is not available in the
# browser (Pygbag), so import it defensively; everything else works without it.
try:
    import matplotlib
    matplotlib.use("Agg")  # save to file only, so it never fights with the Pygame window
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

# ----------------------------- CONFIG ---------------------------------
GRID_W, GRID_H = 40, 30      # number of cells (columns, rows)
CELL = 20                    # pixel size of each cell
SIDEBAR = 250                # width of the info panel on the right
N_ZONES = 6                  # number of random population hotspots
SEED = None                  # set to an int (e.g. 42) for a repeatable map

# Optional: manually placed zones -> (col, row, sigma, weight). Leave [] for random.
MANUAL_ZONES = []
# e.g. MANUAL_ZONES = [(10, 8, 3, 1.0), (30, 20, 4, 0.8)]

BG = (17, 24, 39)
PANEL = (31, 41, 55)
TEXT = (229, 231, 235)
MUTED = (156, 163, 175)
GRID_LINE = (0, 0, 0)

# Density color ramp: (value, RGB)
STOPS = [
    (0.00, (15, 23, 42)),
    (0.25, (30, 64, 175)),
    (0.50, (16, 185, 129)),
    (0.75, (250, 204, 21)),
    (1.00, (220, 38, 38)),
]


# --------------------------- CITY MODEL -------------------------------
def generate_population(w, h, n_zones, rng, manual=None):
    """Return an (h, w) array of population weights in [0, 1]."""
    ys, xs = np.mgrid[0:h, 0:w]
    pop = np.zeros((h, w), dtype=float)

    if manual:
        zones = manual
    else:
        zones = [
            (rng.uniform(2, w - 2), rng.uniform(2, h - 2),
             rng.uniform(2.0, 5.0), rng.uniform(0.5, 1.0))
            for _ in range(n_zones)
        ]

    for cx, cy, sigma, weight in zones:
        pop += weight * np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * sigma ** 2))

    pop += rng.uniform(0, 0.05, size=(h, w))  # sparse background population
    return pop / pop.max()


# ---------------------------- RENDERING -------------------------------
def density_to_color(v):
    """Map a value in [0,1] to an RGB color using STOPS."""
    v = float(np.clip(v, 0, 1))
    for (v0, c0), (v1, c1) in zip(STOPS, STOPS[1:]):
        if v <= v1:
            t = (v - v0) / (v1 - v0)
            return tuple(int(c0[i] + t * (c1[i] - c0[i])) for i in range(3))
    return STOPS[-1][1]


def render_map(pop):
    """Pre-render the population map onto a Surface (only redone on regenerate)."""
    h, w = pop.shape
    surf = pygame.Surface((w * CELL, h * CELL))
    for y in range(h):
        for x in range(w):
            pygame.draw.rect(surf, density_to_color(pop[y, x]),
                             (x * CELL, y * CELL, CELL, CELL))
    return surf


def draw_grid_lines(screen):
    for x in range(0, GRID_W * CELL + 1, CELL):
        pygame.draw.line(screen, GRID_LINE, (x, 0), (x, GRID_H * CELL), 1)
    for y in range(0, GRID_H * CELL + 1, CELL):
        pygame.draw.line(screen, GRID_LINE, (0, y), (GRID_W * CELL, y), 1)


def draw_sidebar(screen, fonts, pop, hover):
    big, small = fonts
    x0 = GRID_W * CELL
    H = GRID_H * CELL
    pygame.draw.rect(screen, PANEL, (x0, 0, SIDEBAR, H))
    px = x0 + 16
    y = 14

    screen.blit(big.render("Cell Network Sim", True, TEXT), (px, y)); y += 26
    screen.blit(small.render("Day 1 - Population map", True, MUTED), (px, y)); y += 30

    # Legend color bar (top = high density)
    screen.blit(small.render("Population density", True, TEXT), (px, y)); y += 22
    bar_h, bar_w = 150, 22
    for i in range(bar_h):
        pygame.draw.line(screen, density_to_color(1 - i / (bar_h - 1)),
                         (px, y + i), (px + bar_w, y + i))
    screen.blit(small.render("High", True, MUTED), (px + bar_w + 10, y))
    screen.blit(small.render("Low", True, MUTED), (px + bar_w + 10, y + bar_h - 16))
    y += bar_h + 20

    # Stats
    screen.blit(small.render(f"Grid: {GRID_W} x {GRID_H} cells", True, TEXT), (px, y)); y += 20
    screen.blit(small.render(f"Total pop. weight: {pop.sum():.1f}", True, TEXT), (px, y)); y += 20
    hot = int((pop > 0.6).sum())
    screen.blit(small.render(f"High-demand cells: {hot}", True, TEXT), (px, y)); y += 28

    # Hover info
    if hover:
        hx, hy = hover
        screen.blit(small.render(f"Cell ({hx}, {hy})", True, TEXT), (px, y)); y += 20
        screen.blit(small.render(f"Density: {pop[hy, hx]:.2f}", True, TEXT), (px, y))
    y = H - 110
    for line in ["R: new map", "G: toggle grid", "P: save Matplotlib PNG",
                 "S: save screenshot", "ESC: quit"]:
        screen.blit(small.render(line, True, MUTED), (px, y)); y += 20


def save_matplotlib_map(pop, filename="population_map.png"):
    if plt is None:
        print("Matplotlib not available here - skipping PNG export")
        return
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(pop, cmap="viridis", origin="upper")
    ax.set_title("City Population Density")
    ax.set_xlabel("Column"); ax.set_ylabel("Row")
    fig.colorbar(im, ax=ax, label="Relative population weight")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    print(f"Saved {filename}")


# ------------------------------- MAIN ---------------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((GRID_W * CELL + SIDEBAR, GRID_H * CELL))
    pygame.display.set_caption("Cellular Network Simulator - Day 1")
    fonts = (pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 20, bold=True),
             pygame.font.SysFont("consolas,menlo,dejavusansmono,arial", 15))
    clock = pygame.time.Clock()

    rng = np.random.default_rng(SEED)
    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
    map_surface = render_map(pop)
    show_grid = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    pop = generate_population(GRID_W, GRID_H, N_ZONES, rng, MANUAL_ZONES)
                    map_surface = render_map(pop)
                elif event.key == pygame.K_g:
                    show_grid = not show_grid
                elif event.key == pygame.K_p:
                    save_matplotlib_map(pop)
                elif event.key == pygame.K_s:
                    pygame.image.save(screen, "screenshot.png")
                    print("Saved screenshot.png")

        # Hover cell
        mx, my = pygame.mouse.get_pos()
        hover = None
        if mx < GRID_W * CELL and my < GRID_H * CELL:
            hover = (mx // CELL, my // CELL)

        screen.fill(BG)
        screen.blit(map_surface, (0, 0))
        if show_grid:
            draw_grid_lines(screen)
        if hover:
            pygame.draw.rect(screen, (255, 255, 255),
                             (hover[0] * CELL, hover[1] * CELL, CELL, CELL), 2)
        draw_sidebar(screen, fonts, pop, hover)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


if __name__ == "__main__":
    main()