"""
Signal Tower - a friendly game version of the Cellular Network Simulator
Put next to city_grid.py ... split_sector.py and run:  python signal_tower.py

Click map = build/select tower | right-click (or Delete) = remove | buttons on the right = color, size, tools
Six levels:  reuse -> budget -> capacity -> splitting & sectors -> UMBRELLA CELL -> demand surge.
Dev shortcut: python signal_tower.py --level 5   (start at level 5)
"""
import asyncio
import os
import sys
import math
import numpy as np
import pygame

# The browser build does not always load the mixer automatically, so import it explicitly.
try:
    import pygame.mixer
    HAVE_MIXER = True
except Exception as ex:
    print("mixer import failed:", ex)
    HAVE_MIXER = False

from city_grid import GRID_W, GRID_H, CELL, generate_population
from towers import RX_THRESHOLD, radius_for_power, XS, YS
from interference import compute_interference
from scoring import tower_cost
from split_sector import Tower5, pick, SECTOR_AZIMUTHS

WEB = sys.platform == "emscripten"      # True when running in the browser (Pygbag)
if WEB:                                 # browser: skip pygame mixer, use Web Audio instead
    HAVE_MIXER = False
HERE = os.path.dirname(os.path.abspath(__file__))
MAP_W, MAP_H = GRID_W * CELL, GRID_H * CELL
TIP_H, PANEL_W = 112, 300
W, H = MAP_W + PANEL_W, MAP_H + TIP_H

INK, SOFT, BG = (40, 48, 66), (125, 134, 152), (243, 241, 236)
GOLD, GREEN, RED = (250, 176, 5), (34, 197, 130), (235, 64, 82)
ORANGE, LAV = (251, 146, 60), (150, 130, 255)
COLORS = [(255, 107, 107), (255, 190, 11), (46, 204, 143), (86, 140, 255)]
NAMES = ["Coral", "Sunny", "Mint", "Sky"]
CHANNELS = [0, 2, 4, 6]
UMB_P = 40                      # umbrella cell = one giant tower on its own layer
UMB_CAP = 100.0                 # people an umbrella can carry
UMB_FACTOR = 0.6                # umbrella price vs a normal tower of the same power
SIZES = [("Small", 22), ("Medium", 30), ("Large", 38), ("Umbrella", UMB_P)]
HOUSES = [(255, 166, 87), (255, 205, 112), (244, 143, 130), (255, 183, 150)]
FOG = (62, 74, 108)
TXT, TXT2, PBG, CARD = (240, 244, 255), (150, 160, 190), (24, 30, 54), (38, 46, 80)


def lvl(**kw):
    base = dict(sir=12, cap=30.0, pen=False, sect=False, umb=0, event=False, hints=3, thrift=.6)
    base.update(kw)
    return base


# target = % of people with GOOD signal.  pen = overloaded towers drop people from the score.
LEVELS = [
    lvl(name="Say hello", target=65, budget=550, colors=4,
        story="Your city has no signal. Build towers near the houses!"),
    lvl(name="Colors to the rescue", target=78, budget=650, colors=4,
        story="Static everywhere? Give neighbours different colors, or try Auto-color!"),
    lvl(name="Smart planner", target=68, budget=650, colors=3, sir=13, cap=40.0, pen=True, hints=2, thrift=.7,
        story="Tight budget, 3 colors, and crowded towers drop people!"),
    lvl(name="Crowd control", target=72, budget=800, colors=3, sir=14, cap=40.0, pen=True, sect=True, hints=2, thrift=.75,
        story="Downtown is packed. Split cells or add sectors."),
    lvl(name="Under the umbrella", target=80, budget=850, colors=3, sir=14, cap=40.0, pen=True, sect=True, umb=1,
        hints=1, thrift=.8, story="Try the umbrella cell: it fills gaps and catches overflow."),
    lvl(name="Concert night", target=72, budget=900, colors=3, sir=15, cap=40.0, pen=True, sect=True, umb=1,
        event=True, hints=1, thrift=.8, story="First reach the target... then a crowd arrives!"),
]
CONCEPTS = {
    "frequency": ("Frequency (channel)", "A tower talks to phones on a radio frequency. In this game each color is a different frequency."),
    "coverage": ("Coverage", "The area where the signal is strong enough to use. Fog on the map means no coverage."),
    "cci": ("Co-channel interference", "Towers on the SAME frequency disturb each other. More towers means more of it. It shows up as red static."),
    "reuse": ("Frequency reuse", "Use each color again, but only far apart, so same-color towers barely hear each other."),
    "overlap": ("Overlap", "Two towers covering the same people. You paid twice but gained no new coverage."),
    "tradeoff": ("Trade-off: reach vs cost", "Bigger towers cover more but cost much more. Two medium towers can beat one large one."),
    "capacity": ("Capacity", "A tower can only serve so many people at once. A crowded tower gets an orange ring, and later on it starts dropping people."),
    "split": ("Cell splitting", "Cut a crowded cell into two smaller, lower-power cells. Each serves fewer people, so the same area gets more capacity."),
    "sector": ("Sectoring", "One site uses 3 directional antennas, each with its own color. Each antenna hears less static and serves its own share of people."),
    "umbrella": ("Umbrella cell", "One huge tower on its own layer sits over many small cells. It fills coverage holes and catches overflow from crowded towers, but it is costly."),
    "surge": ("Demand surge", "Crowds move! Concerts, matches and rush hour bring sudden traffic. Good plans keep spare capacity and a backup layer."),
}
QUIZ = [
    ("Why did red static appear when your towers were close together?",
     ["They shared one frequency and disturbed each other", "The towers ran out of coins", "The houses blocked the signal"], 0,
     "Right: that is co-channel interference. Same frequency + close together = they drown each other out."),
    ("How does frequency reuse fix static?",
     ["Give every tower the same color", "Reuse each color, but only far apart", "Switch the towers off at night"], 1,
     "Same-color towers far apart barely hear each other, so you get coverage AND no static."),
    ("A Large tower costs 313 coins, a Medium 143. Why can two Mediums be smarter?",
     ["Medium towers are louder", "Two Mediums can cover more places for less money", "Large towers never work"], 1,
     "Cost grows much faster than reach. Spreading smaller towers often covers more people per coin."),
    ("A crowded tower has an orange ring. Which fix adds capacity?",
     ["Paint it a brighter color", "Move it into a quiet forest", "Split it into smaller cells or add sectors"], 2,
     "Smaller cells and sector antennas mean each one serves fewer people, so the same area can carry more."),
    ("What does an umbrella cell do?",
     ["It only works when it rains", "It fills gaps and catches overflow from small cells", "It replaces every small tower"], 1,
     "An umbrella is a big, costly backup layer over small cells. It covers holes and takes the overflow."),
    ("A crowd suddenly appears where you have little coverage. Best plan?",
     ["Add capacity there; an umbrella is a good backup", "Ignore it and hope", "Turn every tower off"], 0,
     "Traffic moves. Plans with spare capacity and a backup layer survive surprises."),
]
LESSONS = [["frequency", "coverage", "cci"], ["reuse"], ["tradeoff", "overlap", "capacity"],
           ["split", "sector"], ["umbrella"], ["surge"]]


def wrap(font, text, width):
    lines, cur = [], ""
    for w in text.split():
        if font.size((cur + " " + w).strip())[0] <= width:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    return lines + [cur]


def star(s, cx, cy, r, col):
    pts = [(cx + (r if i % 2 == 0 else r * .45) * math.cos(-math.pi / 2 + i * math.pi / 5),
            cy + (r if i % 2 == 0 else r * .45) * math.sin(-math.pi / 2 + i * math.pi / 5)) for i in range(10)]
    pygame.draw.polygon(s, col, pts)


ROOFS = [(196, 90, 80), (150, 96, 84), (110, 120, 150)]


def draw_house(surf, x, y, wall, roof):
    pygame.draw.rect(surf, wall, (x + 3, y + 9, 14, 9), border_radius=1)
    pygame.draw.polygon(surf, roof, [(x + 1, y + 9), (x + 10, y + 2), (x + 19, y + 9)])
    pygame.draw.rect(surf, (120, 84, 70), (x + 8, y + 13, 4, 5))
    pygame.draw.rect(surf, (255, 244, 200), (x + 4, y + 11, 3, 3))


def draw_block(surf, x, y):
    pygame.draw.rect(surf, (168, 180, 208), (x + 4, y + 2, 12, 17), border_radius=2)
    for j in range(3):
        for i in range(2):
            pygame.draw.rect(surf, (255, 240, 170), (x + 6 + i * 5, y + 4 + j * 5, 3, 3))


def draw_umbrella(s, c, k=1.0):
    r = int(17 * k)
    pygame.draw.circle(s, (150, 150, 160), (c[0] + 1, c[1] + 3), r)
    pygame.draw.circle(s, (255, 255, 255), c, r)
    pygame.draw.circle(s, LAV, c, r - 3)
    canopy = [(c[0] + int(math.cos(math.pi + i * math.pi / 6) * 9), c[1] - 1 + int(math.sin(math.pi + i * math.pi / 6) * 9))
              for i in range(7)]
    pygame.draw.polygon(s, (255, 255, 255), canopy)
    pygame.draw.line(s, (255, 255, 255), (c[0], c[1] - 1), (c[0], c[1] + 7), 2)
    pygame.draw.arc(s, (255, 255, 255), (c[0] - 3, c[1] + 3, 6, 7), math.pi, 2 * math.pi, 2)


def make_map(pop, seed):
    """Houses = few people, tall blocks = many people, trees = empty land."""
    surf = pygame.Surface((MAP_W, MAP_H))
    surf.fill((226, 238, 218))
    rng = np.random.default_rng(seed)
    for r in range(GRID_H):
        for c in range(GRID_W):
            x, y, v = c * CELL, r * CELL, pop[r, c]
            if v > 0.55:
                draw_block(surf, x, y)
            elif v > 0.12:
                draw_house(surf, x, y, HOUSES[int(rng.integers(0, 4))], ROOFS[int(rng.integers(0, 3))])
            elif rng.random() < 0.15:
                pygame.draw.rect(surf, (140, 104, 80), (x + 9, y + 12, 3, 6))
                pygame.draw.circle(surf, (120, 190, 120), (x + 10, y + 9), 6)
    return surf


# ------------------------------ game model ------------------------------
def is_umb(t):
    return t.azimuth is None and t.power_dbm == UMB_P


def n_umb(towers):
    return sum(is_umb(t) for t in towers)


def price(p):
    return int(round(tower_cost(p) * (UMB_FACTOR if p == UMB_P else 1.0)))


def tcost(t):
    return int(round(tower_cost(t.power_dbm) * (UMB_FACTOR if is_umb(t) else t.cost_factor)))


def tcol(t):
    return LAV if is_umb(t) else COLORS[t.cidx]


def mpos(t):
    """Where the tower icon is drawn (sector antennas sit slightly off the site centre)."""
    c = t.center_px
    if t.azimuth is None:
        return c
    a = math.radians(t.azimuth)
    return int(c[0] + 12 * math.cos(a)), int(c[1] + 12 * math.sin(a))


def evaluate(towers, pop, lv):
    """Small towers compete + interfere on one layer; umbrella cells live on their own layer.
    Umbrellas (1) cover holes, (2) carry up to UMB_CAP people, (3) absorb overflow of towers under them."""
    if not towers:
        return None
    nt = [t for t in towers if not is_umb(t)]
    ut = [t for t in towers if is_umb(t)]
    tot = pop.sum()
    zero = np.zeros((GRID_H, GRID_W), bool)
    if nt:
        best, sir, serving = compute_interference(nt)
        cov_n = best >= RX_THRESHOLD
        bad = cov_n & (sir < lv["sir"])
        served = cov_n & ~bad
        load = np.bincount(serving[served], weights=pop[served], minlength=len(nt))
        hearing = (np.stack([t.signal for t in nt]) >= RX_THRESHOLD).sum(0)
    else:
        serving = np.zeros((GRID_H, GRID_W), int)
        cov_n = bad = zero
        load = np.zeros(0)
        hearing = np.zeros((GRID_H, GRID_W), int)
    umb = zero.copy()
    for u in ut:
        umb |= u.signal >= RX_THRESHOLD
    covered = cov_n | umb
    over = np.maximum(load - lv["cap"], 0)
    under = [i for i, t in enumerate(nt) if any(u.signal[t.row, t.col] >= RX_THRESHOLD for u in ut)]
    pool = float(over[under].sum()) if under else 0.0
    direct = float(pop[umb & ~cov_n].sum())
    ucap = UMB_CAP * len(ut)
    absorbed = min(pool, max(ucap - direct, 0.0))
    dropped = float(over.sum()) - absorbed + max(direct - ucap, 0.0)
    if not lv["pen"]:
        dropped = 0.0
    cov, clash, drop = pop[covered].sum() / tot * 100, pop[bad].sum() / tot * 100, dropped / tot * 100
    overlap = pop[hearing >= 2].sum() / max(pop[cov_n].sum(), 1e-9) * 100
    return dict(cov=cov, clash=clash, drop=drop, happy=max(cov - clash - drop, 0.0), covered=covered, cov_n=cov_n,
                bad=bad, serving=serving, nt=nt, ix={id(t): i for i, t in enumerate(nt)}, load=load,
                relieved=set(under) if absorbed >= pool - 1e-9 else set(), overlap=overlap)


def make_overlay(ev):
    """Fog where there is no signal; it lifts (with a color tint) where towers reach."""
    rgb = np.zeros((GRID_H, GRID_W, 3), np.uint8)
    rgb[:] = FOG
    alpha = np.full((GRID_H, GRID_W), 105, np.uint8)
    if ev:
        cov_n = ev["cov_n"]
        if ev["nt"]:
            cidx = np.array([t.cidx for t in ev["nt"]])[ev["serving"]]
            rgb = np.where(cov_n[..., None], np.array(COLORS, np.uint8)[cidx], rgb)
        only_u = ev["covered"] & ~cov_n
        rgb = np.where(only_u[..., None], np.array(LAV, np.uint8), rgb)
        rgb = np.where(ev["bad"][..., None], np.array(RED, np.uint8), rgb)
        alpha = np.where(ev["bad"], 165, np.where(cov_n, 45, np.where(only_u, 70, 105))).astype(np.uint8)
    rgba = np.dstack([rgb.astype(np.uint8), alpha.astype(np.uint8)])       # (H, W, 4)
    buf = np.ascontiguousarray(rgba).tobytes()                            # keep alive until scaled
    surf = pygame.image.frombuffer(buf, (GRID_W, GRID_H), "RGBA")
    return pygame.transform.smoothscale(surf, (MAP_W, MAP_H))


def auto_color(towers, n):
    """Give each tower the color whose nearest same-color tower is farthest away."""
    done = []
    for t in towers:
        if is_umb(t):
            continue

        def gap(i):
            d = [math.hypot(t.col - u.col, t.row - u.row) for u in done if u.cidx == i]
            return min(d) if d else 99
        t.cidx = max(range(n), key=lambda i: (gap(i), -i))
        t.channel = CHANNELS[t.cidx]
        done.append(t)


def best_spot(pop, ev, power, towers):
    """Cell where a tower of this size would reach the most people who have no signal yet."""
    r = radius_for_power(power)
    covered = ev["covered"] if ev else np.zeros((GRID_H, GRID_W), bool)
    occ = {(t.col, t.row) for t in towers}
    best, top = (GRID_W // 2, GRID_H // 2), -1
    for row in range(1, GRID_H, 2):
        for col in range(1, GRID_W, 2):
            if (col, row) in occ:
                continue
            sc = pop[(np.hypot(XS - col, YS - row) <= r) & ~covered].sum()
            if sc > top:
                best, top = (col, row), sc
    return best


def split_tower(t, towers, n):
    """Replace one tower with two smaller ones (cell splitting)."""
    if t.power_dbm <= 22 or t.azimuth is not None or is_umb(t):
        return None
    p = 22 if t.power_dbm <= 30 else 30
    d = max(3, int(round(t.radius_cells * .5)))
    occ = {(u.col, u.row) for u in towers if u is not t}
    kids = []
    for j, dx in enumerate((-d, d)):
        c = t.col + dx
        if not (0 <= c < GRID_W) or (c, t.row) in occ:
            return None
        k = Tower5(c, t.row, p, CHANNELS[(t.cidx + j) % n])
        k.cidx, k.born = (t.cidx + j) % n, pygame.time.get_ticks()
        kids.append(k)
    return kids


def sectorize(t, n):
    """Replace one omni tower with 3 directional antennas, each on its own color."""
    kids = []
    for j, az in enumerate(SECTOR_AZIMUTHS):
        ci = (t.cidx + j) % n
        k = Tower5(t.col, t.row, t.power_dbm, CHANNELS[ci], az)
        k.cidx, k.born = ci, pygame.time.get_ticks()
        kids.append(k)
    return kids


def draw_reach(surf, t, R, col, width):
    """Circle for an omni tower, 120-degree wedge for a sector antenna."""
    c = t.center_px
    if t.azimuth is None:
        pygame.draw.circle(surf, col, c, R, width)
    else:
        pts = [c] + [(c[0] + R * math.cos(math.radians(t.azimuth - 60 + 10 * k)),
                      c[1] + R * math.sin(math.radians(t.azimuth - 60 + 10 * k))) for k in range(13)]
        pygame.draw.polygon(surf, col, pts, width)


async def main():
    global HAVE_MIXER
    if HAVE_MIXER:
        try:
            pygame.mixer.pre_init(44100, -16, 1, 1024 if WEB else 256)
        except Exception as ex:
            print("mixer pre_init failed:", ex)
            HAVE_MIXER = False
    pygame.init()
    if HAVE_MIXER and not pygame.mixer.get_init():
        try:
            pygame.mixer.init(44100, -16, 1, 1024 if WEB else 256)
        except Exception as ex:
            print("mixer init failed:", ex)
            HAVE_MIXER = False
    print("HAVE_MIXER =", HAVE_MIXER)

    # Fallback for the browser: play the same beeps with the Web Audio API.
    web_beep = None
    if WEB and not HAVE_MIXER:
        try:
            from platform import window
            window.eval("""
            window.beep = function(freqs, ms){
                const ctx = window._ac || (window._ac = new AudioContext());
                if (ctx.state === 'suspended') ctx.resume();
                let t = ctx.currentTime;
                String(freqs).split(',').forEach(function(f){
                    const o = ctx.createOscillator(), g = ctx.createGain();
                    o.frequency.value = parseFloat(f);
                    o.connect(g); g.connect(ctx.destination);
                    g.gain.setValueAtTime(0.15, t);
                    g.gain.linearRampToValueAtTime(0, t + ms / 1000);
                    o.start(t); o.stop(t + ms / 1000);
                    t += ms / 1000;
                });
            };
            """)
            web_beep = window.beep
        except Exception as ex:
            print("web audio fallback failed:", ex)
    start = 0
    if "--level" in sys.argv:
        try:
            start = min(max(int(sys.argv[sys.argv.index("--level") + 1]) - 1, 0), len(LEVELS) - 1)
        except (ValueError, IndexError):
            pass
    if WEB:
        # The browser scales the canvas itself, so draw straight to a fixed-size window.
        screen = pygame.display.set_mode((W, H))
    else:
        info = pygame.display.Info()
        sc = min(1.0, (info.current_w - 30) / W, (info.current_h - 110) / H) if info.current_h > 0 else 1.0
        pygame.display.set_mode((int(W * max(sc, .5)), int(H * max(sc, .5))), pygame.RESIZABLE)
        screen = pygame.Surface((W, H))

    def to_v(p):
        ww, wh = pygame.display.get_surface().get_size()
        return int(p[0] * W / ww), int(p[1] * H / wh)
    pygame.display.set_caption("Signal Tower")
    def load_font(size, bold):
        """Bundled sans font (browsers have no system fonts); falls back to Pygame's default."""
        path = os.path.join(HERE, "assets", "sans-bold.ttf" if bold else "sans.ttf")
        try:
            return pygame.font.Font(path, size)
        except (FileNotFoundError, OSError):
            return pygame.font.Font(None, int(size * 1.3))
    F = {k: load_font(s, b) for k, s, b in
         (("s", 15, False), ("m", 17, True), ("b", 22, True), ("h", 30, True), ("t", 18, False),
          ("xs", 13, False))}
    clock = pygame.time.Clock()
    waves = pygame.Surface((MAP_W, MAP_H), pygame.SRCALPHA)

    st = dict(lv=0, towers=[], sel=None, col=0, size=1, flash="", until=0, modal=None, toast_rect=None, toast_btn=None,
              hint=None, found=set(), did=set(), toast=None, conf=[], fired=False, qa=None, stars={}, badges=set(),
              prev=0.0, prevc=0.0, floats=[], evt=None, passed=False, clear_until=0, hused=0)
    pop = map_surf = None

    def load(i, keep=False):
        nonlocal pop, map_surf
        if not keep or "seed" not in st:
            st["seed"] = int(np.random.randint(0, 10 ** 6))
        st.update(lv=i, towers=[], sel=None, col=0, size=1, modal=None, hint=None, toast=None, conf=[], fired=False,
                  qa=None, prev=0.0, prevc=0.0, floats=[], evt=None, passed=False, clear_until=0, hused=0)
        pop = generate_population(GRID_W, GRID_H, 6, np.random.default_rng(st["seed"]))
        map_surf = make_map(pop, st["seed"])

    def tone(freqs, ms=70):
        if not HAVE_MIXER:
            return None
        try:
            n = int(44100 * ms / 1000)
            t_ = np.arange(n) / 44100
            w = np.concatenate([np.sin(2 * np.pi * f * t_) * np.linspace(1, 0, n) for f in freqs])
            return pygame.mixer.Sound(buffer=(w * 9000).astype(np.int16).tobytes())
        except Exception:
            return None

    SPECS = dict(place=([620], 70), disc=([660, 880, 1100], 110), win=([523, 659, 784, 1047], 140),
                 bad=([160], 160), click=([440], 40))
    SND = {k: tone(f, ms) for k, (f, ms) in SPECS.items()}
    muted = [False]

    def sfx(k):
        if muted[0]:
            return
        try:
            if SND.get(k):
                SND[k].play()
            elif web_beep:
                f, ms = SPECS[k]
                web_beep(",".join(str(x) for x in f), ms)
        except Exception as ex:
            print("sound error:", ex)

    def flash(msg, snd=True):
        st["flash"], st["until"] = msg, pygame.time.get_ticks() + 2200
        if snd:
            sfx("bad")

    def spent():
        return sum(tcost(t) for t in st["towers"])

    def burst(x, y, n=40):
        pal = COLORS + [GOLD]
        for _ in range(n):
            a, v = np.random.uniform(0, 2 * math.pi), np.random.uniform(2, 7)
            st["conf"].append([x, y, math.cos(a) * v, math.sin(a) * v - 3, pal[np.random.randint(5)], pygame.time.get_ticks()])

    load(start)
    ev, dirty, btns, overlay, modal_btns = None, True, [], None, []

    running = True
    while running:
        lv = LEVELS[st["lv"]]
        towers, sel = st["towers"], st["sel"]
        mx, my = to_v(pygame.mouse.get_pos())
        cell = (mx // CELL, my // CELL) if mx < MAP_W and my < MAP_H else None

        for e in pygame.event.get():
            if e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE and not WEB):
                running = False
            elif e.type == pygame.KEYDOWN and e.key in (pygame.K_DELETE, pygame.K_BACKSPACE) and not st["modal"]:
                # same as right-click: remove the selected tower, else the one under the mouse
                tgt = sel or (pick(towers, cell, (mx, my)) if cell else None)
                if tgt:
                    site = (tgt.col, tgt.row)
                    towers[:] = [t for t in towers if (t.col, t.row) != site]
                    st["sel"] = None
                    dirty = True
            elif e.type == pygame.KEYDOWN and e.key == pygame.K_m:
                muted[0] = not muted[0]
                flash("Sound off" if muted[0] else "Sound on", False)
            elif e.type == pygame.MOUSEBUTTONDOWN and st["modal"]:
                if e.button == 1:
                    for rect, act in modal_btns:
                        if not rect.collidepoint(to_v(e.pos)):
                            continue
                        if act == "close":
                            st["modal"] = None
                        elif act == "cont":
                            if st["lv"] < len(LEVELS) - 1:
                                load(st["lv"] + 1); dirty = True
                            else:
                                st["modal"] = ("final",)
                        elif act == "restart":
                            st.update(stars={}, badges=set(), found=set(), did=set()); load(0); dirty = True
                        elif isinstance(act, tuple) and st["qa"] is None:
                            st["qa"] = act[1]
                            if act[1] == QUIZ[st["lv"]][2]:
                                st["badges"].add(st["lv"])
                        break
            elif e.type == pygame.MOUSEBUTTONDOWN and st["toast"] and st["toast_rect"] and st["toast_rect"].collidepoint(to_v(e.pos)):
                if e.button == 1 and st["toast_btn"].collidepoint(to_v(e.pos)):
                    st["toast"] = None
                    sfx("click")
            elif e.type == pygame.MOUSEBUTTONDOWN:
                hit = pick(towers, cell, (mx, my)) if cell else None
                if cell and e.button == 1:
                    if hit:
                        st["sel"] = None if hit is st["sel"] else hit
                    else:
                        p = SIZES[st["size"]][1]
                        if p == UMB_P and n_umb(towers) >= lv["umb"]:
                            flash("Only one umbrella cell fits in this city!")
                        elif spent() + price(p) > lv["budget"]:
                            flash("Not enough coins! Try a smaller tower.")
                        else:
                            t = Tower5(cell[0], cell[1], p, CHANNELS[st["col"]])
                            t.cidx = st["col"]
                            t.born = pygame.time.get_ticks()
                            burst(*t.center_px, 8)
                            sfx("place")
                            towers.append(t)
                            st["sel"] = None
                            dirty = True
                elif cell and e.button == 3 and hit:
                    site = (hit.col, hit.row)
                    towers[:] = [t for t in towers if (t.col, t.row) != site]
                    st["sel"] = None
                    dirty = True
                elif e.button == 1:
                    for rect, kind, i in btns:
                        if not rect.collidepoint(to_v(e.pos)):
                            continue
                        sfx("click")
                        if kind == "col":
                            if sel:
                                sel.cidx, sel.channel = i, CHANNELS[i]
                                dirty = True
                            else:
                                st["col"] = i
                        elif kind == "size":
                            if sel:
                                p = SIZES[i][1]
                                if sel.azimuth is not None:
                                    flash("Sector antennas can't be resized")
                                elif p == UMB_P and not is_umb(sel) and n_umb(towers) >= lv["umb"]:
                                    flash("Only one umbrella cell fits in this city!")
                                elif spent() - tcost(sel) + price(p) > lv["budget"]:
                                    flash("Not enough coins to upgrade!")
                                else:
                                    sel.power_dbm = p
                                    sel.update()
                                    dirty = True
                            else:
                                st["size"] = i
                        elif kind == "auto":
                            if towers:
                                auto_color(towers, lv["colors"])
                                flash("Auto-color: neighbours now use different colors", False)
                                dirty = True
                            else:
                                flash("Build a tower first!")
                        elif kind == "ideas":
                            st["modal"] = "glossary"
                        elif kind == "reset":
                            load(st["lv"], keep=True)
                            dirty = True
                        elif kind == "newmap":
                            load(st["lv"])
                            dirty = True
                        elif kind == "hint":
                            if st["hused"] >= lv["hints"]:
                                flash("No hints left on this level!")
                            else:
                                st["hused"] += 1
                                st["hint"] = (best_spot(pop, ev, SIZES[st["size"]][1], towers), pygame.time.get_ticks())
                        elif kind == "split":
                            kids = split_tower(sel, towers, lv["colors"]) if sel else None
                            if not kids:
                                flash("Can't split here (too small, sectored, or no room)")
                            elif spent() - tcost(sel) + sum(tcost(k) for k in kids) > lv["budget"]:
                                flash("Not enough coins to split!")
                            else:
                                towers.remove(sel)
                                towers.extend(kids)
                                st["sel"] = None
                                st["did"].add("split")
                                burst(*kids[0].center_px, 12)
                                sfx("place")
                                dirty = True
                        elif kind == "sect":
                            if not sel:
                                flash("Select a tower first")
                            elif sel.azimuth is not None or is_umb(sel):
                                flash("Only a normal tower can get sectors")
                            else:
                                kids = sectorize(sel, lv["colors"])
                                if spent() - tcost(sel) + sum(tcost(k) for k in kids) > lv["budget"]:
                                    flash("Not enough coins for 3 sectors!")
                                else:
                                    towers.remove(sel)
                                    towers.extend(kids)
                                    st["sel"] = None
                                    st["did"].add("sector")
                                    burst(*kids[0].center_px, 12)
                                    sfx("place")
                                    dirty = True
                        elif kind == "next":
                            st["modal"], st["qa"] = ("quiz", st["lv"]), None
                        break

        towers, sel, lv = st["towers"], st["sel"], LEVELS[st["lv"]]   # events may have reloaded the level

        # Concert night: once the target is reached, a crowd appears far from your towers
        if lv["event"] and st["evt"] is None and towers and ev and not st["modal"] and ev["happy"] >= lv["target"]:
            rng = np.random.default_rng()
            others = [t for t in towers if not is_umb(t)] or towers
            cands = [(int(rng.integers(4, GRID_W - 4)), int(rng.integers(4, GRID_H - 4))) for _ in range(80)]
            spot = max(cands, key=lambda c_: min(math.hypot(c_[0] - t.col, c_[1] - t.row) for t in others))
            pop = pop + 1.0 * np.exp(-((XS - spot[0]) ** 2 + (YS - spot[1]) ** 2) / (2 * 2.2 ** 2))
            map_surf = make_map(pop, st["seed"])
            st["evt"] = spot
            dirty = True
            flash("Concert tonight! A big crowd is heading to the star!")

        if dirty:
            ev = evaluate(towers, pop, lv)
            overlay = make_overlay(ev)
            dirty = False
            nh, nc = (ev["happy"], ev["clash"]) if ev else (0.0, 0.0)
            pos = (mx, my) if cell else (sel.center_px if sel else (MAP_W // 2, MAP_H // 2))
            now = pygame.time.get_ticks()
            if abs(nh - st["prev"]) >= 1:
                st["floats"].append([pos[0], pos[1], f"{nh - st['prev']:+.0f}% people", GREEN if nh > st["prev"] else RED, now])
            if nc - st["prevc"] >= 2:
                st["floats"].append([pos[0], pos[1] + 26, f"+{nc - st['prevc']:.0f}% static!", RED, now])
            st["prev"], st["prevc"] = nh, nc
        happy = ev["happy"] if ev else 0.0
        clash = ev["clash"] if ev else 0.0
        drop = ev["drop"] if ev else 0.0
        used = spent()
        star2 = min(lv["target"] + 10, 98)
        goals = [(f"Good signal for {lv['target']}% of people", happy >= lv["target"]),
                 (f"Reach {star2}% of people", happy >= star2),
                 (f"Win using under {int(lv['thrift'] * 100)}% of your coins",
                  happy >= lv["target"] and used <= lv["thrift"] * lv["budget"])]
        won = goals[0][1] and (not lv["event"] or st["evt"] is not None)
        if won:
            st["stars"][st["lv"]] = max(st["stars"].get(st["lv"], 0), sum(ok for _, ok in goals))
        tnow = pygame.time.get_ticks()
        if not st["modal"] and not st["toast"]:
            nts = [t for t in towers if not is_umb(t)]
            cols = [t.cidx for t in nts]
            over_now = ev is not None and bool((ev["load"] > lv["cap"]).any())
            checks = [
                ("coverage", happy > 0), ("cci", clash >= 3),
                ("frequency", lv["colors"] > 1 and len(set(cols)) >= 2),
                ("overlap", len(towers) >= 4 and ev is not None and ev["overlap"] > 35),
                ("reuse", lv["colors"] > 1 and "cci" in st["found"] and len(towers) >= 4 and clash < 2
                 and len(set(cols)) >= 2 and max(cols.count(c_) for c_ in set(cols)) >= 2),
                ("tradeoff", any(t.power_dbm == 38 for t in towers)),
                ("capacity", over_now),
                ("split", "split" in st["did"]), ("sector", "sector" in st["did"]),
                ("umbrella", n_umb(towers) > 0), ("surge", st["evt"] is not None)]
            for key, cond in checks:
                if cond and key not in st["found"]:
                    st["found"].add(key)
                    st["toast"] = (key, tnow)
                    burst(MAP_W // 2, 70, 30)
                    sfx("disc")
                    break
        cleared = won and (st["lv"] != 0 or "cci" in st["found"])
        if cleared and not st["fired"]:
            st["fired"] = st["passed"] = True
            st["clear_until"] = pygame.time.get_ticks() + 3200      # notification only, then it goes away
            burst(MAP_W // 2, 110, 70)
            sfx("win")
        t_s = pygame.time.get_ticks() / 1000

        # ------------------------------ map ------------------------------
        screen.fill(BG)
        screen.blit(map_surf, (0, 0))
        screen.blit(overlay, (0, 0))
        waves.fill((0, 0, 0, 0))
        for t in towers:
            R = int(t.radius_cells * CELL)
            for k in range(2):
                ph = (t_s * .35 + k * .5) % 1
                draw_reach(waves, t, max(int(R * ph), 3), (*tcol(t), int(170 * (1 - ph))), 3)
        if sel:
            draw_reach(waves, sel, int(sel.radius_cells * CELL), (*tcol(sel), 200), 2)
        if cell and not any((t.col, t.row) == cell for t in towers):
            cx, cy = int((cell[0] + .5) * CELL), int((cell[1] + .5) * CELL)
            pygame.draw.circle(waves, (255, 255, 255, 150), (cx, cy),
                               int(radius_for_power(SIZES[st["size"]][1]) * CELL), 2)
            pygame.draw.circle(waves, (255, 255, 255, 220), (cx, cy), 9, 2)
        screen.blit(waves, (0, 0))
        if st["evt"]:
            ex, ey = int((st["evt"][0] + .5) * CELL), int((st["evt"][1] + .5) * CELL)
            pygame.draw.circle(screen, GOLD, (ex, ey), int(30 + 5 * math.sin(t_s * 5)), 3)
            star(screen, ex, ey, 13, GOLD)
            lab = F["s"].render("Concert crowd!", True, INK)
            lr = lab.get_rect(midtop=(ex, ey + 34)).inflate(14, 6)
            pygame.draw.rect(screen, GOLD, lr, border_radius=10)
            screen.blit(lab, lab.get_rect(center=lr.center))
        for t in towers:
            c = mpos(t)
            age = pygame.time.get_ticks() - getattr(t, "born", -9999)
            k = 1 + .5 * math.sin(age / 350 * math.pi) if age < 350 else 1
            if is_umb(t):
                draw_umbrella(screen, c, k)
            else:
                if t.azimuth is not None:
                    pygame.draw.circle(screen, (255, 255, 255), t.center_px, 3)
                pygame.draw.circle(screen, (150, 150, 160), (c[0] + 1, c[1] + 3), int(14 * k))
                pygame.draw.circle(screen, (255, 255, 255), c, int(14 * k))
                pygame.draw.circle(screen, COLORS[t.cidx], c, int(11 * k))
                ix = ev["ix"].get(id(t)) if ev is not None else None
                if ix is not None and ev["load"][ix] > lv["cap"]:
                    ring = LAV if (ix in ev["relieved"] and lv["pen"]) else ORANGE
                    pygame.draw.circle(screen, ring, c, int(19 + 2 * math.sin(t_s * 8)), 3)
                pygame.draw.polygon(screen, (255, 255, 255), [(c[0] - 4, c[1] + 6), (c[0], c[1] - 6), (c[0] + 4, c[1] + 6)])
                pygame.draw.circle(screen, (255, 255, 255), (c[0], c[1] - 7), 2)
            if t is sel:
                pygame.draw.circle(screen, INK, c, int(20 + 2 * math.sin(t_s * 6)), 2)
        if st["hint"]:
            (hc, hr), t0 = st["hint"]
            if tnow - t0 > 4500:
                st["hint"] = None
            else:
                cx, cy = int((hc + .5) * CELL), int((hr + .5) * CELL)
                rr = int(16 + 5 * math.sin(tnow / 150))
                pygame.draw.circle(screen, GOLD, (cx, cy), rr, 4)
                lab = F["s"].render("Try here!", True, INK)
                lr = lab.get_rect(midbottom=(cx, cy - rr - 4)).inflate(14, 6)
                pygame.draw.rect(screen, GOLD, lr, border_radius=10)
                screen.blit(lab, lab.get_rect(center=lr.center))
        now = pygame.time.get_ticks()
        st["conf"] = [c_ for c_ in st["conf"] if c_[1] < MAP_H and now - c_[5] < 2600]
        for c_ in st["conf"]:
            c_[0] += c_[2]; c_[1] += c_[3]; c_[3] += .25
            pygame.draw.rect(screen, c_[4], (int(c_[0]), int(c_[1]), 7, 5))
        if st["toast"]:
            key, t0 = st["toast"]
            age = now - t0
            ttl, txt = CONCEPTS[key]
            lines = wrap(F["s"], txt, 350)
            hh = 110 + 18 * len(lines)
            y0 = int(-hh + (hh + 52) * (1 - (1 - min(1, age / 350)) ** 3))
            rc = pygame.Rect(MAP_W // 2 - 220, y0, 440, hh)
            pygame.draw.rect(screen, (255, 246, 214), rc, border_radius=18)
            pygame.draw.rect(screen, GOLD, rc, 4, border_radius=18)
            star(screen, rc.x + 34, rc.y + 36, 20, GOLD)
            screen.blit(F["s"].render("NEW DISCOVERY!", True, (200, 120, 0)), (rc.x + 66, rc.y + 8))
            screen.blit(F["b"].render(ttl, True, INK), (rc.x + 66, rc.y + 28))
            for i, ln in enumerate(lines):
                screen.blit(F["s"].render(ln, True, INK), (rc.x + 66, rc.y + 58 + 18 * i))
            gb = pygame.Rect(rc.right - 112, rc.bottom - 46, 96, 32)
            pygame.draw.rect(screen, (20, 140, 90), gb.move(0, 3), border_radius=12)
            pygame.draw.rect(screen, GREEN, gb, border_radius=12)
            gt = F["m"].render("Got it!", True, (255, 255, 255))
            screen.blit(gt, gt.get_rect(center=gb.center))
            st["toast_rect"], st["toast_btn"] = rc, gb
        st["floats"] = [f for f in st["floats"] if now - f[4] < 1500]
        for fx, fy, txt, col, t0 in st["floats"]:
            k = (now - t0) / 1500
            for off, cc in (((1, 1), (255, 255, 255)), ((0, 0), col)):
                t_ = F["b"].render(txt, True, cc)
                t_.set_alpha(int(255 * (1 - k)))
                screen.blit(t_, (fx - t_.get_width() // 2 + off[0], fy - 30 - int(45 * k) + off[1]))
        tot, nb = sum(st["stars"].values()), len(st["badges"])
        pygame.draw.rect(screen, (255, 255, 255), (10, 8, 285, 30), border_radius=15)
        hx = 22
        for txt in (f"Level {st['lv'] + 1}/{len(LEVELS)}", None, f"Badges {nb}/{len(LEVELS)}"):
            if txt is None:
                star(screen, hx + 8, 24, 9, GOLD)
                txt, hx = f"{tot}/{3 * len(LEVELS)}", hx + 20
            t_ = F["s"].render(txt, True, INK)
            screen.blit(t_, (hx, 15)); hx += t_.get_width() + 16
        pygame.draw.rect(screen, (255, 255, 255), (MAP_W - 214, 8, 204, 30), border_radius=15)
        draw_house(screen, MAP_W - 210, 12, HOUSES[0], ROOFS[0])
        screen.blit(F["s"].render("few", True, INK), (MAP_W - 186, 15))
        draw_block(screen, MAP_W - 154, 12)
        screen.blit(F["s"].render("many people", True, INK), (MAP_W - 130, 15))

        # "Level cleared" is only a short notification now; it fades out by itself
        left_ms = st["clear_until"] - pygame.time.get_ticks()
        if left_ms > 0:
            s = F["b"].render("Level cleared!  Next level unlocked", True, (255, 255, 255))
            box = pygame.Rect(0, 0, s.get_width() + 36, s.get_height() + 18)
            note = pygame.Surface(box.size, pygame.SRCALPHA)
            pygame.draw.rect(note, (*GREEN, 255), note.get_rect(), border_radius=20)
            note.blit(s, (18, 9))
            note.set_alpha(int(255 * min(1, left_ms / 600)))
            screen.blit(note, note.get_rect(center=(MAP_W // 2, MAP_H - 84)))

        if cell and not st["modal"]:
            r_, c_ = cell[1], cell[0]
            if ev is None or not ev["covered"][r_, c_]:
                msg = "No signal here"
            elif ev["bad"][r_, c_]:
                msg = "Static! Two same-color towers are about equally strong here"
            elif ev["cov_n"][r_, c_]:
                msg = f"Good signal from a {NAMES[ev['nt'][ev['serving'][r_, c_]].cidx]} tower"
            else:
                msg = "Only the umbrella cell reaches here"
            ts = F["s"].render(msg, True, (255, 255, 255))
            tr = pygame.Rect(0, 0, ts.get_width() + 16, ts.get_height() + 10)
            tr.topleft = (min(mx + 16, MAP_W - tr.w - 4), min(my + 16, MAP_H - tr.h - 4))
            pygame.draw.rect(screen, INK, tr, border_radius=8)
            screen.blit(ts, ts.get_rect(center=tr.center))

        # ----------------------------- coach -----------------------------
        pygame.draw.rect(screen, (255, 246, 216), (0, MAP_H, MAP_W, TIP_H))
        ax, ay = 50, MAP_H + TIP_H // 2
        pygame.draw.circle(screen, GOLD, (ax, ay), 28)
        pygame.draw.circle(screen, INK, (ax - 9, ay - 6), 3)
        pygame.draw.circle(screen, INK, (ax + 9, ay - 6), 3)
        pygame.draw.arc(screen, INK, (ax - 12, ay - 8, 24, 20), math.pi * 1.15, math.pi * 1.85, 3)
        term = None
        over_now = ev is not None and bool((ev["load"] > lv["cap"]).any())
        if not towers:
            tip = "Click anywhere on the map to build your first tower. Tall blocks mean lots of people. Right-click (or Delete) removes a tower."
        elif clash >= 3:
            term = "cci"
            tip = "Towers on the same color shout over each other. Give neighbours different colors, and reuse a color only far away!"
        elif st["evt"] is not None and not won:
            tip = "The concert crowd needs signal AND capacity! An umbrella cell or a couple of small towers near the star can help."
        elif st["lv"] == 0 and "cci" not in st["found"]:
            tip = "Experiment! Build a second tower RIGHT NEXT to the first one and watch the map."
        elif st["lv"] == 1 and "frequency" not in st["found"]:
            tip = "Try it: click a tower, then pick a different color for it. Watch what changes."
        elif st["lv"] == 2 and "tradeoff" not in st["found"]:
            tip = "Try a Large tower somewhere. What do you notice about the price?"
        elif lv["sect"] and "sector" not in st["found"] and over_now:
            tip = "Crowded tower? Click it and press Sectors: 3 antennas, 3 colors, and much less static."
        elif lv["umb"] and "umbrella" not in st["found"]:
            tip = "Try the Umbrella size: one giant tower that fills gaps and catches overflow. The city only has room for one."
        elif over_now and "capacity" in st["found"]:
            term = "capacity"
            tip = ("Orange ring = a crowded tower that DROPS people! Split it, add Sectors, or put an umbrella over it."
                   if lv["pen"] else "Orange ring = a crowded tower! Click it and press Split to share the load.")
        elif ev and len(towers) >= 4 and ev["overlap"] > 35:
            term = "overlap"
            tip = "Many towers now cover the same people. You are paying for coverage you already had."
        elif st["passed"]:
            tip = "Level passed! Press Next level when you are ready, or keep improving for more stars."
        elif happy < lv["target"]:
            tip = "Foggy areas still have no signal. Build near the houses, or make a tower bigger to reach further."
        else:
            tip = "Nice work!"
        if term:
            pill = F["s"].render("Idea: " + CONCEPTS[term][0], True, (255, 255, 255))
            pr = pill.get_rect(topleft=(96, MAP_H + 10)).inflate(20, 8)
            pygame.draw.rect(screen, (99, 102, 241), pr, border_radius=12)
            screen.blit(pill, pill.get_rect(center=pr.center))
        for i, ln in enumerate(wrap(F["t"], tip, MAP_W - 130)[:2 if term else 3]):
            screen.blit(F["t"].render(ln, True, INK), (96, MAP_H + (42 if term else 18) + 24 * i))
        if pygame.time.get_ticks() < st["until"]:
            s = F["b"].render(st["flash"], True, (255, 255, 255))
            box = s.get_rect(center=(MAP_W // 2, MAP_H - 36)).inflate(36, 18)
            pygame.draw.rect(screen, RED, box, border_radius=20)
            screen.blit(s, s.get_rect(center=box.center))

        # ------------------------------ panel ----------------------------
        pygame.draw.rect(screen, PBG, (MAP_W, 0, PANEL_W, H))
        x, cw = MAP_W + 14, PANEL_W - 28
        btns = []
        screen.blit(F["h"].render("Signal Tower", True, TXT), (x, 8))
        screen.blit(F["m"].render(f"Level {st['lv'] + 1} - {lv['name']}", True, (150, 150, 255)), (x, 44))
        y = 70
        for ln in wrap(F["s"], lv["story"], cw):
            screen.blit(F["s"].render(ln, True, TXT2), (x, y)); y += 19
        y += 6

        def card(y0, h):
            pygame.draw.rect(screen, CARD, (x, y0, cw, h), border_radius=16)

        def meter(y0, label, val, color, text, target=None, extra=None):
            screen.blit(F["m"].render(label, True, TXT), (x + 14, y0))
            v = F["m"].render(text, True, color)
            screen.blit(v, (x + cw - 14 - v.get_width(), y0))
            bw_ = cw - 28
            pygame.draw.rect(screen, (60, 68, 100), (x + 14, y0 + 26, bw_, 12), border_radius=6)
            acc = 0.0
            for v_, c_ in [(val, color)] + ([extra] if extra else []):
                if v_ > 0 and acc < 100:
                    pygame.draw.rect(screen, c_, (x + 14 + int(bw_ * acc / 100), y0 + 26,
                                                  max(int(bw_ * min(v_, 100 - acc) / 100), 8), 12), border_radius=6)
                acc += v_
            if target:
                tx = x + 14 + int(bw_ * target / 100)
                pygame.draw.line(screen, TXT, (tx, y0 + 22), (tx, y0 + 42), 3)

        card(y, 14 + 42 * 3)
        yy = y + 12
        meter(yy, "People with signal", happy, GREEN, f"{happy:.0f}%", lv["target"]); yy += 42
        if lv["pen"]:
            meter(yy, "Static | Overload", clash, RED, f"{clash:.0f}% | {drop:.0f}%", extra=(drop, ORANGE))
        else:
            meter(yy, "Red static", clash, RED, f"{clash:.0f}%")
        yy += 42
        left = lv["budget"] - used
        meter(yy, "Coins left", 100 * left / lv["budget"], GOLD, str(left))
        y += 14 + 42 * 3 + 8

        card(y, 106)
        for i in range(3):
            star(screen, x + 34 + i * 46, y + 28, 19, GOLD if goals[i][1] else (70, 78, 110))
        for i, (txt, ok) in enumerate(goals):
            gy = y + 50 + i * 18
            pygame.draw.circle(screen, GREEN if ok else (70, 78, 110), (x + 22, gy + 8), 7)
            if ok:
                pygame.draw.lines(screen, (255, 255, 255), False, [(x + 18, gy + 8), (x + 21, gy + 11), (x + 27, gy + 5)], 2)
            screen.blit(F["s"].render(txt, True, TXT if ok else TXT2), (x + 36, gy))
        y += 114

        def button(rect, kind, i=0, bg=(58, 68, 108), ring=False, on=True):
            r = pygame.Rect(rect)
            base = bg if on else (44, 52, 84)
            pygame.draw.rect(screen, tuple(int(c * .55) for c in base), r.move(0, 4), border_radius=12)
            pygame.draw.rect(screen, base, r, border_radius=12)
            if ring:
                pygame.draw.rect(screen, TXT, r, 3, border_radius=12)
            if on:
                btns.append((r, kind, i))
            return r

        r = button((x + cw - 84, 14, 84, 26), "ideas", bg=(224, 231, 255))
        s_ = F["s"].render(f"Ideas {len(st['found'])}/{len(CONCEPTS)}", True, (67, 56, 202))
        screen.blit(s_, s_.get_rect(center=r.center))
        cur_col = sel.cidx if sel else st["col"]
        if sel:
            cur_sz = 3 if is_umb(sel) else next((i for i, s2 in enumerate(SIZES[:3]) if s2[1] == sel.power_dbm), 1)
        else:
            cur_sz = st["size"]
        ch = 206 + (40 if lv["umb"] else 0)
        card(y, ch)
        yy = y + 12
        nc_ = lv["colors"]
        screen.blit(F["m"].render("Color: SELECTED tower" if sel else "Color: next tower", True, TXT), (x + 14, yy)); yy += 24
        bw = (cw - 28 - 6 * (nc_ - 1)) // nc_
        for i in range(nc_):
            r = button((x + 14 + i * (bw + 6), yy, bw, 32), "col", i, COLORS[i], cur_col == i)
            s = F["s"].render(NAMES[i], True, (30, 36, 56))
            screen.blit(s, s.get_rect(center=r.center))
        yy += 42
        screen.blit(F["m"].render("Size: SELECTED tower" if sel else "Size: next tower", True, TXT), (x + 14, yy)); yy += 24
        bw = (cw - 28 - 12) // 3
        for i, (nm, p) in enumerate(SIZES[:3]):
            r = button((x + 14 + i * (bw + 6), yy, bw, 46), "size", i, ring=cur_sz == i)
            a = F["s"].render(nm, True, TXT)
            b = F["s"].render(f"{price(p)} coins", True, TXT2)
            screen.blit(a, a.get_rect(center=(r.centerx, r.y + 15)))
            screen.blit(b, b.get_rect(center=(r.centerx, r.y + 33)))
        yy += 58
        if lv["umb"]:
            r = button((x + 14, yy, cw - 28, 32), "size", 3, (84, 72, 160), cur_sz == 3)
            s = F["s"].render(f"Umbrella cell  {price(UMB_P)} coins", True, (255, 255, 255))
            screen.blit(s, s.get_rect(center=r.center))
            yy += 40
        r = button((x + 14, yy, cw - 28, 34), "auto", bg=(224, 231, 255))
        s = F["m"].render("Auto-color (help me!)", True, (67, 56, 202))
        screen.blit(s, s.get_rect(center=r.center))
        y += ch + 8

        r1 = button((x, y, cw // 2 - 4, 40), "reset")
        s = F["m"].render("Start over", True, TXT)
        screen.blit(s, s.get_rect(center=r1.center))
        can = st["passed"]
        r2 = button((x + cw // 2 + 4, y, cw // 2 - 4, 40), "next", bg=GREEN, on=can)
        if can:
            pygame.draw.rect(screen, (255, 255, 255), r2.inflate(int(6 + 4 * math.sin(t_s * 6)), int(6 + 4 * math.sin(t_s * 6))),
                             3, border_radius=14)
        s = F["m"].render("Next level" if st["lv"] < len(LEVELS) - 1 else "Finish", True, (255, 255, 255) if can else TXT2)
        screen.blit(s, s.get_rect(center=r2.center))
        y += 46
        hints_left = lv["hints"] - st["hused"]
        row = (("Hint " + str(hints_left), "hint", hints_left > 0),
               ("Split", "split", bool(sel) and sel.azimuth is None and not is_umb(sel)),
               ("Sectors", "sect", lv["sect"] and bool(sel) and sel.azimuth is None and not is_umb(sel)),
               ("New map", "newmap", True))
        b4 = (cw - 12) // 4
        for i, (lab, kind, on_) in enumerate(row):
            r3 = button((x + i * (b4 + 4), y, b4, 28), kind, bg=(58, 68, 108), on=on_)
            fnt = F["s"] if F["s"].size(lab)[0] <= b4 - 6 else F["xs"]      # keep wide labels inside the button
            s = fnt.render(lab, True, TXT if on_ else TXT2)
            screen.blit(s, s.get_rect(center=r3.center))

        modal_btns = []
        if st["modal"]:
            m = st["modal"]
            mk = m if isinstance(m, str) else m[0]
            ov = pygame.Surface((W, H), pygame.SRCALPHA)
            ov.fill((20, 24, 40, 175))
            screen.blit(ov, (0, 0))

            def mbutton(rect, label, act, bg=GREEN, fg=(255, 255, 255)):
                pygame.draw.rect(screen, bg, rect, border_radius=14)
                t_ = F["m"].render(label, True, fg)
                screen.blit(t_, t_.get_rect(center=rect.center))
                if act:
                    modal_btns.append((rect, act))

            def mcard(w_, h_, title, col=(99, 102, 241)):
                mc = pygame.Rect(0, 0, w_, h_)
                mc.center = (W // 2, H // 2)
                pygame.draw.rect(screen, (255, 255, 255), mc, border_radius=22)
                screen.blit(F["h"].render(title, True, col), (mc.x + 30, mc.y + 18))
                return mc

            if mk in ("lesson", "glossary"):
                keys = list(CONCEPTS) if mk == "glossary" else LESSONS[m[1]]
                ncol = 3 if len(keys) > 5 else 1
                colw = 340 if ncol == 3 else 540
                blocks = [((CONCEPTS[k][0] if k in st['found'] else '??? Locked'),
                           wrap(F['s'], CONCEPTS[k][1] if k in st['found'] else 'Keep playing to discover this idea!', colw - 30))
                          for k in keys]
                per = -(-len(blocks) // ncol)
                columns = [blocks[i * per:(i + 1) * per] for i in range(ncol)]
                hmax = max(sum(30 + 19 * len(b_[1]) + 14 for b_ in c_) for c_ in columns)
                mc = mcard(ncol * colw + 50, 140 + hmax, f"Ideas book  {len(st['found'])}/{len(CONCEPTS)}")
                for ci, col_blocks in enumerate(columns):
                    yy = mc.y + 70
                    for title, lines in col_blocks:
                        screen.blit(F["m"].render(title, True, INK), (mc.x + 30 + ci * colw, yy)); yy += 28
                        for ln in lines:
                            screen.blit(F["s"].render(ln, True, SOFT), (mc.x + 30 + ci * colw, yy)); yy += 19
                        yy += 14
                mbutton(pygame.Rect(mc.centerx - 80, mc.bottom - 58, 160, 42), "Got it!", "close")
            elif mk == "quiz":
                q = QUIZ[m[1]]
                mc = mcard(640, 480, "Quick quiz!")
                yy = mc.y + 66
                for ln in wrap(F["b"], q[0], 580):
                    screen.blit(F["b"].render(ln, True, INK), (mc.x + 30, yy)); yy += 28
                yy += 8
                for k, opt in enumerate(q[1]):
                    rc = pygame.Rect(mc.x + 30, yy, 580, 40)
                    if st["qa"] is None:
                        mbutton(rc, opt, ("ans", k), (238, 239, 244), INK)
                    else:
                        bg = GREEN if k == q[2] else RED if k == st["qa"] else (238, 239, 244)
                        mbutton(rc, opt, None, bg, SOFT if bg == (238, 239, 244) else (255, 255, 255))
                    yy += 44
                if st["qa"] is not None:
                    ok = st["qa"] == q[2]
                    screen.blit(F["b"].render("Correct! Badge earned" if ok else "Not quite...", True, GREEN if ok else RED), (mc.x + 30, yy + 4))
                    for i, ln in enumerate(wrap(F["t"], q[3], 580)[:3]):
                        screen.blit(F["t"].render(ln, True, INK), (mc.x + 30, yy + 34 + 22 * i))
                    mbutton(pygame.Rect(mc.centerx - 90, mc.bottom - 58, 180, 42),
                            "Continue" if st["lv"] < len(LEVELS) - 1 else "See results", "cont")
            else:
                tot, nb = sum(st["stars"].values()), len(st["badges"])
                score = tot + 2 * nb
                rank = ("Network Architect" if score >= 22 else "Network Engineer" if score >= 14
                        else "Tower Technician" if score >= 8 else "Apprentice Planner")
                mc = mcard(560, 380, "Mission complete!", GREEN)
                screen.blit(F["b"].render("Your rank:", True, SOFT), (mc.x + 30, mc.y + 76))
                screen.blit(F["h"].render(rank, True, GOLD), (mc.x + 30, mc.y + 102))
                for i in range(3 * len(LEVELS)):
                    star(screen, mc.x + 42 + i * 27, mc.y + 190, 11, GOLD if i < tot else (226, 227, 232))
                screen.blit(F["t"].render(f"Stars {tot}/{3 * len(LEVELS)}    Badges {nb}/{len(LEVELS)}", True, INK), (mc.x + 30, mc.y + 226))
                screen.blit(F["s"].render("Replay levels to collect every star and badge!", True, SOFT), (mc.x + 30, mc.y + 256))
                mbutton(pygame.Rect(mc.centerx - 90, mc.bottom - 60, 180, 42), "Play again", "restart")

        if not WEB:
            win = pygame.display.get_surface()
            if win.get_size() == (W, H):
                win.blit(screen, (0, 0))
            else:
                pygame.transform.smoothscale(screen, win.get_size(), win)
        pygame.display.flip()
        clock.tick(60)
        await asyncio.sleep(0)      # REQUIRED for Pygbag: lets the browser run between frames
    pygame.quit()


if __name__ == "__main__":
    asyncio.run(main())