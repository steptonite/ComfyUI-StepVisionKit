"""Depth Look · «Рух по глибині»: сім покадрових ефектів, що ганяють рух по карті глибини на ВЕСЬ кадр (з предметом),
а не лише по градієнту далекого плану. Пост-прохід після look: apply(frame, depth, t, p) -> frame.

frame (H,W,3) float32 0..1; depth (H,W) float 0..1 (яскраве = близько, вже нормоване як для look); t - секунди кадру;
p - dict параметрів `dm_*` (відсутні ключі = дефолти з SPEC). Сила ефекту 0 = вимкнено й не рахується; якщо вимкнено ВСЕ -
повертається той самий масив (побітово). Без стану між кадрами: результат залежить лише від (frame, depth, t, p) -
тож паралельний рендер по процесах дає той самий результат, що й послідовний. Лише numpy + cv2."""
import math

import numpy as np
import cv2

# (ключ, тип, дефолт, lo, hi, крок, англ. підпис). тип: "f" float · "i" int · "c" hex-колір · список = вибір зі списку
SHAPES = ["circle", "figure8"]
REVEAL_MODES = ["appear_near", "appear_far", "loop_near", "loop_far"]
SPEC = [
    ("dm_scan", "f", 0.0, 0.0, 1.0, 0.01, "Depth scanner"),
    ("dm_scan_speed", "f", 0.25, 0.02, 3.0, 0.01, "Scanner speed, sweeps/s"),
    ("dm_scan_width", "f", 0.10, 0.01, 0.6, 0.01, "Scanner width"),
    ("dm_scan_color", "c", "#7fd4ff", None, None, None, "Scanner colour"),
    ("dm_par", "f", 0.0, 0.0, 1.0, 0.01, "2.5D camera (parallax)"),
    ("dm_par_speed", "f", 0.30, 0.02, 3.0, 0.01, "Parallax speed, loops/s"),
    ("dm_par_focus", "f", 0.50, 0.0, 1.0, 0.01, "Parallax: still plane"),
    ("dm_par_shape", SHAPES, "circle", None, None, None, "Parallax path"),
    ("dm_focus", "f", 0.0, 0.0, 1.0, 0.01, "Focus pull"),
    ("dm_focus_speed", "f", 0.15, 0.02, 3.0, 0.01, "Focus pull speed, sweeps/s"),
    ("dm_focus_range", "f", 0.15, 0.02, 0.8, 0.01, "Sharp zone width"),
    ("dm_reveal", "f", 0.0, 0.0, 1.0, 0.01, "Reveal / dissolve"),
    ("dm_reveal_speed", "f", 0.20, 0.02, 3.0, 0.01, "Reveal speed, cycles/s"),
    ("dm_reveal_soft", "f", 0.15, 0.02, 0.6, 0.01, "Reveal edge softness"),
    ("dm_reveal_mode", REVEAL_MODES, "loop_near", None, None, None, "Reveal mode"),
    ("dm_iso", "f", 0.0, 0.0, 1.0, 0.01, "Depth contour lines"),
    ("dm_iso_count", "i", 14, 2, 80, 1, "Contour lines count"),
    ("dm_iso_speed", "f", 0.30, -3.0, 3.0, 0.01, "Contour run speed, lines/s"),
    ("dm_iso_width", "f", 1.5, 0.5, 12.0, 0.1, "Contour thickness, px"),
    ("dm_iso_color", "c", "#ffffff", None, None, None, "Contour colour"),
    ("dm_echo", "f", 0.0, 0.0, 1.0, 0.01, "Echo copies"),
    ("dm_echo_count", "i", 3, 1, 8, 1, "Echo copies count"),
    ("dm_echo_speed", "f", 0.40, 0.02, 3.0, 0.01, "Echo wave speed, waves/s"),
    ("dm_echo_scale", "f", 0.60, 0.05, 1.5, 0.01, "Echo growth"),
    ("dm_echo_thresh", "f", 0.60, 0.1, 0.95, 0.01, "Echo: near threshold"),
    ("dm_light", "f", 0.0, 0.0, 1.0, 0.01, "Relief light"),
    ("dm_light_speed", "f", 0.15, -2.0, 2.0, 0.01, "Light rotation, turns/s"),
    ("dm_light_height", "f", 0.80, 0.1, 3.0, 0.01, "Light height"),
    ("dm_light_relief", "f", 3.0, 0.5, 20.0, 0.1, "Relief depth"),
    ("dm_light_spec", "f", 0.40, 0.0, 1.0, 0.01, "Highlight"),
]
DEFAULTS = {s[0]: s[2] for s in SPEC}
STRENGTH_KEYS = ("dm_scan", "dm_par", "dm_focus", "dm_reveal", "dm_iso", "dm_echo", "dm_light")
KEYS = tuple(s[0] for s in SPEC)


# ---------------------------------------------------------------- хелпери
def _get(p, k):
    v = (p or {}).get(k, DEFAULTS[k])
    return DEFAULTS[k] if v is None else v


_ROW = {s[0]: s for s in SPEC}


def _f(p, k):
    s = _ROW[k]
    try:
        v = float(_get(p, k))
    except (TypeError, ValueError):
        v = float(s[2])
    if not math.isfinite(v):
        v = float(s[2])
    return min(max(v, s[3]), s[4])


def _hex(p, k):
    s = str(_get(p, k)).strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    try:
        v = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        s = str(DEFAULTS[k]).lstrip("#")
        v = [int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    return np.array(v, np.float32)


def active(p):
    """True, якщо хоч один ефект має силу > 0."""
    return any(_f(p, k) > 0.0 for k in STRENGTH_KEYS)


def _tri(x):
    """Трикутна хвиля 0..1..0 з періодом 1 (туди-назад)."""
    x = x - math.floor(x)
    return 1.0 - abs(2.0 * x - 1.0)


def _smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _blur(img, sigma):
    """Гаусс зі стелею вартості: великі сигми - на зменшеній копії."""
    if sigma < 0.3:
        return img
    if sigma <= 4.0:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    f = 3.0 / sigma
    h, w = img.shape[:2]
    sm = cv2.resize(img, (max(2, int(round(w * f))), max(2, int(round(h * f)))), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 3.0)
    return cv2.resize(sm, (w, h), interpolation=cv2.INTER_LINEAR)


def _side(frame):
    return float(max(frame.shape[0], frame.shape[1]))


# ---------------------------------------------------------------- 2. паралакс (міняє і кадр, і глибину)
def _parallax(frame, depth, t, p):
    s = _f(p, "dm_par")
    H, W = depth.shape
    ang = 2.0 * math.pi * _f(p, "dm_par_speed") * t
    if str(_get(p, "dm_par_shape")) == "figure8":
        cx, cy = math.sin(ang), 0.5 * math.sin(2.0 * ang)
    else:
        cx, cy = math.cos(ang), math.sin(ang)
    amp = s * 0.04 * _side(frame)                                   # макс. зсув у px при s=1
    focus = _f(p, "dm_par_focus")
    ds = _blur(depth, max(1.0, 0.004 * _side(frame)))               # гладка глибина - без розривів на краях предмета
    k = (ds - focus) * amp
    zoom = 1.0 + 0.03 * s + 0.01 * s                                # запас, щоб краї не вилазили
    xs = np.arange(W, dtype=np.float32)[None, :]
    ys = np.arange(H, dtype=np.float32)[:, None]
    ox, oy = (W - 1) * 0.5, (H - 1) * 0.5
    mx = (xs - ox) / zoom + ox - k * cx
    my = (ys - oy) / zoom + oy - k * cy
    mx = np.ascontiguousarray(np.broadcast_to(mx, (H, W)), np.float32)
    my = np.ascontiguousarray(np.broadcast_to(my, (H, W)), np.float32)
    fr = cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    dp = cv2.remap(depth, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return fr, dp


# ---------------------------------------------------------------- 3. перевід фокуса
def _focus(frame, depth, t, p):
    s = _f(p, "dm_focus")
    fp = _tri(_f(p, "dm_focus_speed") * t)
    rng = _f(p, "dm_focus_range")
    sig_max = s * 0.012 * _side(frame)
    if sig_max < 0.4:
        return frame
    off = np.clip((np.abs(depth - fp) - 0.5 * rng) / 0.35, 0.0, 1.0)   # 0 різко .. 1 макс. розмиття
    N = 3
    levels = [frame] + [_blur(frame, sig_max * (k + 1) / N) for k in range(N)]
    pos = off * N
    out = np.zeros_like(frame)
    for k, lv in enumerate(levels):
        out += lv * np.clip(1.0 - np.abs(pos - k), 0.0, 1.0)[..., None]
    return out


# ---------------------------------------------------------------- 1. сканер
def _scanner(frame, depth, t, p):
    s = _f(p, "dm_scan")
    ph = 1.0 - _tri(_f(p, "dm_scan_speed") * t)                     # 1 (ближній) -> 0 (далекий) -> назад
    w = _f(p, "dm_scan_width")
    dist = np.abs(depth - ph)
    band = 1.0 - _smooth((dist - 0.25 * w) / max(0.75 * w, 1e-4))   # плато в центрі, м'які краї
    col = _hex(p, "dm_scan_color")
    return frame + (s * band)[..., None] * col[None, None, :]


# ---------------------------------------------------------------- 4. поява / розчинення
def _reveal(frame, depth, t, p):
    s = _f(p, "dm_reveal")
    soft = _f(p, "dm_reveal_soft")
    mode = str(_get(p, "dm_reveal_mode"))
    c = _f(p, "dm_reveal_speed") * t
    u = (1.0 - depth) if mode.endswith("near") else depth          # 0 = ця сторона з'являється першою
    dissolve = False
    if mode.startswith("appear"):
        prog = min(max(c, 0.0), 1.0)
    else:
        cyc = c - math.floor(c)                                    # appear 0..0.4 · hold .4..0.6 · dissolve .6..1
        if cyc < 0.4:
            prog = cyc / 0.4
        elif cyc < 0.6:
            prog = 1.0
        else:
            prog, dissolve = (cyc - 0.6) / 0.4, True
    front = prog * (1.0 + soft)
    m = _smooth((front - np.clip(u, 0.0, 1.0)) / soft)
    if dissolve:
        m = 1.0 - m
    m = 1.0 - s * (1.0 - m)                                        # s<1: «до появи» лишається частка кадру
    return frame * m[..., None]


# ---------------------------------------------------------------- 5. ізолінії
def _iso(frame, depth, t, p):
    s = _f(p, "dm_iso")
    n = int(round(_f(p, "dm_iso_count")))
    n = max(2, n)
    wpx = _f(p, "dm_iso_width")
    ds = _blur(depth, max(1.0, 0.003 * _side(frame)))
    v = ds * n - _f(p, "dm_iso_speed") * t
    gx = cv2.Sobel(v, cv2.CV_32F, 1, 0, ksize=3) * 0.125
    gy = cv2.Sobel(v, cv2.CV_32F, 0, 1, ksize=3) * 0.125
    g = np.maximum(np.sqrt(gx * gx + gy * gy), 0.003)              # зміна «лінійного» v на піксель
    f = v - np.floor(v)
    dpx = np.minimum(f, 1.0 - f) / g                               # відстань до лінії в px
    a = np.clip(0.5 * wpx - dpx + 0.5, 0.0, 1.0) * s               # антиаліас: товщина wpx пікселів
    col = _hex(p, "dm_iso_color")
    return frame * (1.0 - a[..., None]) + col[None, None, :] * a[..., None]


# ---------------------------------------------------------------- 6. ехо-копії
def _echo(frame, depth, t, p):
    s = _f(p, "dm_echo")
    n = int(round(_f(p, "dm_echo_count")))
    th = _f(p, "dm_echo_thresh")
    mask = _smooth((depth - (th - 0.04)) / 0.08).astype(np.float32)
    mask = _blur(mask, max(1.0, 0.002 * _side(frame)))
    tot = float(mask.sum())
    if tot < 1e-4 * mask.size:
        return frame
    H, W = mask.shape
    ys, xs = np.mgrid[0:H, 0:W]
    cx = float((mask * xs).sum() / tot)
    cy = float((mask * ys).sum() / tot)
    grow = _f(p, "dm_echo_scale")
    spd = _f(p, "dm_echo_speed")
    cop = []
    for k in range(n):
        ph = (spd * t + k / float(n)) % 1.0
        cop.append((ph, 1.0 + grow * ph, s * 0.9 * (1.0 - ph) ** 0.8))
    cop.sort(key=lambda c: -c[0])                                  # найдальші (найбільші) - першими, під усім
    out = frame.copy()
    under = (1.0 - mask)[..., None]
    for ph, sc, al in cop:
        M = np.array([[sc, 0, cx - sc * cx], [0, sc, cy - sc * cy]], np.float32)
        fw = cv2.warpAffine(frame, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        mw = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        a = (al * mw)[..., None] * under
        out = out * (1.0 - a) + fw * a
    return out


# ---------------------------------------------------------------- 7. світло по рельєфу
def _light(frame, depth, t, p):
    s = _f(p, "dm_light")
    ds = _blur(depth, max(1.0, 0.004 * _side(frame)))
    hpx = _f(p, "dm_light_relief") * 0.02 * _side(frame)           # «висота» рельєфу в px при depth=1
    gx = cv2.Sobel(ds, cv2.CV_32F, 1, 0, ksize=3) * 0.125 * hpx
    gy = cv2.Sobel(ds, cv2.CV_32F, 0, 1, ksize=3) * 0.125 * hpx
    inv = 1.0 / np.sqrt(gx * gx + gy * gy + 1.0)
    nx, ny, nz = -gx * inv, -gy * inv, inv
    th = 2.0 * math.pi * _f(p, "dm_light_speed") * t
    el = math.atan(_f(p, "dm_light_height"))                       # висота над горизонтом: height=1 -> 45°
    lx, ly, lz = math.cos(th) * math.cos(el), math.sin(th) * math.cos(el), math.sin(el)
    lam = nx * lx + ny * ly + nz * lz
    shade = np.clip(lam / max(lz, 1e-3), 0.0, 2.5)                 # плоска поверхня = 1 (кадр не темнішає загалом)
    hx, hy, hz = lx, ly, lz + 1.0
    hn = math.sqrt(hx * hx + hy * hy + hz * hz)
    spec = np.clip((nx * hx + ny * hy + nz * hz) / hn, 0.0, 1.0) ** 48
    spec = spec * _f(p, "dm_light_spec")
    g = 1.0 + s * (shade - 1.0)
    return frame * g[..., None] + (s * spec)[..., None] * 0.6


# ---------------------------------------------------------------- вхід
def apply(frame, depth, t, p):
    """Кадр після look -> кадр з «рухом по глибині». Нічого не вмикано = повертає frame без змін (той самий обʼєкт)."""
    if not p or not active(p):
        return frame
    H, W = frame.shape[:2]
    fr = np.ascontiguousarray(frame[..., :3], np.float32)
    d = np.asarray(depth, np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    d = np.ascontiguousarray(np.clip(np.nan_to_num(d), 0.0, 1.0), np.float32)
    t = float(t)
    if _f(p, "dm_par") > 0.0:
        fr, d = _parallax(fr, d, t, p)
    if _f(p, "dm_echo") > 0.0:
        fr = _echo(fr, d, t, p)
    if _f(p, "dm_light") > 0.0:
        fr = _light(fr, d, t, p)
    if _f(p, "dm_focus") > 0.0:
        fr = _focus(fr, d, t, p)
    if _f(p, "dm_iso") > 0.0:
        fr = _iso(fr, d, t, p)
    if _f(p, "dm_scan") > 0.0:
        fr = _scanner(fr, d, t, p)
    if _f(p, "dm_reveal") > 0.0:
        fr = _reveal(fr, d, t, p)
    fr = np.clip(np.nan_to_num(fr), 0.0, 1.0).astype(np.float32, copy=False)
    if frame.shape[2] > 3:
        fr = np.concatenate([fr, frame[..., 3:]], axis=2)
    return fr
