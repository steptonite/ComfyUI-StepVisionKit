"""Camera Realism v2: підписи пристроїв і спільне ядро (крива світла, колір, моушн-блюр). Викликається з engine.apply_realism.
Кожна функція бере кадр f float32 (H,W,3) 0..1 (кодовані значення) і повертає новий; сила 0 -> функцію не кличуть.

Ядро:   _tone_colour (мʼякий rolloff / жорсткий кліп світлих, чорні, тепло, насиченість, відтінок шкіри), motion_blur (затвор за оптичним потоком).
16mm:   halation (червоне сяйво довкола світлих), film_grain (зерно по шарах емульсії), gate weave (у engine, через геометрію), dust (пил, волосинки, подряпина).
iPhone: smart_hdr (локальний тонмап з ореолами), deep_fusion (пересушена текстура), shadow_smear (акварель у тінях),
        portrait_bokeh (фальшиве боке за глибиною з помилками на краях), ghost_flare (зелений привид світлих точок, дзеркально через центр).
DSLR 2012: line_skip (пропуск рядків: муар і мʼякість), chroma_blotch (великі пурпурно-зелені плями в тінях), banding (смуги 8 біт у градієнтах).
"""
import math

import cv2
import numpy as np

LUMA = np.array([0.299, 0.587, 0.114], np.float32)


def _luma(f):
    return f @ LUMA


def _smooth(x, a, b):
    t = np.clip((x - a) / (b - a), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blur_big(img, sigma):
    H, W = img.shape[:2]
    s = max(1, int(sigma // 4))
    if s == 1:
        return cv2.GaussianBlur(img, (0, 0), max(sigma, 0.3))
    small = cv2.resize(img, (max(2, W // s), max(2, H // s)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def _noise(rng, H, W, sigma):
    n = rng.standard_normal((H, W), dtype=np.float32)
    if sigma > 0.05:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= max(float(n.std()), 1e-6)
    return n


# ---------------------------------------------------------------- ядро: крива світла і колір
def tone_colour(f, p):
    y = _luma(f)
    s = p["highlight_rolloff"]
    if s > 1e-6:                                   # мʼяке плече: верх стискається, верхні середні світлішають, світлі знебарвлюються (ARRI, плівка)
        s = min(s, 1.0)
        k = 1.0 - 0.6 * s
        t = np.clip((y - k) / (1.0 - k), 0.0, 1.0)
        g = t + 0.85 * s * t * t * (1.0 - t)
        y2 = np.where(y > k, k + (1.0 - k) * g, y)
        f = f * (y2 / np.maximum(y, 1e-4))[..., None]
        w = (0.35 * s * _smooth(y2, k, 1.0))[..., None]
        f = f + (y2[..., None] - f) * w
        y = y2
    s = p["highlight_clip"]
    if s > 1e-6:                                   # жорсткий кліп по каналах: пласкі білі плями, зсув відтінку (відео 8 біт)
        f = f / (1.0 - 0.2 * min(s, 1.0))
        np.clip(f, 0.0, 1.0, out=f)
    b = p["blacks"]
    if b > 1e-6:                                   # «вбиті» чорні
        c = 0.07 * min(b, 1.0)
        f = np.maximum(f - c, 0.0) / (1.0 - c)
    elif b < -1e-6:                                # підняті тіні (плівка, телефон)
        c = 0.08 * min(-b, 1.0)
        f = f * (1.0 - c) + c * (1.0 - f) ** 3
    w = p["warmth"]
    if abs(w) > 1e-6:
        f = f * np.array([1.0 + 0.07 * w, 1.0 + 0.01 * w, 1.0 - 0.08 * w], np.float32)
    h = p["skin_hue"]
    if abs(h) > 1e-6:                              # тілесні відтінки: + до жовто-зеленого (Sony), - до пурпурно-червоного (Canon)
        hsv = cv2.cvtColor(np.clip(f, 0.0, 1.0).astype(np.float32), cv2.COLOR_RGB2HSV)
        hd = np.abs(((hsv[..., 0] - 25.0 + 180.0) % 360.0) - 180.0)
        m = np.clip(1.0 - hd / 30.0, 0.0, 1.0) * np.clip(hsv[..., 1] * 3.0, 0.0, 1.0)
        hsv[..., 0] = (hsv[..., 0] + 9.0 * h * m) % 360.0
        f = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    s = p["saturation"]
    if abs(s) > 1e-6:
        y = _luma(f)[..., None]
        f = y + (f - y) * (1.0 + 0.4 * s)
    return f


# ---------------------------------------------------------------- моушн-блюр затвора
def flow_velocity(frames, i):
    """Рух за кадр (px, повна роздільність) у кадрі i: потік між сусідами (центральна різниця), ~480 px по довгій стороні."""
    B, H, W = frames.shape[:3]
    a, b = max(i - 1, 0), min(i + 1, B - 1)
    if a == b:
        return None
    sc = min(1.0, 480.0 / max(H, W))
    sw, sh = max(8, int(W * sc)), max(8, int(H * sc))

    def g(k):
        return cv2.resize((np.clip(_luma(frames[k]), 0, 1) * 255).astype(np.uint8), (sw, sh), interpolation=cv2.INTER_AREA)
    fl = cv2.calcOpticalFlowFarneback(g(a), g(b), None, 0.5, 3, 15, 3, 5, 1.2, 0)
    fl = fl / (float(b - a) * sc)
    return cv2.resize(fl, (W, H), interpolation=cv2.INTER_LINEAR)


def motion_blur(f, vel, shutter):
    """Розмиття вздовж руху: shutter 0..1 = частка кадрового часу (0.5 = затвор 180°)."""
    H, W = f.shape[:2]
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    vx, vy = vel[..., 0] * shutter, vel[..., 1] * shutter
    taps = 7
    acc = np.zeros_like(f)
    src = np.ascontiguousarray(f)
    for t in np.linspace(-0.5, 0.5, taps, dtype=np.float32):
        acc += cv2.remap(src, gx + t * vx, gy + t * vy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return acc / taps


# ---------------------------------------------------------------- 16mm плівка
def halation(f, s):
    H, W = f.shape[:2]
    y = _luma(f)
    m = (_smooth(y, 0.62, 1.0) * y).astype(np.float32)
    glow = _blur_big(m, 0.010 * max(H, W))
    wide = _blur_big(m, 0.028 * max(H, W))
    g = (0.65 * glow + 0.35 * wide)[..., None]
    return f + (0.55 * min(s, 1.5)) * g * np.array([1.0, 0.28, 0.06], np.float32)


def film_grain(f, s, rng):
    """Зерно трьох шарів емульсії: синій шар найгрубший, зерно в тінях і середніх тонах, у світлих майже зникає."""
    H, W = f.shape[:2]
    y = np.clip(_luma(f), 0.0, 1.0)
    w = np.clip(0.25 + 2.6 * y * (1.0 - y) ** 1.4, 0.0, 1.0) * (1.0 - _smooth(y, 0.8, 1.0) * 0.7)
    px = max(0.7, 0.0011 * max(H, W))
    common = 0.65 * _noise(rng, H, W, px) + 0.35 * _noise(rng, H, W, px * 2.4)
    out = []
    for size, amp in ((1.0, 0.85), (0.85, 0.75), (1.35, 1.2)):
        own = 0.6 * _noise(rng, H, W, px * size) + 0.4 * _noise(rng, H, W, px * size * 2.4)
        out.append(amp * (0.6 * common + 0.4 * own))
    n = np.stack(out, axis=-1)
    return f + (0.075 * min(s, 1.5)) * w[..., None] * n


def dust(f, s, seed, fi):
    """Пил і волосинки (світлі на копії з негатива, іноді темні), рідка вертикальна подряпина, що живе кілька десятків кадрів."""
    H, W = f.shape[:2]
    rng = np.random.default_rng([seed & 0xFFFFFFFF, 8100, fi])
    sc = max(H, W) / 1000.0
    m_w = np.zeros((H, W), np.float32)
    m_b = np.zeros((H, W), np.float32)
    n = int(rng.poisson(10.0 * min(s, 1.5)))
    for _ in range(n):
        m = m_w if rng.random() < 0.75 else m_b
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        if rng.random() < 0.15:                      # волосинка: коротка крива
            pts = [(x, y)]
            ang = rng.uniform(0, 2 * math.pi)
            for _k in range(6):
                ang += rng.normal(0, 0.5)
                x += int(math.cos(ang) * 6 * sc)
                y += int(math.sin(ang) * 6 * sc)
                pts.append((x, y))
            cv2.polylines(m, [np.array(pts, np.int32)], False, float(rng.uniform(0.5, 1.0)), max(1, int(round(sc))))
        else:
            cv2.circle(m, (x, y), max(1, int(round(rng.uniform(0.6, 2.8) * sc))), float(rng.uniform(0.5, 1.0)), -1)
    epoch = fi // 40                                 # подряпина: та сама протягом ~40 кадрів, зʼявляється не завжди
    re = np.random.default_rng([seed & 0xFFFFFFFF, 8200, epoch])
    if re.random() < 0.45 * min(s, 1.0):
        x0 = int(re.integers(int(0.1 * W), int(0.9 * W))) + int(rng.integers(-2, 3))
        cv2.line(m_w, (x0, 0), (x0 + int(re.integers(-4, 5)), H - 1), 0.35, max(1, int(round(sc))))
    m_w = cv2.GaussianBlur(m_w, (0, 0), 0.6 * sc + 0.3)[..., None]
    m_b = cv2.GaussianBlur(m_b, (0, 0), 0.6 * sc + 0.3)[..., None]
    a = min(s * 1.3, 1.0)
    return f + (1.0 - f) * m_w * a - f * m_b * a


# ---------------------------------------------------------------- iPhone
def smart_hdr(f, s):
    """Локальний тонмап: тіні підняті, світлі приплющені, ореоли біля контрастних країв (базовий шар - гаусів, без edge-aware)."""
    H, W = f.shape[:2]
    y = np.clip(_luma(f), 1e-4, 1.0)
    L = np.log(y + 0.02)
    base = _blur_big(L, 0.018 * max(H, W))
    mu = float(base.mean())
    k = min(s, 1.0)
    L2 = mu + (base - mu) * (1.0 - 0.55 * k) + (L - base) * (1.0 + 0.3 * k)
    y2 = np.exp(L2) - 0.02
    y2 = y2 * (0.92 / max(float(np.percentile(y2, 99.5)), 1e-3)) ** (0.5 * k)
    return f * (np.clip(y2, 0.0, 1.2) / y)[..., None]


def deep_fusion(f, s):
    """Пересушена текстура: підсилення середніх частот лише там, де вже є фактура (волосся, тканина, шкіра)."""
    y = _luma(f)
    band = cv2.GaussianBlur(y, (0, 0), 0.7) - cv2.GaussianBlur(y, (0, 0), 2.4)
    tex = cv2.GaussianBlur(np.abs(band), (0, 0), 3.0)
    w = _smooth(tex, 0.004, 0.02)
    return f + (2.2 * min(s, 1.5) * band * w)[..., None]


def shadow_smear(f, s):
    """Шумодав телефона в тінях: дрібна фактура розмазується в плями («акварель»)."""
    H, W = f.shape[:2]
    y = _luma(f)
    m = (_smooth(0.42 - y, 0.0, 0.3) * min(s, 1.0))[..., None]
    src = np.ascontiguousarray(f, dtype=np.float32)
    small = cv2.resize(src, (max(2, W // 2), max(2, H // 2)), interpolation=cv2.INTER_AREA)
    small = cv2.bilateralFilter(small, 7, 0.10, 3.0)
    small = cv2.bilateralFilter(small, 7, 0.10, 3.0)
    sm = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)
    return f + (sm - f) * m


def portrait_bokeh(f, d, s):
    """Портретний режим: фон за глибиною розмивається трьома рівнями; межу дає груба карта глибини - звідси помилки на волоссі."""
    H, W = f.shape[:2]
    focus = float(np.percentile(d, 93))
    far = np.clip((focus - 0.06 - d) / 0.35, 0.0, 1.0)
    far = cv2.GaussianBlur(far, (0, 0), 1.5)
    r = 0.014 * max(H, W) * min(s, 1.5)
    lv = [f] + [_blur_big(f, r * q) for q in (0.33, 0.66, 1.0)]
    x = far * 3.0
    out = np.zeros_like(f)
    for k in range(4):
        wk = np.clip(1.0 - np.abs(x - k), 0.0, 1.0)[..., None]
        out += lv[k] * wk
    return out


def ghost_flare(f, s):
    """Зелений «привид» яскравих точок, дзеркально через центр кадру (відбиття між лінзою й сенсором телефона)."""
    H, W = f.shape[:2]
    y = _luma(f)
    br = _smooth(y, 0.88, 1.0)
    if float(br.max()) < 0.05:
        return f
    g = cv2.GaussianBlur(br[::-1, ::-1].copy(), (0, 0), 0.004 * max(H, W) + 0.5)
    g = g / max(float(g.max()), 1e-3)
    return f + (0.35 * min(s, 1.5)) * g[..., None] * np.array([0.35, 1.0, 0.6], np.float32)


# ---------------------------------------------------------------- DSLR 2012
def line_skip(f, s):
    """Відео з пропуском рядків (Canon 5D Mk III / 7D): рядки беруться через k, стовпці усереднюються, потім розтяг назад."""
    H, W = f.shape[:2]
    k = 1.0 + 2.0 * min(s, 1.0)
    hs, ws = max(2, int(round(H / k))), max(2, int(round(W / k)))
    rows = np.minimum((np.arange(hs) * k).astype(np.int64), H - 1)
    small = cv2.resize(np.ascontiguousarray(f[rows]), (ws, hs), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def chroma_blotch(f, s, rng):
    """Великі плями кольорового шуму в тінях, переважно по осі пурпур-зелений."""
    H, W = f.shape[:2]
    y = np.clip(_luma(f), 0.0, 1.0)
    m = (1.0 - y) ** 2.2
    sig = max(2.0, 0.007 * max(H, W))
    n1 = _noise(rng, H, W, sig)
    n2 = _noise(rng, H, W, sig * 1.6)
    a = 0.05 * min(s, 1.5) * m
    return f + np.stack([a * (0.6 * n1 + 0.2 * n2), a * (-1.0 * n1), a * (0.6 * n1 - 0.35 * n2)], axis=-1)


def banding(f, s):
    """Смуги 8-бітного кодування в плавних градієнтах (небо, стіни), без дизеру; на фактурі не видно."""
    y = _luma(f)
    gx = cv2.Sobel(y, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(y, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.GaussianBlur(np.sqrt(gx * gx + gy * gy), (0, 0), 2.0)
    w = (1.0 - _smooth(grad, 0.01, 0.05))[..., None]
    levels = 255.0 - 205.0 * min(s, 1.0)
    q = np.round(f * levels) / levels
    return f + (q - f) * w


def own_grain(f, p, seed, fi):
    """Власне зерно з повзунків «Own grain»: розмір у px кадру 1000 px, грудкуватість, мʼякість, кольоровість (синій шар
    окремого розміру), відгук по тінях/середніх/світлих і частка зерна, що міняється щокадру (0 = прилипле до екрана)."""
    H, W = f.shape[:2]
    px = max(0.3, p["og_size"] * max(H, W) / 1000.0)
    c, soft = float(np.clip(p["og_clump"], 0, 1)), float(np.clip(p["og_softness"], 0, 1))
    anim = float(np.clip(p["og_animate"], 0, 1))
    s = seed & 0xFFFFFFFF

    def layer(rng, size):
        n = (1.0 - c) * _noise(rng, H, W, 0.5 * px * size) + c * _noise(rng, H, W, 2.0 * px * size)
        n /= math.hypot(1.0 - c, c)
        if soft > 0.02:                              # мʼякість гасить і край, і частину контрасту зерна
            n = cv2.GaussianBlur(n, (0, 0), 0.8 * soft * px * size)
            n /= max(float(n.std()), 1e-6) ** 0.5
        return n

    def field(tag, size):
        dyn = layer(np.random.default_rng([s, 6201, tag, fi]), size) if anim > 1e-3 else 0.0
        st = layer(np.random.default_rng([s, 6202, tag]), size) if anim < 1.0 - 1e-3 else 0.0
        return (anim * dyn + (1.0 - anim) * st) / math.hypot(anim, 1.0 - anim)

    col = float(np.clip(p["og_colour"], 0, 1))
    mono = field(0, 1.0)
    if col < 0.01:
        n = mono[..., None]
    else:
        k = math.hypot(1.0 - col, col)
        n = np.stack([((1.0 - col) * mono + col * field(1 + i, sz)) / k
                      for i, sz in enumerate((1.0, 0.9, float(p["og_blue"])))], axis=-1)
    y = np.clip(_luma(f), 0.0, 1.0)
    w = p["og_shadows"] * (1.0 - y) ** 2 + p["og_mids"] * 4.0 * y * (1.0 - y) + p["og_highlights"] * y * y
    return f + (0.08 * p["og_amount"]) * w[..., None] * n
