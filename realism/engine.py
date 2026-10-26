"""Camera Realism v1: програмний «реалізм камери» на numpy (+ cv2 для блюру/ремапу). Без torch, без GPU.

apply_realism(frames, depth, params, seed, frame_offset):
  frames float32 (B,H,W,3) 0..1 (кодовані значення, як у Comfy IMAGE); depth (B|1,H,W) 0..1, світле = близько, або None;
  params - уже зведені (params.resolve: профіль x майстер-сила). Кадри обробляються по одному, у памʼяті - вхід і вихід.
Правило паку: ефект із силою 0 не обчислюється. Усі сили 0 -> повертається сам вхід.
Шум детермінований від (seed, номер кадру): щокадру новий. Фіксований патерн сенсора залежить лише від seed і розміру.

Порядок у кадрі (фізичний, від сцени до файлу камери):
  глибина (серпанок, мікродеталь) -> оптика (вуаль, мʼякі кути, віньєтка, геометрія: дисторсія/хроматика/тремтіння/rolling shutter)
  -> мерехтіння експозиції -> сенсор (шум, фікс. патерн) -> байєр -> ISP (шумодав, шарп, блоки).
"""
import math

import cv2
import numpy as np

from .params import NEUTRAL

LUMA = np.array([0.299, 0.587, 0.114], np.float32)
_SKY = np.array([0.62, 0.70, 0.82], np.float32)          # колір серпанку далі
_TAU = 2.0 * math.pi

# групи -> ключі-сили, що її вмикають
_SENSOR = ("noise_luma", "noise_chroma", "fixed_pattern")
_ISP = ("isp_denoise", "isp_sharpen", "isp_blocks")
_OPTICS = ("vignette", "distortion", "chroma_ab", "corner_softness", "veil")
_DEPTH = ("depth_haze", "depth_detail")
_VIDEO = ("shake", "flicker")      # rolling_shutter сам по собі нічого не робить: швидкість береться з тремтіння


def _on(p, *keys):
    return any(abs(p[k]) > 1e-6 for k in keys)


# ---------------------------------------------------------------- допоміжне
_GRID = {}


def _grid(H, W):
    """Центровані координати пікселів (u, v) і r² (кут кадру = 1). Кеш на один розмір."""
    key = (H, W)
    g = _GRID.get("g")
    if _GRID.get("k") != key:
        cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
        u = (np.arange(W, dtype=np.float32) - cx)[None, :].repeat(H, 0)
        v = (np.arange(H, dtype=np.float32) - cy)[:, None].repeat(W, 1)
        inv_r2 = np.float32(1.0 / (cx * cx + cy * cy + 1e-6))
        g = (u, v, inv_r2, cx, cy)
        _GRID["k"], _GRID["g"] = key, g
    return g


_FP = {}


def _fixed_pattern(seed, H, W):
    """Фіксований патерн сенсора: однаковий для всіх кадрів (залежить лише від seed і розміру)."""
    key = (int(seed), H, W)
    if _FP.get("k") != key:
        rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 9001])
        col = rng.standard_normal(W).astype(np.float32)
        row = rng.standard_normal(H).astype(np.float32)
        prnu = rng.standard_normal((H, W)).astype(np.float32)
        _FP["k"], _FP["v"] = key, (col, row, prnu)
    return _FP["v"]


def _luma(f):
    return f @ LUMA


def _big_blur(img, sigma):
    """Гаусів блюр з великою сигмою: рахується на зменшеній копії (дешево на 4K)."""
    H, W = img.shape[:2]
    s = max(1, int(sigma // 4))
    if s == 1:
        return cv2.GaussianBlur(img, (0, 0), sigma)
    small = cv2.resize(img, (max(2, W // s), max(2, H // s)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), sigma / s)
    return cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR)


def _unit_noise(rng, H, W, sigma):
    """Гаусів шум одиничної дисперсії; sigma > 0 збиває його в грудки (грубе зерно)."""
    n = rng.standard_normal((H, W), dtype=np.float32)
    if sigma > 0.05:
        n = cv2.GaussianBlur(n, (0, 0), sigma)
        n /= max(float(n.std()), 1e-6)
    return n


def _wobble(seed, axis, f, speed):
    """Гладкий «шум» від номера кадру: сума синусоїд із неспівмірними періодами (чиста функція f), std ~ 1."""
    rng = np.random.default_rng([int(seed) & 0xFFFFFFFF, 7001 + axis])
    ph = rng.uniform(0.0, _TAU, 4)
    per = (np.array([47.0, 29.0, 17.3, 9.1]) / max(speed, 1e-3))
    amp = np.array([1.0, 0.7, 0.45, 0.25])
    v = float(np.sum(amp * np.sin(_TAU * f / per + ph)))
    return v / math.sqrt(float(np.sum(amp * amp)) / 2.0)


# ---------------------------------------------------------------- глибина
def _depth_group(f, d, p):
    H, W = f.shape[:2]
    if p["depth_haze"] > 1e-6:
        far = (1.0 - d) ** 1.5
        t = (p["depth_haze"] * 0.6 * far)[..., None]
        gray = _luma(f)[..., None]
        lum = float(np.clip(gray.mean() * 1.4, 0.45, 1.0))      # світло серпанку під експозицію кадру, щоб темна сцена не світлішала
        f = f * (1.0 - 0.6 * t) + gray * (0.6 * t)              # менше насиченості
        f = f * (1.0 - t) + (_SKY * lum) * t                    # менше контрасту, у тон неба
    if p["depth_detail"] > 1e-6:
        near = (d ** 1.5)
        y = _luma(f)
        dy = y - cv2.GaussianBlur(y, (0, 0), max(1.5, 0.0025 * max(H, W)))
        f = f + ((0.9 * p["depth_detail"]) * near * dy)[..., None]
    return f


# ---------------------------------------------------------------- оптика
def _optics_pre(f, p):
    H, W = f.shape[:2]
    if p["veil"] > 1e-6:
        v = min(p["veil"], 1.0)
        glow = _big_blur(f, 0.03 * max(H, W))
        f = f * (1.0 - 0.2 * v) + glow * (0.2 * v) + 0.025 * v
    if p["corner_softness"] > 1e-6:
        s = min(p["corner_softness"], 1.0)
        u, v, inv_r2, _, _ = _grid(H, W)
        r = np.sqrt((u * u + v * v) * inv_r2)
        w = np.clip((r - 0.35) / 0.65, 0.0, 1.0)
        w = (w * w * (3.0 - 2.0 * w))[..., None]
        blur = cv2.GaussianBlur(f, (0, 0), max(0.8, 0.0025 * max(H, W) * s * 1.5))
        f = f + (blur - f) * w
    if p["vignette"] > 1e-6:
        s = min(p["vignette"], 1.0)
        u, v, inv_r2, _, _ = _grid(H, W)
        r2 = (u * u + v * v) * inv_r2
        cos4 = 1.0 / (1.0 + 1.44 * r2) ** 2                      # cos⁴θ, tanθ = 1.2·r
        gain = (1.0 - s * (1.0 - cos4)) ** 0.4545                # множення в лінійному світлі ~ у кодованому через 1/2.2
        f = f * gain[..., None]
    return f


def _geometry(f, p, tx, ty, rot, rs_row, zoom):
    """Один ремап на кадр: тремтіння (зсув/поворот/rolling shutter) -> дисторсія -> хроматика (R і B з іншим масштабом)."""
    H, W = f.shape[:2]
    u, v, inv_r2, cx, cy = _grid(H, W)
    if tx is not None:
        c, s = math.cos(rot), math.sin(rot)
        uu = (c * u - s * v) / zoom + tx
        vv = (s * u + c * v) / zoom + ty
        if rs_row is not None:
            uu = uu + rs_row[:, None]
    else:
        uu, vv = u, v
    k = 0.22 * p["distortion"]
    if abs(k) > 1e-9:
        r2 = (uu * uu + vv * vv) * inv_r2
        sc = (1.0 + k * r2) / (1.0 + max(k, 0.0))                # нормування: кути не виходять за кадр
    else:
        sc = None
    ca = 0.0016 * p["chroma_ab"]

    def maps(scale):
        if sc is None and scale == 1.0:
            return (cx + uu).astype(np.float32), (cy + vv).astype(np.float32)
        m = scale if sc is None else sc * scale
        return (cx + uu * m).astype(np.float32), (cy + vv * m).astype(np.float32)

    kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    if ca > 1e-9:
        planes = cv2.split(np.ascontiguousarray(f))
        out = []
        for ch, scale in zip(planes, (1.0 + ca, 1.0, 1.0 - ca)):   # R трохи більший, B трохи менший; G - еталон
            mx, my = maps(scale)
            out.append(cv2.remap(ch, mx, my, **kw))
        return cv2.merge(out)
    mx, my = maps(1.0)
    return cv2.remap(np.ascontiguousarray(f), mx, my, **kw)


# ---------------------------------------------------------------- сенсор
def _sensor(f, p, rng, seed):
    H, W = f.shape[:2]
    y = np.clip(_luma(f), 0.0, 1.0)
    if p["noise_luma"] > 1e-6 or p["noise_chroma"] > 1e-6:
        isp = max(p["iso"], 1.0) / 800.0
        a, r = 0.008 * isp, 0.002 * math.sqrt(isp)
        L = np.power(y, 2.2)
        # шум у лінійному світлі sqrt(a·L + r²) -> у кодований простір через похідну кривої: тіні шумлять сильніше
        sig = np.sqrt(a * L + r * r) * 0.4545 * np.power(np.maximum(L, 0.002), -0.545)
        np.minimum(sig, 0.3, out=sig)
        if p["noise_luma"] > 1e-6:
            n = _unit_noise(rng, H, W, 0.35 + p["noise_size"] * 1.6 if p["noise_size"] > 0.02 else 0.0)
            f = f + (p["noise_luma"] * sig * n)[..., None]
        if p["noise_chroma"] > 1e-6:
            sc = 0.9 + p["noise_size"] * 1.5
            cb = _unit_noise(rng, H, W, sc)
            cr = _unit_noise(rng, H, W, sc)
            ac = (p["noise_chroma"] * 0.6) * sig
            nr = 1.402 * cr
            nb = 1.772 * cb
            ng = -0.344 * cb - 0.714 * cr
            f = f + np.stack([ac * nr, ac * ng, ac * nb], axis=-1)
    if p["fixed_pattern"] > 1e-6:
        col, row, prnu = _fixed_pattern(seed, H, W)
        fp = p["fixed_pattern"]
        pat = (0.0030 * col[None, :] + 0.0020 * row[:, None]) * (1.0 - y) ** 2     # смуги видніші в тінях
        f = f * (1.0 + (fp * 0.006) * prnu)[..., None] + (fp * pat)[..., None]
    return f


# ---------------------------------------------------------------- байєр
_MASKS = {}
_K_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32) / 4.0
_K_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], np.float32) / 4.0


def _bayer(f, amount):
    H, W = f.shape[:2]
    m = _MASKS.get((H, W))
    if m is None:
        yy, xx = np.indices((H, W))
        rm = ((yy % 2 == 0) & (xx % 2 == 0)).astype(np.float32)
        bm = ((yy % 2 == 1) & (xx % 2 == 1)).astype(np.float32)
        m = (rm, 1.0 - rm - bm, bm)
        _MASKS.clear()
        _MASKS[(H, W)] = m
    rm, gm, bm = m
    mos = f[..., 0] * rm + f[..., 1] * gm + f[..., 2] * bm
    bd = cv2.BORDER_REFLECT_101                                  # дзеркало відносно краю зберігає парність пікселя - патерн не ламається
    r = cv2.filter2D(mos * rm, -1, _K_RB, borderType=bd)
    g = cv2.filter2D(mos * gm, -1, _K_G, borderType=bd)
    b = cv2.filter2D(mos * bm, -1, _K_RB, borderType=bd)
    demo = np.stack([r, g, b], axis=-1)
    a = min(amount, 1.0)
    return f + (demo - f) * a if a < 1.0 else demo


# ---------------------------------------------------------------- ISP
_D8 = np.array([[(math.sqrt(1 / 8) if k == 0 else math.sqrt(2 / 8)) * math.cos((2 * n + 1) * k * math.pi / 16)
                 for n in range(8)] for k in range(8)], np.float32)
_QY = np.array([[16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55], [14, 13, 16, 24, 40, 57, 69, 56],
                [14, 17, 22, 29, 51, 87, 80, 62], [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
                [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], np.float32)
_QC = np.full((8, 8), 99.0, np.float32)
_QC[:4, :4] = np.array([[17, 18, 24, 47], [18, 21, 26, 66], [24, 26, 56, 99], [47, 66, 99, 99]], np.float32)


def _qtable(base, quality):
    sc = 5000.0 / quality if quality < 50 else 200.0 - 2.0 * quality
    return np.clip(np.floor((base * sc + 50.0) / 100.0), 1.0, 255.0).astype(np.float32)


def _dct_quant(plane, q):
    """Квантування 8×8 DCT однієї площини (значення 0..255, зсув -128)."""
    H, W = plane.shape
    ph, pw = (-H) % 8, (-W) % 8
    if ph or pw:
        plane = np.pad(plane, ((0, ph), (0, pw)), mode="edge")
    Hp, Wp = plane.shape
    blk = (plane - 128.0).reshape(Hp // 8, 8, Wp // 8, 8).transpose(0, 2, 1, 3)
    co = _D8 @ blk @ _D8.T
    co = np.round(co / q) * q
    rec = (_D8.T @ co @ _D8).transpose(0, 2, 1, 3).reshape(Hp, Wp) + 128.0
    return rec[:H, :W]


def _blocks(f, amount):
    quality = 90.0 - 60.0 * min(amount, 1.0)
    qy, qc = _qtable(_QY, quality), _qtable(_QC, quality)
    x = f * 255.0
    Y = x @ LUMA
    Cb = (x[..., 2] - Y) * 0.564 + 128.0
    Cr = (x[..., 0] - Y) * 0.713 + 128.0
    Y = _dct_quant(Y, qy)
    Cb = _dct_quant(Cb, qc) - 128.0
    Cr = _dct_quant(Cr, qc) - 128.0
    R = Y + 1.403 * Cr
    B = Y + 1.773 * Cb
    G = (Y - 0.299 * R - 0.114 * B) / 0.587
    return np.stack([R, G, B], axis=-1) / 255.0


def _isp(f, p):
    if p["isp_denoise"] > 1e-6:
        a = min(p["isp_denoise"], 1.0)
        src = np.ascontiguousarray(f, dtype=np.float32)
        bil = cv2.bilateralFilter(src, 5, 0.04 + 0.12 * a, 2.0)
        if a > 0.6:                                              # сильний шумодав: другий прохід - «віск»
            bil = cv2.bilateralFilter(bil, 5, 0.04 + 0.12 * a, 2.0)
        w = min(1.0, 1.2 * a)
        f = f + (bil - f) * w
    if p["isp_sharpen"] > 1e-6:
        y = _luma(f)
        dy = y - cv2.GaussianBlur(y, (0, 0), max(0.4, p["isp_sharpen_radius"]))
        f = f + (2.0 * p["isp_sharpen"] * dy)[..., None]
    if p["isp_blocks"] > 1e-6:
        f = _blocks(f, p["isp_blocks"])
    return f


# ---------------------------------------------------------------- головна
def apply_realism(frames, depth, params, seed=0, frame_offset=0):
    """Див. шапку модуля. frames не змінюється; якщо жодна група не активна, повертається сам frames."""
    p = {k: float(params.get(k, NEUTRAL[k])) for k in NEUTRAL}
    seed = int(seed)
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    B, H, W = frames.shape[:3]
    have_depth = depth is not None
    multi = B > 1 or int(frame_offset) > 0

    do_depth = have_depth and _on(p, *_DEPTH)
    do_optics = _on(p, *_OPTICS)
    do_video = multi and _on(p, *_VIDEO)
    do_shake = multi and p["shake"] > 1e-6
    do_flicker = multi and p["flicker"] > 1e-6
    do_sensor = _on(p, *_SENSOR)
    do_bayer = p["bayer_amount"] > 1e-6
    do_isp = _on(p, *_ISP)
    do_geo = _on(p, "distortion", "chroma_ab") or do_shake
    if not (do_depth or do_optics or do_video or do_sensor or do_bayer or do_isp):
        return frames

    if do_shake:
        amp = p["shake"] * 0.003 * max(H, W)                     # std зсуву, px
        rot_amp = math.radians(0.25 * p["shake"])                # std повороту
        margin = 2.2 * amp + 2.2 * rot_amp * 0.5 * max(H, W)
        zoom = 1.0 + 2.0 * margin / min(H, W)                    # постійний кроп: краї не показують відзеркалення
        speed = p["shake_speed"]

    dep = None
    if do_depth:
        dep = np.asarray(depth, dtype=np.float32)
        if dep.ndim == 2:
            dep = dep[None]

    out = np.empty_like(frames)
    for i in range(B):
        fi = int(frame_offset) + i
        f = frames[i].copy()
        if do_depth:
            d = dep[min(i, dep.shape[0] - 1)]
            if d.shape != (H, W):
                d = cv2.resize(d, (W, H), interpolation=cv2.INTER_LINEAR)
            f = _depth_group(f, np.clip(d, 0.0, 1.0), p)
        if do_optics:
            f = _optics_pre(f, p)
        if do_geo:
            tx = ty = rot = rs_row = None
            z = 1.0
            if do_shake:
                tx = amp * _wobble(seed, 0, fi, speed)
                ty = 0.7 * amp * _wobble(seed, 1, fi, speed)
                rot = rot_amp * _wobble(seed, 2, fi, speed)
                z = zoom
                if p["rolling_shutter"] > 1e-6:                   # швидкість тремтіння -> зсув рядків від верху до низу кадру
                    vx = tx - amp * _wobble(seed, 0, fi - 1, speed)
                    rows = (np.arange(H, dtype=np.float32) / max(H - 1, 1) - 0.5)
                    rs_row = (1.5 * p["rolling_shutter"] * vx) * rows
            f = _geometry(f, p, tx, ty, rot, rs_row, z)
        if do_flicker:
            rg = np.random.default_rng([seed & 0xFFFFFFFF, 7100, fi])
            g = 0.5 * _wobble(seed, 3, fi, 2.0) + 0.5 * float(rg.standard_normal())
            f = f * (1.0 + 0.06 * p["flicker"] * g)
        if do_sensor:
            rng = np.random.default_rng([seed & 0xFFFFFFFF, 5001, fi])
            f = _sensor(f, p, rng, seed)
        if do_bayer:
            f = _bayer(f, p["bayer_amount"])
        if do_isp:
            f = _isp(f, p)
        np.clip(f, 0.0, 1.0, out=f)
        out[i] = f
    return out
