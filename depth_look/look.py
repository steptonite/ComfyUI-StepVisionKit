"""depth_look: «look від глибини» — прототип майбутньої Comfy-ноди.

Усе — float32 0..1, RGB. Залежності: лише numpy + cv2. Без torch, без мережі.

Конвеєр (кожен крок = група повзунків):
  1. clean_depth   : float, edge-aware (guided filter через boxFilter, фото = guide), без квантування
  2. tone          : яскравість фото -> t (глибокі чорні)  [tone_*]
  3. color_source  : t = luma | depth | mix              [color_source, mix, depth_*]
  4. black_by_depth: затемнення далекого/близького        [black_by_depth, black_curve]
  5. blur+streak   : розмиття за |depth-focus| + directional по фону  [focus_*, blur_strength, streak_*]
  6. gradient map  : t -> 3-5 тонів                        [gradient_stops]
  7. chroma_ab     : радіальний зсув R/B                   [chroma_ab]
  8. halation/bloom: червоне сяйво по яскравому            [halation_*, bloom]
  9. grain         : film_grain() як Camera Raw (amount/size/roughness 0..100) [grain_*, seed]
"""
import os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # EXR-глибина; діє лише якщо cv2 ще не імпортовано
import cv2
import numpy as np

DEFAULT_STOPS = [  # реф IMG_4672: чорний -> фіолет -> лаванда -> оранж -> тепле світле
    (0.012, 0.008, 0.018),
    (0.17, 0.10, 0.36),
    (0.66, 0.64, 0.92),
    (0.96, 0.60, 0.26),
    (1.00, 0.90, 0.72),
]


def _hex(c):
    if isinstance(c, str):
        c = c.lstrip("#")
        return tuple(int(c[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    return tuple(c)


def _smooth(x, a, b):
    t = np.clip((x - a) / max(b - a, 1e-6), 0, 1)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------- depth
def _guided(I, p, r, eps):
    """Guided filter (He et al.), I = guide (HxW float), p = вхід. Лише boxFilter."""
    k = (2 * r + 1, 2 * r + 1)
    bf = lambda x: cv2.boxFilter(x, -1, k, borderType=cv2.BORDER_REFLECT)
    mI, mp = bf(I), bf(p)
    cov = bf(I * p) - mI * mp
    var = bf(I * I) - mI * mI
    a = cov / (var + eps)
    b = mp - a * mI
    return bf(a) * I + bf(b)


def prepare_depth(depth, guide_rgb, light_is_near=True, passes=((4, 1e-4), (10, 4e-4), (20, 1e-3)),
                pre_sigma=2.5, stretch=True):
    """depth: HxW (uint8/uint16/float, будь-який розмір) -> float32 0..1 розміру guide_rgb.
    pre_sigma — гаус ДО апскейлу (ламає сходинки 8-біт/білінійного ×4), далі guided filter
    кількома проходами (дрібний радіус тримає край, великий вирівнює пологі градієнти)."""
    d = np.asarray(depth)
    if d.ndim == 3:      # 1/3/4 канали (RGB/BGR/RGBA-«сіра» карта): якщо канали однакові - беремо перший, інакше середнє перших трьох
        if d.shape[2] == 1:
            d = d[..., 0]
        elif d.shape[2] >= 3 and not np.array_equal(d[..., 0], d[..., 1]):
            d = d[..., :3].astype(np.float32).mean(-1)
        else:
            d = d[..., 0]
    isint = np.issubdtype(d.dtype, np.integer)
    dt = d.dtype
    d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if isint:            # 8/16 біт - за типом (не за max, бо темна 16-бітна карта ламала масштаб)
        d /= 65535.0 if dt.itemsize >= 2 else 255.0
    else:
        mx = float(d.max()) if d.size else 0.0
        if mx > 1.0 + 1e-3 or float(d.min()) < -1e-3:   # float не 0..1 (метри, EXR, 0..255 у float) - min-max, без огрублення
            mn = float(d.min()); d = (d - mn) / max(mx - mn, 1e-12)
    if pre_sigma > 0:
        d = cv2.GaussianBlur(d, (0, 0), pre_sigma)
    H, W = guide_rgb.shape[:2]
    dh, dw = d.shape[:2]
    if (dh > dw * 1.1 and W > H * 1.1) or (dw > dh * 1.1 and H > W * 1.1):
        raise ValueError("depth %dx%d має іншу орієнтацію, ніж кадр %dx%d" % (dw, dh, W, H))
    d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)   # трохи інше співвідношення сторін - ок, тягнемо до кадру
    g = cv2.cvtColor(guide_rgb.astype(np.float32), cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (0, 0), 1.0)
    for r, eps in passes:
        d = _guided(g, d, r, eps)
    d = np.clip(d, 0, 1)
    if stretch:          # покадровий розтяг 0.5/99.5 %; для відео - stretch=False і рівні з усього кліпу (clip_levels)
        lo, hi = np.percentile(d, 0.5), np.percentile(d, 99.5)
        d = np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1)
    return d if light_is_near else 1.0 - d


def clip_levels(depths, lo_q=0.5, hi_q=99.5, sample=8):
    """Спільні рівні для кліпу: квантилі по кількох кадрах (stretch=False), щоб глибина не тремтіла між кадрами."""
    idx = np.linspace(0, len(depths) - 1, min(sample, len(depths))).round().astype(int)
    v = np.concatenate([np.asarray(depths[i], np.float32)[::4, ::4].ravel() for i in idx])
    return float(np.percentile(v, lo_q)), float(np.percentile(v, hi_q))


def apply_levels(d, lo, hi):
    return np.clip((d - lo) / max(hi - lo, 1e-6), 0, 1).astype(np.float32)


clean_depth = prepare_depth  # старе ім'я


# ---------------------------------------------------------------- helpers
def _gradient_map(t, stops):
    stops = [_hex(s) for s in stops]
    n = len(stops)
    xs = np.linspace(0, 1, n)
    S = np.array(stops, np.float32)
    t = np.clip(t, 0, 1)
    out = np.empty(t.shape + (3,), np.float32)
    for c in range(3):
        out[..., c] = np.interp(t, xs, S[:, c])
    return out


def _line_kernel(length, angle_deg):
    L = int(max(1, round(length)))
    if L < 2:
        return None
    size = L | 1
    k = np.zeros((size, size), np.float32)
    c = size // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * (L - 1) / 2, -np.sin(a) * (L - 1) / 2
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1.0, 1,
             lineType=cv2.LINE_AA)
    return k / k.sum()


def _variable_blur(t, coc, max_sigma, streak_angle, streak_length, levels=8):
    """t: HxW. coc 0..1 — на скільки розмити кожен піксель. Пірамідка рівнів + лінійна інтерполяція.
    Кожен рівень = гаус(sigma_i) + лінійне розмиття довжини streak_length*i/K під streak_angle."""
    if max_sigma <= 0 and streak_length <= 0:
        return t
    stack = []
    for i in range(levels):
        f = i / (levels - 1)
        s = max_sigma * f ** 1.3
        lvl = cv2.GaussianBlur(t, (0, 0), s) if s > 0.3 else t
        k = _line_kernel(streak_length * f, streak_angle)
        if k is not None:
            lvl = cv2.filter2D(lvl, -1, k, borderType=cv2.BORDER_REFLECT)
        stack.append(lvl)
    pos = np.clip(coc, 0, 1) * (levels - 1)
    i0 = np.floor(pos).astype(np.int32)
    i0 = np.clip(i0, 0, levels - 2)
    w = (pos - i0).astype(np.float32)
    st = np.stack(stack, 0)
    yy, xx = np.mgrid[0:t.shape[0], 0:t.shape[1]]
    a = st[i0, yy, xx]
    b = st[i0 + 1, yy, xx]
    return a * (1 - w) + b * w


def _chroma_ab(img, px):
    if abs(px) < 1e-3:
        return img
    H, W = img.shape[:2]
    cx, cy = (W - 1) / 2, (H - 1) / 2
    k = px / np.hypot(cx, cy)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    out = img.copy()
    for ch, sgn in ((0, 1.0), (2, -1.0)):
        s = 1.0 + sgn * k
        mx = (cx + (xx - cx) / s).astype(np.float32)
        my = (cy + (yy - cy) / s).astype(np.float32)
        out[..., ch] = cv2.remap(img[..., ch], mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return out


def film_grain(img, amount=27.0, size=54.0, roughness=35.0, seed=7, color=0.25, ref_width=2848):
    """Зерно в дусі Camera Raw (Effects > Grain), шкали 0..100 як у Camera Raw.
    amount    — сила (std шуму ~ amount/100*0.22 у display-просторі).
    size      — масштаб зерна: клітинка шуму ~ 0.8+size/100*2.4 px на ref_width (2848) і
                масштабується з роздільністю, тож вигляд однаковий на half-res і на повному.
    roughness — нерівномірність: 0 = м'яке гладке, 100 = контрастне, з дрібною октавою.
    Ваги по тонах: сильніше в півтонах, слабше в чорних/світлах. Основа = яскравість,
    + легка кольорова складова (color)."""
    img = np.clip(img, 0, 1).astype(np.float32)
    if amount <= 0:
        return img
    H, W = img.shape[:2]
    rng = np.random.default_rng(seed)
    cell = (0.8 + size / 100.0 * 2.4) * (min(H, W) / ref_width)   # коротка сторона (портрет = W, як було)
    cell = max(cell, 0.6)
    # 24.10: на превʼю (довга сторона < 2144) клітинка за масштабом стає суб-піксельною (480 px: 0.09..0.36 px) і
    # clamp 0.6 робив grain_size безсилим. Тож на малих кадрах підлога видимості: 1.0..2.4 px по size,
    # з плавним зникненням: вага 1 при довгій стороні <= 720, 0 при >= 2144 (фінал рахується рівно як було).
    vis_w = float(np.clip((2144.0 - max(H, W)) / (2144.0 - 720.0), 0.0, 1.0))
    if vis_w > 0:
        cell = max(cell, vis_w * (1.0 + size / 100.0 * 1.4))

    def layer(c, ch=None):
        h, w = max(2, int(round(H / c))), max(2, int(round(W / c)))
        shp = (h, w) if ch is None else (h, w, ch)
        n = rng.standard_normal(shp).astype(np.float32)
        return cv2.resize(n, (W, H), interpolation=cv2.INTER_CUBIC)

    r = np.clip(roughness / 100.0, 0, 1)
    base = layer(cell)                       # м'яке зерно
    if r > 0:
        fine = layer(cell * 0.5)             # дрібна октава
        n = (1 - r) * base + r * (0.55 * base + 0.85 * fine)
    else:                                    # roughness 0: дрібна октава не потрібна (rng лишаємо в тому ж стані)
        rng.standard_normal((max(2, int(round(H / (cell * 0.5)))), max(2, int(round(W / (cell * 0.5))))))
        n = (1 - r) * base
    n = n / (n.std() + 1e-6)
    if r > 0:                                 # roughness = ще й контраст (гострі піки)
        n = np.sign(n) * np.abs(n) ** (1.0 - 0.25 * r)
        n = n / (n.std() + 1e-6)
    y = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    wgt = 0.22 + 0.78 * np.clip(4 * y * (1 - y), 0, 1) ** 0.8   # півтони
    k = amount / 100.0 * 0.22
    if color > 0:
        chroma = layer(cell, 3)
        chroma = chroma / (chroma.std() + 1e-6)
        noise = (n[..., None] * (1 - color) + chroma * color * 0.8) * k
    else:                                    # color 0: кольорова складова не потрібна (rng далі не читається)
        noise = n[..., None] * k
    return np.clip(img + noise * wgt[..., None], 0, 1).astype(np.float32)


# ---------------------------------------------------------------- main
def depth_look(img, depth, *,
               gradient_stops=None,
               color_source="luma",      # 'luma' | 'depth' | 'mix'
               mix=0.5,                  # 0 = luma, 1 = depth (для 'mix')
               tone_black=0.05, tone_white=0.85, tone_gamma=1.25,   # яскравість фото -> t, глибокі чорні
               depth_invert=False,       # True: далеке світліше
               black_by_depth=0.5,       # >0 чорнить ДАЛЕКЕ, <0 чорнить БЛИЗЬКЕ, 0 вимк.
               black_curve=1.6,
               focus_plane=0.82, focus_range=0.04, focus_falloff=0.2,
               blur_strength=90.0,       # макс. sigma розмиття, px на робочій роздільності
               streak_angle=0.0, streak_length=220.0,   # градуси / px
               streak_subject=0.35,       # 0..1 шлейф і на різкому предметі (ghost: lighten із шлейфом)
               halation_threshold=0.55, halation_radius=30.0,
               halation_tint=(1.0, 0.28, 0.10), halation_strength=1.6,
               bloom=0.18, bloom_radius=60.0,
               chroma_ab=5.0,
               grain_amount=27.0, grain_size=54.0, grain_roughness=35.0, seed=7):  # зерно = camera_raw_like
    """img: HxWx3 float RGB 0..1. depth: HxW float 0..1 (вже очищена clean_depth, світле = близьке).
    Повертає HxWx3 float RGB 0..1. Чиста функція: нічого не читає з диску, нічого не кешує.
    Усі px-параметри (blur_strength, streak_length, halation_radius, bloom_radius, chroma_ab)
    задані для довгої сторони 2144 px і самі масштабуються до фактичної роздільності —
    превʼю на 1024 px виглядає так само, як half-res рендер.
    Глибина = результат prepare_depth(), який рахується ОДИН раз на кадр."""
    sc = max(img.shape[:2]) / 2144.0
    blur_strength *= sc; streak_length *= sc; halation_radius *= sc; bloom_radius *= sc; chroma_ab *= sc
    img = np.clip(img.astype(np.float32), 0, 1)
    d = depth.astype(np.float32)
    if depth_invert:
        d = 1.0 - d
    stops = gradient_stops or DEFAULT_STOPS
    H, W = d.shape

    # --- t від яскравості (з глибокими чорними)
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    t_l = np.clip((luma - tone_black) / max(tone_white - tone_black, 1e-6), 0, 1) ** tone_gamma
    t_l = t_l * t_l * (3 - 2 * t_l) * 0.6 + t_l * 0.4  # м'яка S-крива
    t_d = np.clip(d, 0, 1)
    if color_source == "luma":
        t = t_l
    elif color_source == "depth":
        t = t_d
    else:
        t = (1 - mix) * t_l + mix * t_d

    # --- чорне за глибиною
    if black_by_depth != 0:
        far = 1.0 - d if black_by_depth > 0 else d
        k = _smooth(far, 0.25, 1.0) ** black_curve
        t = t * (1.0 - min(abs(black_by_depth), 1.0) * k)

    # --- розмиття по |depth - focus| + streak
    coc = _smooth(np.abs(d - focus_plane), focus_range, focus_range + focus_falloff)
    tb = _variable_blur(t, coc, blur_strength, streak_angle, streak_length)
    if streak_subject > 0 and streak_length > 0:
        k = _line_kernel(streak_length, streak_angle)
        ghost = cv2.filter2D(tb, -1, k, borderType=cv2.BORDER_REFLECT)
        tb = np.maximum(tb, ghost * streak_subject * 1.7)
    t = np.clip(tb, 0, 1)

    # --- градієнт-мапа
    out = _gradient_map(t, stops)

    # --- хроматична аберація
    out = _chroma_ab(out, chroma_ab)

    # --- halation: кожен канал із власним радіусом (R широкий, B вузький) => червоний обідок
    y = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    m = _smooth(y, halation_threshold, min(halation_threshold + 0.2, 0.999)) * y
    if halation_strength > 0:
        tint = np.array(halation_tint, np.float32)
        glow = np.zeros_like(out)
        for c, rm in ((0, 1.0), (1, 0.55), (2, 0.3)):
            glow[..., c] = cv2.GaussianBlur(m, (0, 0), max(halation_radius * rm, 0.5)) * tint[c]
        glow += 0.0
        out = 1.0 - (1.0 - out) * (1.0 - np.clip(glow * halation_strength * 2.0, 0, 1))
    if bloom > 0:
        bl = cv2.GaussianBlur(np.clip(out - 0.45, 0, 1), (0, 0), bloom_radius)
        out = 1.0 - (1.0 - out) * (1.0 - np.clip(bl * bloom * 2.0, 0, 1))

    # --- зерно (Camera Raw-подібне, шкали 0..100)
    out = film_grain(out, grain_amount, grain_size, grain_roughness, seed)
    return np.clip(out, 0, 1).astype(np.float32)


# =====================================================================================
# depth_look_ref — «інвертований контраст» за розбором рефа IMG_4672 (ітерація 2, 23.10)
# 3 плани: світлий розфокусований КОЛЬОРОВИЙ ФОН у смугах / темний силует предмета /
# тонкі яскраві спекуляри з halation і каймою; темний шлейф предмета.
# =====================================================================================
REF_STOPS = [  # (t, rgb) — з рефа: чорний, коричнево-червоний, мовно-фіолетовий, лаванда, оранж, персик
    (0.00, (0.010, 0.005, 0.012)),
    (0.15, (0.26, 0.085, 0.075)),
    (0.30, (0.50, 0.34, 0.56)),
    (0.55, (0.62, 0.62, 0.91)),
    (0.78, (0.93, 0.62, 0.30)),
    (1.00, (1.00, 0.80, 0.55)),
]
# вертикальний профіль t(y) фону: чорне -> коричнево-оранж -> фіолет -> лаванда -> оранж -> персик -> темно-червоне внизу
REF_ENVELOPE = [(0.0, 0.0), (0.26, 0.02), (0.31, 0.15), (0.38, 0.30), (0.55, 0.55), (0.68, 0.58),
                (0.76, 0.78), (0.84, 0.85), (0.92, 0.35), (1.0, 0.15)]


def _gradient_map_pos(t, stops):
    xs = np.array([s[0] for s in stops], np.float32)
    S = np.array([_hex(s[1]) for s in stops], np.float32)
    t = np.clip(t, 0, 1)
    return np.stack([np.interp(t, xs, S[:, c]) for c in range(3)], -1).astype(np.float32)


def _line_kernel_oneside(length, angle_deg):
    """Однобічний лінійний ядро (від центру в бік кута) — для шлейфу (drag)."""
    L = int(max(2, round(length)))
    size = 2 * L + 1
    k = np.zeros((size, size), np.float32)
    a = np.deg2rad(angle_deg)
    # filter2D = кореляція, тож зсув «назад»: ставимо лінію у протилежний бік
    ex, ey = int(round(L - np.cos(a) * L)), int(round(L + np.sin(a) * L))
    cv2.line(k, (L, L), (ex, ey), 1.0, 1, lineType=cv2.LINE_AA)
    return k / k.sum()


def _norm_blur(x, w, sigma):
    num = cv2.GaussianBlur(x * w, (0, 0), sigma)
    den = cv2.GaussianBlur(w, (0, 0), sigma)
    return num / np.maximum(den, 1e-4), den


def depth_look_ref(img, depth, *,
                   # маска предмета з чистої глибини
                   subject_lo=0.62, subject_hi=0.76, mask_soft=0.004, mask_ybottom=(0.72, 0.92, 0.1),
                   # фон
                   bg_blur=0.09, depth_gamma=0.6, band_collapse=0.9, band_x_sigma=0.45,
                   env_strength=0.92, bg_gamma=0.8, bg_lift=0.12,
                   gradient_stops=None, envelope=None,
                   # предмет
                   subject_dark=0.28, subject_soft=0.007, shadow_color=(0.16, 0.10, 0.10),
                   spec_percentile=98.0, spec_color=(1.0, 0.86, 0.86), spec_gain=1.0,
                   # шлейф
                   trail_angle=255.0, trail_length=0.20, trail_strength=1.0,
                   trail_color=(0.12, 0.05, 0.04),
                   # світіння / аберація / зерно
                   halation_radius=0.012, halation_strength=2.2, halation_tint=(1.0, 0.30, 0.12),
                   bloom=0.12, bloom_radius=0.04, chroma_ab=7.0,
                   grain_amount=27.0, grain_size=54.0, grain_roughness=35.0, seed=7):
    """img HxWx3 float RGB, depth HxW float (prepare_depth, світле=близьке). Розміри в долях висоти кадру,
    тож результат не залежить від роздільності. Чиста функція."""
    img = np.clip(img.astype(np.float32), 0, 1)
    d = np.clip(depth.astype(np.float32), 0, 1)
    H, W = d.shape
    S = float(H)
    stops = gradient_stops or REF_STOPS
    env = envelope or REF_ENVELOPE

    # --- маска предмета
    m = _smooth(d, subject_lo, subject_hi)
    m = cv2.GaussianBlur(m, (0, 0), max(mask_soft * S, 0.5))
    if mask_ybottom:   # низ кадру віддаємо фону (персикова смуга), предмет «розчиняється» вниз
        y0, y1, keep = mask_ybottom
        yy_ = np.linspace(0, 1, H, dtype=np.float32)[:, None]
        m = m * (1 - (1 - keep) * _smooth(yy_, y0, y1))
    luma = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # --- ФОН: заповнити область предмета сусідами (нормалізований blur), потім змінний blur за глибиною
    w = 1.0 - m
    fill1, den1 = _norm_blur(luma, w, 0.03 * S)
    fill2, _ = _norm_blur(luma, w, 0.12 * S)
    a = _smooth(den1, 0.02, 0.25)
    fill = fill1 * a + fill2 * (1 - a)
    L_bg = luma * w + fill * m                                 # різка сцена без предмета
    coc = (1.0 - d) ** depth_gamma                             # безперервно від глибини
    bgL = _variable_blur(L_bg.astype(np.float32), coc, bg_blur * S, 0, 0, levels=8)
    # вертикальний колапс -> смуги (x-blur з великим sigma)
    if band_collapse > 0:
        rows = cv2.GaussianBlur(bgL, (0, 0), sigmaX=band_x_sigma * W, sigmaY=0.014 * S)
        bgL = bgL * (1 - band_collapse) + rows * band_collapse
    lo, hi = np.percentile(bgL, 3), np.percentile(bgL, 97)
    t_bg = np.clip((bgL - lo) / max(hi - lo, 1e-6), 0, 1) ** bg_gamma
    t_bg = t_bg * (1 - bg_lift) + bg_lift * 0.5
    ys = np.linspace(0, 1, H, dtype=np.float32)
    prof = np.interp(ys, [e[0] for e in env], [e[1] for e in env]).astype(np.float32)[:, None]
    t = (1 - env_strength) * t_bg + env_strength * (prof + (t_bg - t_bg.mean()) * 0.2)
    bg = _gradient_map_pos(t, stops)

    # --- ШЛЕЙФ: однобічний directional blur маски предмета -> темне на фон
    if trail_strength > 0 and trail_length > 0:
        k = _line_kernel_oneside(trail_length * S, trail_angle)
        tr = cv2.filter2D(cv2.GaussianBlur(m, (0, 0), 0.006 * S), -1, k, borderType=cv2.BORDER_CONSTANT)
        tr = np.clip(tr * (1 - m) * 1.2, 0, 1) * trail_strength
        bg = bg * (1 - tr[..., None]) + np.array(trail_color, np.float32) * tr[..., None]

    # --- ПРЕДМЕТ: силует, холодно-коричневий; лише спекуляри лишаються
    Ls = cv2.GaussianBlur(luma, (0, 0), max(subject_soft * S, 0.5))
    pm = np.percentile(Ls[m > 0.5], 99) if (m > 0.5).any() else 1.0
    ln = np.clip(Ls / max(pm, 1e-4), 0, 1) ** 1.6
    sh = np.array(shadow_color, np.float32)
    subj = sh * (0.55 + ln[..., None] * subject_dark * 3.0 * (1 - 0.0))
    out = bg * (1 - m[..., None]) + subj * m[..., None]

    # спекуляри: верхні N% яскравості в межах предмета + high-pass
    hp = luma - cv2.GaussianBlur(luma, (0, 0), 0.01 * S)
    score = np.maximum(hp, 0) + 0.25 * luma
    inm = m > 0.5
    thr = np.percentile(score[inm], spec_percentile) if inm.any() else 1.0
    sp = _smooth(score, thr, thr + 0.06) * np.clip(m * 1.3, 0, 1)
    spec = sp[..., None] * np.array(spec_color, np.float32) * spec_gain
    spec = _chroma_ab(spec, chroma_ab * S / 2144.0)           # кольорова кайма саме на спекулярах
    out = np.clip(out + spec, 0, 1)

    # halation: різний радіус по каналах (R широкий) + легкий bloom
    if halation_strength > 0:
        ym = cv2.cvtColor(np.clip(spec, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        tint = np.array(halation_tint, np.float32)
        glow = np.zeros_like(out)
        for c, rm in ((0, 1.0), (1, 0.55), (2, 0.3)):
            glow[..., c] = cv2.GaussianBlur(ym, (0, 0), max(halation_radius * S * rm, 0.5)) * tint[c]
        out = 1 - (1 - out) * (1 - np.clip(glow * halation_strength, 0, 1))
    if bloom > 0:
        bl = cv2.GaussianBlur(np.clip(out - 0.5, 0, 1), (0, 0), bloom_radius * S)
        out = 1 - (1 - out) * (1 - np.clip(bl * bloom * 2, 0, 1))

    out = film_grain(out, grain_amount, grain_size, grain_roughness, seed)
    return np.clip(out, 0, 1).astype(np.float32)


# =====================================================================================
# depth_look_lens — ітерація 3 (23.10): лінзовий блюр по глибині + райдужний градієнт далекого
# + «предмет тягнеться в глибину». Дві стадії:
#   prepare_lens(img, depth, focus, ...)  -> cache (шари блюру, залежать лише від глибини/focus/r)
#   depth_look_lens(img, depth, cache=..., **кольорові повзунки)  -> швидко
# =====================================================================================
LENS_STOPS = [  # t: 0 = ближнє/у фокусі ... 1 = найдальше (темний -> бордо -> оранж -> персик -> лаванда -> блакитний)
    (0.00, (0.04, 0.015, 0.03)),
    (0.20, (0.36, 0.08, 0.11)),
    (0.42, (0.95, 0.50, 0.16)),
    (0.62, (1.00, 0.75, 0.53)),
    (0.82, (0.67, 0.62, 0.93)),
    (1.00, (0.52, 0.72, 0.98)),
]


def _disk_kernel(r):
    R = int(np.ceil(r)) + 1
    yy, xx = np.mgrid[-R:R + 1, -R:R + 1].astype(np.float32)
    k = np.clip(r + 0.5 - np.hypot(xx, yy), 0, 1)
    return k / k.sum()


def _disk_blur(x, r):
    """Блюр диском радіуса r px (float32 HxW). Для великих r — на зменшеній копії (швидко)."""
    if r < 0.6:
        return x
    H, W = x.shape
    if r > 18:
        f = r / 18.0
        sm = cv2.resize(x, (max(2, int(W / f)), max(2, int(H / f))), interpolation=cv2.INTER_AREA)
        sm = cv2.filter2D(sm, -1, _disk_kernel(18.0), borderType=cv2.BORDER_REFLECT)
        return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.filter2D(x, -1, _disk_kernel(r), borderType=cv2.BORDER_REFLECT)


def _focus_from_target(d, blur_target):
    """blur_target 0..1 -> площина фокуса за перцентилями глибини кадру: 1 = герой/близьке різке, 0 = різке тло."""
    p5, p95 = np.percentile(d, [5, 95])
    return float(p5 + float(np.clip(blur_target, 0, 1)) * (p95 - p5))


def prepare_lens(img, depth, *, focus=0.85, n_layers=10, r_min=0.0025, r_max=0.04, lens_gamma=0.8,
                 dispersion=0.15, blur_amount=1.0, blur_target=-1.0):
    """Шари глибини (м'які трикутні ваги, сума=1) -> диск-блюр premult RGBA з дисперсією каналів
    (R: r·(1+disp), B: r·(1−disp)). Залежить лише від img/depth/focus/r — кеш для повзунків кольору.
    blur_amount множить r_min і r_max; blur_target >= 0 перекриває focus (див. _focus_from_target).
    Радіуси - частки довгої сторони кадру (портрет: = H, як було)."""
    img = np.clip(img.astype(np.float32), 0, 1)
    d = np.clip(depth.astype(np.float32), 0, 1)
    H, W = d.shape
    HL = max(H, W)
    r_min = r_min * blur_amount
    r_max = r_max * blur_amount
    if blur_target is not None and blur_target >= 0:
        focus = _focus_from_target(d, blur_target)
    cs = np.linspace(0, 1, n_layers, dtype=np.float32)
    step = float(cs[1] - cs[0])
    layers = []
    for c in cs:                                      # від далекого (c=0) до ближнього (c=1)
        w = np.clip(1.0 - np.abs(d - c) / step, 0, 1).astype(np.float32)
        if w.max() < 1e-3:
            layers.append(None); continue
        r = (r_min + r_max * abs(float(c) - focus) ** lens_gamma) * HL
        a = _disk_blur(w, r)
        rgb = np.empty((H, W, 3), np.float32)
        for ch, mul in ((0, 1 + dispersion), (1, 1.0), (2, 1 - dispersion)):
            rgb[..., ch] = _disk_blur(img[..., ch] * w, r * mul)
        layers.append(dict(c=float(c), r=float(r), rgb=rgb.astype(np.float16), a=a.astype(np.float16)))
    return dict(layers=layers, focus=focus, r_max_px=(r_min + r_max) * HL, H=H, W=W, step=step)


def _stretch_copies(img, m, N, length, vp, shift, blur, opacity, fade, shift_col, stops, focus_t0=0.0):
    """Список (premult RGBA, k) для копій предмета, від найдальшої до найближчої."""
    H, W = m.shape
    lum = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    rgba = np.dstack([img * m[..., None], m]).astype(np.float32)
    vx, vy = vp[0] * W, vp[1] * H
    out = []
    for k in range(N, 0, -1):
        u = k / N
        s = 1.0 - length * u
        M = np.array([[s, 0, vx * (1 - s) + shift[0] * W * u], [0, s, vy * (1 - s) + shift[1] * H * u]], np.float32)
        c = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        sg = blur * max(H, W) * (0.25 + u)
        if sg > 0.4:
            c = _gblur(c, sg)          # великі sigma - на зменшеній копії (було 2/3 часу рендера)
        a = np.clip(c[..., 3], 1e-4, 1)
        col = c[..., :3] / a[..., None]
        g0 = _gradient_map_pos(np.full((1, 1), focus_t0 + u * (1 - focus_t0), np.float32), stops)[0, 0]
        lm = cv2.cvtColor(np.clip(col, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
        g = g0[None, None, :] * (0.55 + 0.9 * np.clip(lm * 2.0, 0, 1))[..., None]
        sc = float(np.clip(shift_col * (0.25 + 0.75 * u), 0, 1))
        col = col * (1 - sc) + g * sc
        f = opacity * (1 - u) ** fade
        a2 = a * f
        out.append((np.dstack([col * a2[..., None], a2]).astype(np.float32), k))
    return out


def _screen(a, b):
    return 1 - (1 - a) * (1 - np.clip(b, 0, 1))


def _silhouette(out, d, stops, amt, lo, hi, bg_amt, bg_lo, detail, rim, rim_r):
    """Глибина як ножиці: близьке -> майже чорний силует (тон = нижній стоп палітри), далеке -> світле
    градієнтне тло (верхня частина палітри, своя яскравість лишається смугами). rim = сяйво тла,
    що «заливає» контур силуету (light wrap)."""
    H = max(d.shape)
    ms = cv2.GaussianBlur(_smooth(d, lo, hi), (0, 0), max(0.002 * H, 0.5))
    bright = None
    if bg_amt > 0 or rim > 0:                                    # світле тло потрібне і тлу, і сяйву по контуру
        y = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        yb = cv2.GaussianBlur(y, (0, 0), max(0.01 * H, 0.5))
        bgm = ms < 0.5
        if bgm.any():
            p0, p1 = np.percentile(yb[bgm], 2), np.percentile(yb[bgm], 98)
        else:
            p0, p1 = 0.0, 1.0
        tb = np.clip((yb - p0) / max(p1 - p0, 1e-4), 0, 1)
        tb = bg_lo + (1 - bg_lo) * (0.7 * tb + 0.3 * (1 - d))    # далі = світліше
        bright = _gradient_map_pos(tb.astype(np.float32), stops)
    if bg_amt > 0:
        wb = (amt * bg_amt * (1 - ms))[..., None]
        out = out * (1 - wb) + bright * wb
    dark = _gradient_map_pos(np.zeros((1, 1), np.float32), stops)[0, 0] * 0.35
    subj = dark[None, None, :] + out * detail
    ws = (amt * ms)[..., None]
    out = out * (1 - ws) + subj * ws
    if rim > 0:
        src = cv2.GaussianBlur(bright * (1 - ms[..., None]), (0, 0), max(rim_r * H, 0.5))
        edge = np.clip(ms * (1 - cv2.GaussianBlur(ms, (0, 0), max(rim_r * H, 0.5))) * 4, 0, 1)
        out = _screen(out, src * (edge * rim * amt)[..., None] * 1.6)
    return np.clip(out, 0, 1).astype(np.float32)


def _drift_noise(shape, time, scale_px=None):
    """Повільний низькочастотний шум -1..1 (3 синусоїди, зсуваються з часом), рахується на малій сітці."""
    H, W = shape
    HL = max(H, W)
    f = 64.0 / HL
    h, w = max(2, int(round(H * f))), max(2, int(round(W * f)))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x /= 64.0; y /= 64.0
    tau = 2 * np.pi
    n = (np.sin(tau * (1.3 * x + 0.7 * y + 0.08 * time))
         + np.sin(tau * (-0.6 * x + 1.9 * y + 0.05 * time + 0.3))
         + np.sin(tau * (0.9 * x - 1.1 * y + 0.11 * time + 0.6))) / 3.0
    return cv2.resize(n.astype(np.float32), (W, H), interpolation=cv2.INTER_CUBIC)


def _haze(out, d, stops, amt, start, curve, t_col, scatter, drift=0.0, time=0.0):
    """Серпанок за глибиною: густина = smooth(1-d) ** curve; колір = стоп палітри t_col,
    розсіяний із самої сцени (scatter: 0 = рівний туман, 1 = туман бере колір сусідніх світел)."""
    H = max(d.shape)
    dens = amt * _smooth(1 - d, start, 1.0) ** curve
    if drift > 0:                     # густина «дихає»: повільний шум зсувається з часом
        dens = dens * np.clip(1.0 + drift * _drift_noise(d.shape, time), 0, 3)
    col = _gradient_map_pos(np.full((1, 1), t_col, np.float32), stops)[0, 0][None, None, :]
    if scatter > 0:
        sc = cv2.GaussianBlur(out, (0, 0), max(0.06 * H, 1.0))
        col = col * (1 - scatter) + _screen(sc, col * 0.6) * scatter
    w = np.clip(dens, 0, 1)[..., None]
    return np.clip(out * (1 - w) + col * w, 0, 1).astype(np.float32)


def _light_trail(out, d, stops, amt, angle, length, source, thr, decay, tint, s_lo, s_hi, N=None):
    """Напрямний шлейф (не Gaussian), вектор angle (0° = вправо, 90° = вгору), довжина length·H.
    Два роди, source змішує:
      source 1 = «тягне в світло» (HF7): копії самої людини (за глибиною) лягають ПІД неї вздовж вектора,
                 прозорість exp(-decay·u), колір хвоста ковзає по палітрі (tint) — видно і на світлому тлі;
      source 0 = світло-смуги (HF2/HF84): тягнуться лише відблиски (локально яскраве, не рівне небо), screen.
    Рахується на 1/2 роздільності. Для відео angle потім береться з руху."""
    H, W = d.shape
    f = 2 if max(H, W) > 500 else 1
    h, w = max(2, H // f), max(2, W // f)
    hsz = max(h, w)
    o = cv2.resize(out, (w, h), interpolation=cv2.INTER_AREA)
    ds = cv2.resize(d, (w, h), interpolation=cv2.INTER_AREA)
    ms = _smooth(ds, s_lo, s_hi)
    y = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
    hp = y - cv2.GaussianBlur(y, (0, 0), max(0.03 * hsz, 1.0))
    hl = _smooth(hp, 0.02, 0.12) * _smooth(y, thr - 0.25, thr)
    L = length * hsz
    N = N or int(np.clip(L / 1.5, 16, 120))   # крок копій <= 1.5 px, інакше сходинки
    a = np.deg2rad(angle)
    dx, dy = np.cos(a), -np.sin(a)

    def shift(x, u):
        M = np.float32([[1, 0, dx * L * u], [0, 1, dy * L * u]])
        return cv2.warpAffine(x, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    def tinted(c, u):
        if tint <= 0:
            return c
        g = _gradient_map_pos(np.full((1, 1), 0.3 + 0.7 * u, np.float32), stops)[0, 0]
        lum = cv2.cvtColor(np.clip(c, 0, 1), cv2.COLOR_RGB2GRAY)[..., None]
        return c * (1 - tint) + np.clip(g[None, None, :] * (0.4 + 0.9 * lum), 0, 1) * tint

    res = o.copy()
    if source > 0:                      # копії людини: від найдальшої до найближчої, «над» тлом
        prem = o * ms[..., None]
        for k in range(N, 0, -1):
            u = k / N
            sa = shift(ms, u)
            col = shift(prem, u) / np.maximum(sa[..., None], 1e-4)
            sg = 0.004 * hsz * (0.3 + 2.5 * u)
            if sg > 0.5:
                col = cv2.GaussianBlur(col, (0, 0), sg); sa = cv2.GaussianBlur(sa, (0, 0), sg)
            al = np.clip(sa * float(np.exp(-decay * u)) * amt * source * (28.0 / N), 0, 1)[..., None]
            res = res * (1 - al) + tinted(col, u) * al
        res = o * ms[..., None] + res * (1 - ms[..., None])     # сама людина зверху, різка
    if source < 1:                      # світло-смуги від відблисків
        src = o * hl[..., None]
        acc, ws = np.zeros_like(o), 0.0
        for k in range(1, N + 1):
            u = k / N
            wk = float(np.exp(-decay * u))
            acc += tinted(shift(src, u), u) * wk
            ws += wk
        acc = cv2.GaussianBlur(acc / ws, (0, 0), 1.0)
        res = _screen(res, acc * amt * (1 - source) * 4.0)
    if f > 1:                           # повертаємо різкість: до повної лише різницю
        res = out + cv2.resize(res - o, (W, H), interpolation=cv2.INTER_LINEAR)
    return np.clip(res, 0, 1).astype(np.float32)



# ---------------------------------------------------------------- глоу / іскри / анімація
def _gblur(x, sigma):
    """Гаус; для великих sigma - на зменшеній копії (швидко, результат майже той самий)."""
    if sigma < 0.5:
        return x
    if sigma > 10:
        f = sigma / 5.0
        H, W = x.shape[:2]
        sm = cv2.resize(x, (max(2, int(round(W / f))), max(2, int(round(H / f)))), interpolation=cv2.INTER_AREA)
        sm = cv2.GaussianBlur(sm, (0, 0), 5.0)
        return cv2.resize(sm, (W, H), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(x, (0, 0), sigma)


def _glow(out, amt, radius_px, thr, warmth, stops):
    """М'який bloom яскравого: 3 гауси (r, 2r, 4r) суми, screen. warmth: 0 = колір самого джерела,
    1 = колір палітри на t = 0.5 + 0.5*warmth (світло з тієї ж гами, що й кадр)."""
    y = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
    m = _smooth(y, thr, min(thr + 0.25, 0.999))
    src = out * m[..., None]
    bl = (_gblur(src, radius_px) + _gblur(src, 2 * radius_px) + _gblur(src, 4 * radius_px)) * (1.0 / 3.0)
    if warmth > 0:
        pal = _gradient_map_pos(np.full((1, 1), 0.5 + 0.5 * warmth, np.float32), stops)[0, 0]
        by = cv2.cvtColor(bl, cv2.COLOR_RGB2GRAY)[..., None]
        bl = bl * (1 - warmth) + by * np.clip(pal / max(float(pal.max()), 1e-3), 0, 1)[None, None, :] * warmth
    return _screen(out, np.clip(bl * amt * 1.0, 0, 1))


def _ray_mask(length, angle_deg):
    L = int(max(3, round(length))) | 1
    k = np.zeros((L, L), np.uint8)
    c = L // 2
    a = np.deg2rad(angle_deg)
    dx, dy = np.cos(a) * c, -np.sin(a) * c
    cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
    return k


def _sparks(out, amt, thr, size, rays, angle, length):
    """Точковий глоу: дрібні яскраві точки (top-hat по яскравості + поріг) -> круглий блиск
    (+ промені зірки rays = 1..8, лінійні max-фільтри зі спадом) -> screen. Рахується на <= 960 px."""
    H, W = out.shape[:2]
    HL = max(H, W)
    f = min(1.0, 960.0 / HL)
    o = cv2.resize(out, (max(2, int(round(W * f))), max(2, int(round(H * f)))), interpolation=cv2.INTER_AREA) if f < 1 else out
    h, w = o.shape[:2]
    hl = max(h, w)
    y = cv2.cvtColor(o, cv2.COLOR_RGB2GRAY)
    D = int(max(3, round(size * hl * 3.0))) | 1
    th = y - cv2.morphologyEx(y, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (D, D)))
    m = _smooth(y, thr, min(thr + 0.1, 0.9999)) * _smooth(th, 0.05, 0.2)
    if not m.any():
        return out
    sp = o * m[..., None]
    sig = max(size * hl, 0.8)
    kd = int(max(1, round(sig))) * 2 + 1
    core = cv2.dilate(sp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kd, kd)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
    glint = cv2.GaussianBlur(core, (0, 0), sig * 0.8)
    res = glint
    n = int(rays)
    if n > 0 and length > 0:
        Lr = length * hl
        seed_ = cv2.dilate(sp, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), borderType=cv2.BORDER_CONSTANT, borderValue=0)
        ang = [angle + i * 360.0 / n for i in range(n)] if n % 2 else [angle + i * 360.0 / n for i in range(n // 2)]
        acc = np.zeros_like(o)
        J = 6                                   # J ступенів довжини -> лінійний спад променя
        for a in ang:
            line = np.zeros_like(o)
            for j in range(1, J + 1):
                Lj = Lr * j / J
                if n % 2:                       # непарне n: промінь в один бік (півлінія)
                    k = _ray_mask(2 * Lj, a)
                    c = k.shape[0] // 2
                    ax, ay = np.cos(np.deg2rad(a)), -np.sin(np.deg2rad(a))
                    yy, xx = np.mgrid[0:k.shape[0], 0:k.shape[1]]
                    k = k * (((xx - c) * ax + (yy - c) * ay) <= 0.5)   # dilate віддзеркалює ядро
                else:
                    k = _ray_mask(2 * Lj, a)
                line += cv2.dilate(seed_, np.ascontiguousarray(k, np.uint8), borderType=cv2.BORDER_CONSTANT, borderValue=0)
            acc = np.maximum(acc, line / J)
        acc = cv2.GaussianBlur(acc, (0, 0), 0.7)
        res = np.maximum(glint, np.clip(acc * 0.9, 0, 1))
    res = np.clip(res * amt, 0, 1)
    if f < 1:
        res = cv2.resize(res, (W, H), interpolation=cv2.INTER_LINEAR)
    return _screen(out, res)


def _warp_fields(seed, gs, ww, hh, phase):
    """Поля зсуву dx, dy (одиниця амплітуди). phase=0: як було (сід = seed). phase>0: плавний перехід
    між полями сідів seed+k і seed+k+1 (smoothstep), енергія тримається сталою - хвиля «дихає»."""
    def one(sd):
        rs = np.random.default_rng(int(sd))
        a = cv2.resize(rs.standard_normal((gs, gs)).astype(np.float32), (ww, hh), interpolation=cv2.INTER_CUBIC)
        b = cv2.resize(rs.standard_normal((gs, gs)).astype(np.float32), (ww, hh), interpolation=cv2.INTER_CUBIC)
        return a, b
    k = int(np.floor(phase))
    fr = phase - k
    ax, ay = one(seed + k)
    if fr <= 0:
        return ax, ay
    f = fr * fr * (3 - 2 * fr)
    bx, by = one(seed + k + 1)
    nrm = 1.0 / np.sqrt((1 - f) ** 2 + f ** 2)
    return (((1 - f) * ax + f * bx) * nrm).astype(np.float32), (((1 - f) * ay + f * by) * nrm).astype(np.float32)


# ---------------------------------------------------------------- анімація градієнта (режими grad_mode)
GRAD_MODES = ("static", "camera", "drift", "shimmer", "depth-walk", "wave")


def _pingpong(x):
    """Дзеркальна (ping-pong) циклічність з періодом 2: x у [0,1] лишається x, далі відбивається - без жорсткого шва."""
    return 1.0 - np.abs(1.0 - np.mod(x, 2.0))


def _grad_cycles(speed, loop, phase, time):
    """Фаза в циклах = speed*time + phase. loop>0: швидкість квантується так, щоб за loop секунд вкластось ціле число циклів."""
    speed = float(speed)
    if loop and loop > 0 and speed != 0.0:
        n = int(round(speed * loop))
        if n == 0:
            n = 1 if speed > 0 else -1
        speed = n / float(loop)
    return speed * float(time) + float(phase)


def _lut_apply(t, lut):
    n = lut.shape[0]
    return lut[np.clip((np.asarray(t) * (n - 1) + 0.5).astype(np.int32), 0, n - 1)]


def camera_offsets(frames_rgb, depths=None, far_thr=0.4, work_side=720, normalize=False):
    """Кумулятивний рух фону камери відносно кадру 0: [(dx, dy), ...] (len = len(frames)).
    Одиниці: пікселі вхідних кадрів (normalize=True -> частки довгої сторони = grad_offset_x/y).
    Знак: додатний dx/dy = вміст фону зсунувся вправо/вниз відносно кадру 0 (так і треба подавати в grad_offset_x/y).
    Метод: cv2.phaseCorrelate між сусідніми кадрами на сірому зменшеному зображенні, лише по далеких пікселях (depth < far_thr);
    якщо далеких мало - по всьому кадру. Цей рух сумується."""
    import cv2
    n = len(frames_rgb)
    H, W = frames_rgb[0].shape[:2]
    HLn = max(H, W)
    f = min(1.0, work_side / float(HLn))
    sz = (max(8, int(round(W * f))), max(8, int(round(H * f))))
    g, mk = [], []
    for i in range(n):
        im = np.asarray(frames_rgb[i], np.float32)
        if im.max() > 1.5:
            im = im / 255.0
        gi = cv2.resize(cv2.cvtColor(np.clip(im, 0, 1), cv2.COLOR_RGB2GRAY), sz, interpolation=cv2.INTER_AREA)
        if depths is not None:
            d = np.asarray(depths[i] if not isinstance(depths, np.ndarray) or depths.ndim == 3 else depths, np.float32)
            m = (cv2.resize(d, sz, interpolation=cv2.INTER_AREA) < far_thr).astype(np.float32)
        else:
            m = np.ones(gi.shape, np.float32)
        g.append(gi); mk.append(m)
    win = cv2.createHanningWindow((sz[0], sz[1]), cv2.CV_32F)
    out = [(0.0, 0.0)]
    cx = cy = 0.0
    for i in range(1, n):
        m = mk[i - 1] * mk[i]
        if m.mean() < 0.05:                      # далекого майже нема -> весь кадр
            m = np.ones_like(m)
        m = cv2.GaussianBlur(m, (0, 0), 2.0)
        a, b = g[i - 1], g[i]
        # прибираємо середнє по масці, щоб рівний фон не давав піку на (0,0)
        wa = float((a * m).sum() / max(m.sum(), 1e-6)); wb = float((b * m).sum() / max(m.sum(), 1e-6))
        A = ((a - wa) * m).astype(np.float32); B = ((b - wb) * m).astype(np.float32)
        (sx, sy), _resp = cv2.phaseCorrelate(A, B, win)
        cx += sx / f; cy += sy / f
        out.append((cx, cy))
    if normalize:
        out = [(x / HLn, y / HLn) for x, y in out]
    return out



def _near_map(d, layers, rmaxpx, onset):
    """Карта «близькості» 0..1 за глибиною: near_d шару (1 - smooth(r/rmax)) інтерпольований по c шарів
    (трикутні ваги шарів = лінійна інтерполяція). near_d <= 0.05 -> 0, як було в пошаровому warp/smear."""
    cs, nd = [], []
    for L in layers:
        if L is None:
            continue
        v = float(1 - _smooth(L["r"] / rmaxpx, onset, 0.55))
        cs.append(L["c"]); nd.append(v if v > 0.05 else 0.0)
    if not cs or max(nd) <= 0.0:
        return None
    return np.interp(d, np.asarray(cs, np.float32), np.asarray(nd, np.float32)).astype(np.float32)


def _warp_map(d, nmap, warp, warp_scale, seed, phase):
    """ОДНЕ поле зсуву для всього кадру: F = шум * warp * nmap(d) * 0.04*довга сторона (nmap = близькість за глибиною).
    Повертає (map_x, map_y) для cv2.remap: ним ремапляться і шари (rgb/alpha), і глибина d, і img для копій предмета,
    тож усе, що далі рахується від d (маска предмета, силует, серпанок, шлейф, поле градієнта), їде разом із фігурою."""
    H, W = d.shape
    gs = max(2, int(round(1 / max(warp_scale, 0.02))))
    amp = warp * nmap * (max(H, W) * 0.04)
    fx, fy = _warp_fields(int(seed), gs, W, H, phase)
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    return (gx + fx * amp).astype(np.float32), (gy + fy * amp).astype(np.float32)


def _remap(x, mp):
    return cv2.remap(x, mp[0], mp[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def _smear_depth(d, nmap, smear, smear_angle, smear_radial):
    """Глибина слідом за розмазуванням близького (ті ж 12 зсувів і ваги, що в пошаровому smear картинки):
    у хвості шлейфу глибина = глибина предмета, що тягнеться. Рахується по карті близькості, не по шарах."""
    H, W = d.shape
    ln = smear * max(H, W) * 0.15
    ang = np.deg2rad(smear_angle)
    AD = np.dstack([nmap, d * nmap]).astype(np.float32)
    acc, wsum = np.zeros_like(AD), 0.0
    if smear_radial > 0.5:
        ys, xs = np.nonzero(nmap > 0.5)
        cx, cy = (xs.mean(), ys.mean()) if len(xs) else (W / 2, H / 2)
    for k in range(12):
        t = k / 11.0
        if smear_radial > 0.5:
            sc = 1 + (ln / H) * 2.0 * t
            M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        else:
            M = np.float32([[1, 0, np.cos(ang) * ln * t], [0, 1, np.sin(ang) * ln * t]])
        wk = (1 - t) ** 1.5 + 0.05
        acc += cv2.warpAffine(AD, M, (W, H)) * wk
        wsum += wk
    A2 = np.clip(np.maximum(AD[..., 0], acc[..., 0] / wsum * 1.6), 0, 1)
    D2 = np.minimum(np.maximum(AD[..., 1], acc[..., 1] / wsum * 1.6), A2)
    return np.clip(D2 + d * (1 - A2), 0, 1).astype(np.float32)


def depth_look_lens(img, depth, *, cache=None,
                    # лінзовий блюр (ідуть у prepare_lens, якщо cache не передано)
                    focus=0.88, n_layers=10, r_min=0.0025, r_max=0.05, lens_gamma=1.2, dispersion=0.15,
                    blur_amount=1.0,   # множник r_min/r_max (0 = без блюру, 3 = втричі сильніше)
                    blur_target=-1.0,  # -1 = focus як є; 0..1: 1 = герой/близьке різке, 0 = різке тло (focus з перцентилів глибини)
                    # колір далекого
                    gradient_stops=None, far_gradient=0.85, gradient_onset=0.08, near_tint=0.12, luma_keep=0.5, gradient_gain=1.0, subject_map=0.0, subject_contrast=1.0, bg_bands=0.0, band_flip=0.0, warp=0.0, warp_scale=0.08, smear=0.0, smear_angle=0.0, warp_seed=7, smear_radial=0.0,
                    # предмет тягнеться в глибину
                    subject_lo=0.66, subject_hi=0.8,
                    stretch_length=0.45, stretch_steps=18, vanishing_point=(0.5, 0.42), stretch_shift=(0.0, -0.10),
                    stretch_blur=0.012, stretch_opacity=0.9, stretch_fade=1.1, stretch_color_shift=0.9,
                    stretch_lighten=0.0,  # 1 = копії лише освітлюють (лікує темний ореол), 0 = як було
                    # силует на сяйві (Pinterest: HF24/HF54/ФБК): близьке -> чорне, далеке -> світле тло
                    silhouette=0.0, sil_lo=0.66, sil_hi=0.8, sil_bg=1.0, sil_bg_lo=0.55, sil_detail=0.12,
                    sil_rim=0.6, sil_rim_radius=0.012,
                    # серпанок, що густішає з глибиною (далі = світліше, колір з палітри)
                    haze=0.0, haze_start=0.25, haze_curve=1.4, haze_t=0.65, haze_scatter=0.5,
                    # напрямний світло-шлейф (HF7/HF2/HF84): яскраве/предмет тягнеться по вектору
                    trail=0.0, trail_angle=200.0, trail_length=0.3, trail_source=0.5, trail_threshold=0.6,
                    trail_decay=2.0, trail_tint=0.6,
                    # світіння / аберація / зерно
                    halation_threshold=0.62, halation_radius=0.012, halation_strength=0.8,
                    halation_tint=(1.0, 0.30, 0.12), chroma_ab=3.0,
                    # глоу (м'який bloom яскравого) і точковий глоу (іскри на дрібних відблисках)
                    glow=0.0, glow_radius=0.04, glow_threshold=0.6, glow_warmth=0.5,
                    spark=0.0, spark_threshold=0.85, spark_size=0.012, spark_rays=0, spark_angle=30.0, spark_length=0.08,
                    grain_amount=27.0, grain_size=54.0, grain_roughness=35.0, seed=7,
                    # анімація (відео): time у секундах; усі повзунки 0 = статика, time=0 і дефолти = як без анімації
                    time=0.0, grain_animate=0, grain_fps=24.0, warp_speed=0.0, glow_pulse=0.0, pulse_hz=0.5,
                    haze_drift=0.0, trail_wobble=0.0,
                    # анімація градієнта: grad_mode 0 static 1 camera 2 drift 3 shimmer 4 depth-walk 5 wave; grad_amount=0 або mode 0 = як без неї
                    grad_mode=0, grad_speed=0.25, grad_phase=0.0, grad_loop=0.0, grad_amount=1.0,
                    grad_angle=90.0, grad_wavelength=0.5, grad_amp=0.15, grad_offset_x=0.0, grad_offset_y=0.0):
    img = np.clip(img.astype(np.float32), 0, 1)
    d = np.clip(depth.astype(np.float32), 0, 1)
    H, W = d.shape
    HL = max(H, W)                                            # радіуси - частки довгої сторони (портрет: = H)
    time = float(time)
    stops = gradient_stops or LENS_STOPS
    if cache is None:
        cache = prepare_lens(img, d, focus=focus, n_layers=n_layers, r_min=r_min, r_max=r_max,
                             lens_gamma=lens_gamma, dispersion=dispersion, blur_amount=blur_amount,
                             blur_target=blur_target)
    focus = cache["focus"]
    rmaxpx = max(cache["r_max_px"], 1e-6)                     # blur_amount=0 -> радіус 0, без ділення на нуль
    comp = np.zeros((H, W, 4), np.float32)                    # premult RGB + A (над: назад -> вперед)
    # --- деформація (24.10): ОДНЕ поле зсуву на шари, глибину й img. Раніше їхали лише rgb/alpha шарів, а d, маска
    # предмета, силует, серпанок, шлейф і поле градієнта лишались на місці -> грейд і контури «їхали» від фігури.
    # warp = 0 / smear = 0 -> відповідний блок не рахується взагалі.
    nmap = _near_map(d, cache["layers"], rmaxpx, gradient_onset) if (warp > 0 or smear > 0) else None
    wmap = None
    if warp > 0 and nmap is not None:
        wmap = _warp_map(d, nmap, warp, warp_scale, warp_seed, warp_speed * time if warp_speed > 0 else 0.0)
        d = np.clip(_remap(d, wmap), 0, 1)                    # глибина тим самим полем (далі все від неї)
        img = _remap(img, wmap)                               # для копій предмета (stretch)
        nmap = _near_map(d, cache["layers"], rmaxpx, gradient_onset)
    # сила 0 = не рахуємо: копії предмета (steps/opacity), градієнт (усі 4 споживачі) — пропуск усього блоку
    stretch_done = stretch_steps <= 0 or stretch_opacity <= 0
    if not stretch_done:
        m = _smooth(d, subject_lo, subject_hi)
        m = cv2.GaussianBlur(m, (0, 0), 0.003 * HL)
    use_grad = far_gradient > 0 or near_tint > 0 or subject_map > 0 or bg_bands > 0

    # --- анімація градієнта (усе нижче неактивне при grad_mode=0 / grad_amount=0 -> шлях як раніше)
    gmode = int(grad_mode) if (use_grad and int(grad_mode) in (1, 2, 3, 4, 5) and float(grad_amount) > 0) else 0
    gamt = float(np.clip(grad_amount, 0, 1))
    g_lut = g_field4 = g_wave = band_c = None
    g_cyc = 0.0
    if gmode:
        g_cyc = _grad_cycles(grad_speed, grad_loop, grad_phase, time)
        g_lut = _gradient_map_pos(np.linspace(0, 1, 2048, dtype=np.float32), stops)
        g_wl = max(float(grad_wavelength), 1e-3)
        g_ang = np.deg2rad(float(grad_angle))
        g_dyn = float(grad_offset_y) * HL / H              # частки висоти кадру (bands йдуть по y/H)
        yy_ = np.arange(H, dtype=np.float32)[:, None]
        xx_ = np.arange(W, dtype=np.float32)[None, :]
        if gmode == 4:                                       # хвиля по карті глибини: тло = відстань від фокуса
            tp = np.clip(np.abs(focus - d) / max(focus, 1e-3), 0, 1)
            t4 = _pingpong(2.0 * (tp / g_wl - g_cyc))
            t4 = tp * (1 - gamt) + t4 * gamt
            g_field4 = _lut_apply(t4, g_lut)
            g_d4 = (t4 - tp).astype(np.float32)              # зміщення координати смуг тією ж хвилею
        if gmode == 5:                                       # біжить синусоїда, при куті 90 - вздовж смуг
            a5 = g_ang - np.pi / 2
            g_wave = np.sin(2 * np.pi * (((xx_ * np.cos(a5) + yy_ * np.sin(a5)) / HL) / g_wl - g_cyc)).astype(np.float32) * (float(grad_amp) * gamt)

    def grad_t0(t0):                                          # скалярна координата палітри шару для режимів 1,2,3
        if gmode == 1:
            return float(_pingpong(t0 - g_dyn * gamt))
        if gmode == 2:
            return float(_pingpong(t0 - 2.0 * g_cyc * gamt))
        if gmode == 3:
            return float(t0 * (1 - gamt) + _pingpong(t0 + 2.0 * g_cyc) * gamt)
        return t0

    def grad_band_coord():                                   # координата смуг (H,W) у режимах 1..5
        yn = yy_ / max(H - 1, 1)
        xn = xx_ / max(H - 1, 1)
        flip = band_flip > 0.5
        if gmode == 2:
            c = xn * np.cos(g_ang) + yn * np.sin(g_ang)       # кут 90 -> чисто по y (як static)
        else:
            c = yn + 0.0 * xn
        if flip:
            c = 1.0 - c
        if gmode == 1:
            c = _pingpong(c - g_dyn * gamt)
        elif gmode == 2:
            c = _pingpong(c - 2.0 * g_cyc * gamt)
        elif gmode == 3:
            c = c * (1 - gamt) + _pingpong(c + 2.0 * g_cyc) * gamt
        elif gmode == 4:
            c = _pingpong(c + g_d4)
        elif gmode == 5:
            c = _pingpong(c + g_wave)
        return np.broadcast_to(c, (H, W)).astype(np.float32)

    def over(comp, rgb, a):
        comp[..., :3] = rgb + comp[..., :3] * (1 - a[..., None])
        comp[..., 3] = a + comp[..., 3] * (1 - a)

    for L in cache["layers"]:
        if L is None:
            continue
        if not stretch_done and L["c"] >= subject_lo:         # копії предмета лягають між фоном і предметом
            for c4, _ in _stretch_copies(img, m, int(stretch_steps), stretch_length, vanishing_point,
                                         stretch_shift, stretch_blur, stretch_opacity, stretch_fade,
                                         stretch_color_shift, stops):
                prev = comp[..., :3].copy() if stretch_lighten > 0 else None
                over(comp, c4[..., :3], c4[..., 3])
                if prev is not None:  # копія не може затемнити те, що під нею
                    comp[..., :3] = comp[..., :3] * (1 - stretch_lighten) + np.maximum(comp[..., :3], prev) * stretch_lighten
            stretch_done = True
        a = L["a"].astype(np.float32)
        rgb = L["rgb"].astype(np.float32)
        near_d = float(1 - _smooth(L["r"] / rmaxpx, gradient_onset, 0.55))
        if wmap is not None:                                  # те саме поле, що й для d (пошарової амплітуди більше нема)
            rgb = _remap(rgb, wmap)
            a = _remap(a, wmap)
        if smear > 0 and near_d > 0.05:  # предмет розмазується слідом (як шлейф годинника)
            hh, ww = a.shape
            ln = smear * near_d * max(hh, ww) * 0.15
            ang = np.deg2rad(smear_angle)
            acc_rgb, acc_a, wsum = np.zeros_like(rgb), np.zeros_like(a), 0.0
            for k in range(12):
                t = k / 11.0
                if smear_radial > 0.5:  # радіальний (zoom) шлейф від центру мас предмета
                    ys, xs = np.nonzero(a > 0.5)
                    cx, cy = (xs.mean(), ys.mean()) if len(xs) else (ww / 2, hh / 2)
                    sc = 1 + (ln / hh) * 2.0 * t
                    M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
                else:
                    M = np.float32([[1, 0, np.cos(ang) * ln * t], [0, 1, np.sin(ang) * ln * t]])
                wk = (1 - t) ** 1.5 + 0.05
                acc_rgb += cv2.warpAffine(rgb, M, (ww, hh)) * wk
                acc_a += cv2.warpAffine(a, M, (ww, hh)) * wk
                wsum += wk
            rgb = np.maximum(rgb, acc_rgb / wsum * 1.6)
            a = np.clip(np.maximum(a, acc_a / wsum * 1.6), 0, 1)
            rgb = np.minimum(rgb, a[..., None])
        # колір шару: фото -> градієнт від глибини, чим більший радіус блюру, тим сильніше
        mix = near_tint * (1 - _smooth(L["r"] / rmaxpx, 0, 0.1)) + far_gradient * _smooth(L["r"] / rmaxpx, gradient_onset, 0.55)
        mix = float(np.clip(mix, 0, 1))
        un = rgb / np.maximum(a[..., None], 1e-4)
        lm = None
        if mix > 0:                                           # mix = 0 -> колір шару = фото, градієнт не рахуємо
            t0 = float(np.clip(abs(focus - L["c"]) / max(focus, 1e-3), 0, 1))
            if gmode in (1, 2, 3):
                t0 = grad_t0(t0)
            g = _gradient_map_pos(np.full((1, 1), t0, np.float32), stops)[0, 0]
            gg = g[None, None, :]
            if gmode == 4:
                gg = g_field4
            elif gmode == 5:
                gg = _lut_apply(_pingpong(t0 + g_wave), g_lut)
            lm = cv2.cvtColor(np.clip(un, 0, 1), cv2.COLOR_RGB2GRAY)
            mod = (1 - luma_keep) + luma_keep * np.clip(lm * 2.5, 0, 1.3)
            col = un * (1 - mix) + np.clip(gg * mod[..., None] * gradient_gain, 0, 1) * mix
        else:
            col = un
        if subject_map > 0:  # стилізація близького: фото -> мапа градієнта від власної яскравості (силует замість реалізму)
            if lm is None:
                lm = cv2.cvtColor(np.clip(un, 0, 1), cv2.COLOR_RGB2GRAY)
            near_w = float(subject_map * (1 - _smooth(L["r"] / rmaxpx, gradient_onset, 0.55)))
            lc = np.clip((lm - 0.5) * subject_contrast + 0.5, 0, 1)
            sm = _gradient_map_pos(lc.astype(np.float32), stops)
            col = col * (1 - near_w) + np.clip(sm * gradient_gain, 0, 1) * near_w
        if bg_bands > 0:  # далеке розчиняється в смуги градієнта по висоті кадру (як фон рефа), сцена зникає
            far_w = float(bg_bands * _smooth(L["r"] / rmaxpx, gradient_onset, 0.55))
            hh = col.shape[0]
            yy = np.linspace(0, 1, hh, dtype=np.float32)
            yy = yy[::-1] if band_flip > 0.5 else yy
            if gmode:
                if band_c is None:                            # смуги однакові для всіх шарів - рахуємо раз
                    band_c = cv2.GaussianBlur(_gradient_map_pos(grad_band_coord(), stops), (0, 0), max(1.0, max(hh, col.shape[1]) * 0.04))
                band = band_c
            else:
                if band_c is None or band_c.shape[:2] != col.shape[:2]:   # смуги однакові для всіх шарів - рахуємо раз
                    band_c = _gradient_map_pos(np.repeat(yy[:, None], col.shape[1], 1), stops)
                    band_c = cv2.GaussianBlur(band_c, (0, 0), max(1.0, max(hh, col.shape[1]) * 0.04))
                band = band_c
            col = col * (1 - far_w) + np.clip(band * gradient_gain, 0, 1) * far_w
        over(comp, col * a[..., None], a)
    out = comp[..., :3] / np.maximum(comp[..., 3:4], 1e-3)
    out = np.clip(out, 0, 1)
    if smear > 0 and nmap is not None:
        d = _smear_depth(d, nmap, smear, smear_angle, smear_radial)   # хвіст шлейфу = глибина предмета (для силуету/серпанку/шлейфу)
    if silhouette > 0:
        out = _silhouette(out, d, stops, silhouette, sil_lo, sil_hi, sil_bg, sil_bg_lo, sil_detail, sil_rim, sil_rim_radius)
    if haze > 0:
        out = _haze(out, d, stops, haze, haze_start, haze_curve, haze_t, haze_scatter, haze_drift, time)
    if trail > 0:
        if trail_wobble:
            trail_angle = float(trail_angle + trail_wobble * np.sin(2 * np.pi * time * 0.3))
        out = _light_trail(out, d, stops, trail, trail_angle, trail_length, trail_source, trail_threshold,
                           trail_decay, trail_tint, subject_lo, subject_hi)
    out = _chroma_ab(out, chroma_ab * HL / 2144.0)

    if halation_strength > 0:                                   # легкий halation по яскравому
        y = cv2.cvtColor(out, cv2.COLOR_RGB2GRAY)
        mk = _smooth(y, halation_threshold, min(halation_threshold + 0.2, 0.999)) * y
        tint = np.array(halation_tint, np.float32)
        glow_ = np.zeros_like(out)
        for c, rm in ((0, 1.0), (1, 0.55), (2, 0.3)):
            glow_[..., c] = cv2.GaussianBlur(mk, (0, 0), max(halation_radius * HL * rm, 0.5)) * tint[c]
        out = 1 - (1 - out) * (1 - np.clip(glow_ * halation_strength * 2.0, 0, 1))
    pulse = 1.0
    if glow_pulse:
        pulse = max(0.0, 1.0 + glow_pulse * float(np.sin(2 * np.pi * time * pulse_hz)))
    if glow > 0:                                                # м'який bloom яскравого (після halation)
        out = _glow(out, glow * pulse, glow_radius * HL, glow_threshold, glow_warmth, stops)
    if spark > 0:                                               # точковий глоу на дрібних відблисках
        out = _sparks(out, spark * pulse, spark_threshold, spark_size, spark_rays, spark_angle, spark_length)
    if grain_animate:                                           # нове зерно кожен кадр
        seed = int(seed) + int(np.floor(time * grain_fps))
    if grain_amount > 0:
        out = film_grain(out, grain_amount, grain_size, grain_roughness, seed)
    return np.clip(out, 0, 1).astype(np.float32)
