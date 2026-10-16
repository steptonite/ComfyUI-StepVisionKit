"""SVKDepthGrade — грейдинг карти глибини мовою візуальщика + антимерехтіння відео.

Нащо (16.10.2026): LexDepthCurve говорив мовою коду (near_bias, clip_high, normalize),
і в ній не було способу ПРИГАСИТИ найближчий план — усі кроки тримали найближче = білим.
Тут те саме ядро, але:
  • повзунки названо як у Lightroom/Levels (blacks, whites, midtones, clarity, texture);
  • новий foreground_ceiling — «стеля» переднього плану (машини в кадрі стають сірими);
  • відео: deflicker прибирає мерехтіння (спільні рівні на кліп + часовий фільтр уздовж руху).

🔴 Звідки мерехтіння (заміряно 16.10.2026 на depth_ant_v3_00001, 100 кадрів):
  1) стрибки яскравості всього кадру — лікуються спільними рівнями на кліп (levels_scope);
  2) дрібне тремтіння поверхонь — шум моделі від кадру до кадру. Лікує deflicker:
     рекурсивний часовий фільтр уздовж оптичного потоку (вперед + назад, без запізнення),
     а motion_guard не дає йому мазати там, де щось справді рухається.

Стара нода лишається як була — збережені воркфлоу на ній не ламаються.
"""

import asyncio
import base64
import io
import threading
import time
from collections import OrderedDict

import numpy as np
import torch

from .depth_curve import _lohi, _stretch, _edge_base


# ───────────────────────── рух ─────────────────────────

def _small_side(h, w, target=512):
    s = min(1.0, target / float(max(h, w)))
    return max(16, int(round(w * s))), max(16, int(round(h * s))), s


def _to_u8(v):
    return np.clip(v * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _flows(gray_small):
    """Потоки між сусідами на зменшених кадрах. bw[t]: t→t-1, fw[t]: t→t+1.

    calcOpticalFlow(a, b) дає f, де a(x) ≈ b(x + f). Тож bw[t] тягне попередній кадр
    у геометрію поточного, fw[t] — наступний.
    """
    import cv2

    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    u8 = [_to_u8(g) for g in gray_small]
    n = len(u8)
    bw, fw = [None] * n, [None] * n
    for t in range(1, n):
        bw[t] = dis.calc(u8[t], u8[t - 1], None)
        fw[t - 1] = dis.calc(u8[t - 1], u8[t], None)
    return bw, fw


def _warp(img, flow_small, scale):
    """Зсунути img (повна або мала роздільність) уздовж потоку. Повертає (кадр, валідність)."""
    import cv2

    h, w = img.shape
    f = flow_small
    if f.shape[0] != h or f.shape[1] != w:
        f = cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) * (1.0 / scale)
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    mx, my = gx + f[..., 0], gy + f[..., 1]
    out = cv2.remap(img.astype(np.float32), mx, my, cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE)
    valid = (mx >= 0) & (mx <= w - 1) & (my >= 0) & (my <= h - 1)
    return out, valid


def _temporal_pass(vs, flows, strength, guard, scale, order):
    """Рекурсивний фільтр уздовж руху. Шум між кадрами усереднюється, рух лишається."""
    acc = None
    for t in order:
        v = vs[t]
        if acc is None:
            acc = v.copy()
        else:
            w, valid = _warp(acc, flows[t], scale)
            conf = np.exp(-np.square(np.abs(v - w) / guard)) * valid
            alpha = (0.9 * strength) * conf
            acc = v + alpha * (w - v)
        yield t, acc


def _jitter(frames_small, bw):
    """Індекс тремтіння: середня різниця кадру з попереднім, зсунутим потоком (×1000)."""
    vals = []
    for t in range(1, len(frames_small)):
        w, valid = _warp(frames_small[t - 1], bw[t], 1.0)
        if valid.any():
            vals.append(float(np.abs(frames_small[t] - w)[valid].mean()))
    return 1000.0 * float(np.median(vals)) if vals else 0.0


# ───────────────────────── живий прев'ю ─────────────────────────

_PCT_LO = np.arange(0.0, 20.0001, 0.5)
_PCT_HI = np.arange(80.0, 100.0001, 0.5)


def _qtable(x):
    flat = x.reshape(-1)
    if flat.size > 200_000:
        flat = flat[np.random.default_rng(0).integers(0, flat.size, 200_000)]
    return ([float(v) for v in np.percentile(flat, _PCT_LO)],
            [float(v) for v in np.percentile(flat, _PCT_HI)])


def _encode_u16(sm):
    """Кадр → (x0, x1, base64 uint16). Один формат для _preview_payload і _frame_job."""
    x0, x1 = float(sm.min()), float(sm.max())
    enc = np.round((sm - x0) / max(x1 - x0, 1e-9) * 65535.0).astype("<u2")
    return x0, x1, base64.b64encode(enc.tobytes()).decode("ascii")


def _preview_payload(x, side=384, max_keys=8):
    """Мініатюри СИРОГО депсу (до грейду) для браузера: далі він грейдить сам, без Run.

    Шлемо саме вхід x, а не готовий результат — тоді depth_is, trim, рівні й крива
    перераховуються в браузері миттєво. Для відео — до max_keys рівномірних кадрів
    (менших, щоб повідомлення лишалось ~1 МБ), кожен зі своїми квантилями для
    levels_scope = per frame; таблиці кліпу — спільні.
    """
    import cv2

    n = int(x.shape[0])
    idx = sorted(set(np.linspace(0, n - 1, min(n, max_keys)).round().astype(int).tolist()))
    if n > 1:
        side = min(side, 320)
    keys = []
    for i in idx:
        f = x[i]
        h, w = f.shape
        s = min(1.0, side / float(max(h, w)))
        sm = cv2.resize(f, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        x0, x1, data = _encode_u16(sm)
        flo, fhi = _qtable(f)
        keys.append({"i": int(i), "x0": x0, "x1": x1, "frame_lo": flo, "frame_hi": fhi,
                     "data": data})
    clo, chi = _qtable(x)
    return {"w": int(sm.shape[1]), "h": int(sm.shape[0]), "frames": n,
            "clip_lo": clo, "clip_hi": chi, "keys": keys}


# ───────────────────────── ядро грейду (спільне для Run і живого прев'ю) ─────────────────────────

class _Superseded(Exception):
    """Новіший запит прев'ю витіснив цей — ядро перериває рахунок."""


def _grade_steps(n, video, deflicker):
    """Кроки прогресу Run: тон (n), потоки (n) і два проходи deflicker (2n) — якщо є відео й deflicker > 0."""
    return n + (3 * n if video and deflicker > 0 else 0)


def _grade_array(x, p, motion=None, progress=None, clip_lohi=None):
    """Тіло колишнього SVKDepthGrade.grade. Формули не змінені.

    x — float32 (n,h,w) сирий депс; p — параметри за іменами віджетів; motion — float32 (n,h,w)
    сірий motion_guide або None; progress(k_done, k_total) — колбек або None (може кинути _Superseded);
    clip_lohi — спільні рівні кліпу (для прев'ю по фрагменту); None = рахувати з самого x, як у Run.
    Повертає (out, info).
    """
    import cv2

    depth_is, look = p["depth_is"], p["look"]
    foreground_pop, blacks, whites, midtones = p["foreground_pop"], p["blacks"], p["whites"], p["midtones"]
    foreground_ceiling, ceiling_from = p["foreground_ceiling"], p["ceiling_from"]
    hero_plane, hero_pop = p["hero_plane"], p["hero_pop"]
    clarity, texture = p["clarity"], p["texture"]
    smooth_banding, banding_edge = p["smooth_banding"], p["banding_edge"]
    trim_far, trim_near = p["trim_far"], p["trim_near"]
    flip, levels_scope = p["flip"], p["levels_scope"]
    deflicker, motion_guard = p["deflicker"], p["motion_guard"]

    n, h, w = x.shape
    video = n > 1
    run_defl = video and deflicker > 0
    total = _grade_steps(n, video, deflicker)
    done = [0]

    def tick(step=1):
        done[0] += step
        if progress is not None:
            progress(done[0], total)

    # 1) ВІДСТАНЬ: 0 = найближче
    dist = (1.0 - x) if depth_is == "near is white" else x

    sw, sh, sc = _small_side(h, w)
    small = np.stack([cv2.resize(dist[i], (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)])

    bw = fw = None
    if run_defl:
        if motion is not None and motion.shape[0] == n:
            gs = [cv2.resize(motion[i], (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)]
        else:
            gs = [_stretch(s, *_lohi(s, 1.0, 99.0)) for s in small]   # кожен кадр у свій діапазон — потоку так легше
        bw, fw = _flows(gs)
        tick(n)

    # «до»: той самий кліп у тій самій шкалі, але ще без замка й без deflicker
    jit_in = _jitter([_stretch(s, *_lohi(small, 1.0, 99.0)) for s in small], bw) if bw else None

    # 🔴 Тут був замок шкали (афінна підгонка кадру до сусіда уздовж потоку) — ПРИБРАНО
    #    16.10.2026 після двох замірів на depth_ant_v3_00001: і ланцюжок, і «стабілізатор
    #    траєкторії» ПОСИЛЮВАЛИ і тремтіння, і пульсацію (1.32 → 2.36). Глобального
    #    «дихання» в DA3 після спільних рівнів майже нема, а підгонка вносить свій шум ±0.02.
    #    Мерехтіння DA3 — ЛОКАЛЬНЕ, його гасить deflicker нижче.
    scale_note = ""

    # 3) РІВНІ по всьому кліпу або по кадру
    if levels_scope == "whole clip" and video:
        if clip_lohi is None:
            clip_lohi = _lohi(dist, trim_near, 100.0 - trim_far)
    else:
        clip_lohi = None

    near_bias = 2.0 * (0.01 ** float(foreground_pop))   # 0 → 2.0 (лінійно), 1 → 0.02 (максимум)
    bp = float(blacks)
    wp = max(1.0 - float(whites), bp + 1e-3)
    gamma = 2.0 ** (-float(midtones))
    c, k = float(foreground_ceiling), float(ceiling_from)

    out = np.empty_like(dist)
    for i in range(n):
        lohi = clip_lohi if clip_lohi is not None else _lohi(dist[i], trim_near, 100.0 - trim_far)
        d = _stretch(dist[i], *lohi)

        if look == "V2 volume":
            v = 1.0 / (d + near_bias)
            v = _stretch(v, 1.0 / (1.0 + near_bias), 1.0 / near_bias)
        else:
            v = 1.0 - d

        v = np.clip((v - bp) / (wp - bp), 0.0, 1.0)
        if gamma != 1.0:
            v = np.power(v, gamma)

        if hero_pop > 0:
            s, hp = float(hero_pop), float(hero_plane)
            g0 = float(1.0 / (1.0 + np.exp(s * hp)))
            g1 = float(1.0 / (1.0 + np.exp(-s * (1.0 - hp))))
            v = ((1.0 / (1.0 + np.exp(-s * (v - hp)))) - g0) / max(g1 - g0, 1e-6)
            v = np.clip(v, 0.0, 1.0).astype(np.float32)

        # СТЕЛЯ переднього плану: усе світліше за k м'яко лягає під c.
        # c > k: плече з нахилом 1 у точці k (без зламу), найближче = c.
        # c ≤ k: найближче ТЕМНІШЕ за героя — творчий «провал» переднього плану.
        if c < 1.0 and k < 1.0:
            m = v > k
            # clip: float32 дає t = 1.0000001 на вершині → np.power(негативна основа) = NaN, і він
            # розповзається по всьому кліпу через deflicker. Для скінченних пікселів нічого не змінює.
            t = np.clip((v[m] - k) / (1.0 - k), 0.0, 1.0)
            if c > k:
                pw = (1.0 - k) / max(c - k, 1e-3)
                shape = 1.0 - np.power(1.0 - t, pw)
            else:
                shape = t * t * (3.0 - 2.0 * t)
            v[m] = k + (c - k) * shape

        v = v.astype(np.float32)
        if smooth_banding > 0:
            r = max(3, int(round(min(v.shape) / 220.0)) | 1)
            sm = cv2.bilateralFilter(v, -1, float(banding_edge), r)
            v = v * (1.0 - smooth_banding) + sm * smooth_banding

        if clarity > 0:
            rad = int(max(4, min(v.shape) / 45))
            base = _edge_base(v, rad)
            v = np.clip(v + (v - base) * float(clarity) * 1.6, 0.0, 1.0)

        if texture > 0:
            blur = cv2.GaussianBlur(v, (0, 0), 2.0)
            v = np.clip(v + (v - blur) * float(texture), 0.0, 1.0)

        out[i] = (1.0 - v) if flip else v
        tick()

    # 4) DEFLICKER: вперед і назад уздовж руху, середнє — без запізнення й без шлейфу в один бік
    if run_defl:
        st, gd = float(deflicker), float(motion_guard)
        fwd = np.empty_like(out)
        for t, acc in _temporal_pass(out, bw, st, gd, sc, range(n)):
            fwd[t] = acc
            tick()
        for t, acc in _temporal_pass(out, fw, st, gd, sc, range(n - 1, -1, -1)):
            fwd[t] = 0.5 * (fwd[t] + acc)
            tick()
        out = np.clip(fwd, 0.0, 1.0)

    jit_note = ""
    if jit_in is not None:
        # «після»: вихід, зведений до лінійної шкали входу, — щоб криві грейду не міняли мірило
        so = np.stack([cv2.resize(out[i], (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)])
        jit_note = f" · тремтіння ×1000: вхід {jit_in:.1f} → вихід {_jitter(list(_stretch(so, *_lohi(so, 1.0, 99.0))), bw):.1f}"

    q = np.percentile(out[0], [2, 25, 50, 75, 98])
    info = (f"кадрів {n} · {look} pop {foreground_pop}"
            + f" · blacks {blacks} whites {whites} midtones {midtones}"
            + (f" · стеля {c} від {k}" if c < 1.0 else "")
            + (f" · герой {hero_plane}×{hero_pop}" if hero_pop > 0 else "")
            + scale_note + jit_note
            + " · тони p2/25/50/75/98: " + "/".join(f"{v:.2f}" for v in q))
    hist = np.histogram(out[0], bins=10, range=(0.0, 1.0))[0] / out[0].size
    bars = "".join(" ▁▂▃▄▅▆▇█"[min(8, int(p * 8 / max(hist.max(), 1e-6) + 0.5))] for p in hist)
    print(f"[DepthGrade] {info}")
    print(f"[DepthGrade] плани 0.0→1.0: |{bars}|  (hero_plane став на горб героя)")
    return out, info


# ───────────────────────── кеш кадрів для живого прев'ю ─────────────────────────

_CACHE = OrderedDict()                # ключ ноди → (x16, motion16 | None, n_orig, stride)
_CACHE_MAX = 2                        # скільки ключів тримаємо
_CACHE_SIDE = 768                     # довша сторона кешу
_CACHE_BYTES = 250 * 1024 * 1024      # більше — беремо кожен stride-й кадр
_CLIP_Q = {}                          # ключ → (entry, clip_lo, clip_hi): квантилі всього кешу, раз на запис
_LATEST = {}                          # ключ → (gen, monotonic час запиту)
_LOCK = threading.Lock()
_STALE_SEC = 60.0                     # після такої паузи gen можна починати з нуля (перезавантажили сторінку)
_WARM = 8                             # запас кадрів з кожного боку — розігрів deflicker


def _resize_stack(a, side):
    """Зменшити стек кадрів (n,h,w) так, щоб довша сторона була ≤ side. Збільшення немає."""
    import cv2

    h, w = int(a.shape[1]), int(a.shape[2])
    s = min(1.0, side / float(max(h, w)))
    if s >= 1.0:
        return a
    nh, nw = max(1, int(round(h * s))), max(1, int(round(w * s)))
    return np.stack([cv2.resize(f, (nw, nh), interpolation=cv2.INTER_AREA) for f in a]).astype(np.float32)


def _cache_put(key, x, motion, n):
    """Кладе сирий депс (зменшений, float16) під ключ ноди. Найстаріший ключ викидається."""
    xs = _resize_stack(x, _CACHE_SIDE).astype(np.float16)
    ms = None
    if motion is not None and motion.shape[0] == n:
        ms = _resize_stack(motion, _CACHE_SIDE).astype(np.float16)
    stride = max(1, -(-int(xs.nbytes) // _CACHE_BYTES))
    if stride > 1:
        xs = xs[::stride]
        ms = ms[::stride] if ms is not None else None
    with _LOCK:
        _CACHE[key] = (xs, ms, int(n), int(stride))
        _CACHE.move_to_end(key)
        _CLIP_Q.pop(key, None)
        while len(_CACHE) > _CACHE_MAX:
            old, _ = _CACHE.popitem(last=False)
            _LATEST.pop(old, None)
            _CLIP_Q.pop(old, None)


def _cache_clip_lohi(x16, p):
    """Рівні всього кліпу з кешу тією ж логікою, що в Run: щоб прев'ю по фрагменту не міняло яскравість."""
    flat = x16.reshape(-1)
    if flat.size > 200_000:
        flat = flat[np.random.default_rng(0).integers(0, flat.size, 200_000)]
    flat = flat.astype(np.float32)
    dist = (1.0 - flat) if p["depth_is"] == "near is white" else flat
    return _lohi(dist, p["trim_near"], 100.0 - p["trim_far"])


def _clip_qtable(key, entry, x16):
    """Квантилі всього кешу для браузера (Статика, levels_scope = whole clip). Рахуємо раз на запис кешу."""
    with _LOCK:
        cq = _CLIP_Q.get(key)
    if cq is not None and cq[0] is entry:
        return cq[1], cq[2]
    flat = x16.reshape(-1)
    if flat.size > 200_000:
        flat = flat[np.random.default_rng(0).integers(0, flat.size, 200_000)]
    clo, chi = _qtable(flat.astype(np.float32))
    with _LOCK:
        if _CACHE.get(key) is entry:
            _CLIP_Q[key] = (entry, clo, chi)
    return clo, chi


def _is_superseded(key, gen):
    with _LOCK:
        cur = _LATEST.get(key)
    return cur is not None and cur[0] > gen


# ───────────────────────── нода ─────────────────────────

class SVKDepthGrade:
    """Грейдинг карти глибини (Levels/Curves/Clarity) + стеля переднього плану + антимерехтіння."""

    @classmethod
    def INPUT_TYPES(cls):
        F = "FLOAT"
        return {
            "required": {
                "depth": ("IMAGE", {"tooltip": "карта глибини: вихід DA3Render або будь-який депсмеп"}),
                "depth_is": (["near is white", "far is white (raw)"], {"default": "near is white",
                             "tooltip": "звичайний депсмеп = близьке світле. 'far is white' — сирий DA3 (режим raw), де значення = відстань"}),
                "look": (["V2 volume", "linear"], {"default": "V2 volume",
                         "tooltip": "🥇 V2 volume = як Depth Anything V2: передній план отримує більше тонів, обличчя об'ємне. linear = як є"}),
                "foreground_pop": (F, {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01,
                                   "tooltip": "скільки тонів віддати передньому плану (лише V2 volume). 0 = майже лінійно · 0.45 звично · 1.0 = максимум, усе далеке провалюється в темне"}),
                "blacks": (F, {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01,
                           "tooltip": "лівий повзунок Levels: заганяє ДАЛЬНІЙ план у чорне. Підіймай, доки фон не зникне"}),
                "whites": (F, {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01,
                           "tooltip": "правий повзунок Levels: заганяє НАЙБЛИЖЧЕ в біле. Чим вища — тим ширша біла пляма"}),
                "midtones": (F, {"default": 0.0, "min": -2.0, "max": 2.0, "step": 0.05,
                             "tooltip": "середній повзунок Levels. + світліше середні плани, − темніше"}),
                "foreground_ceiling": (F, {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                       "tooltip": "🆕 ПРИГАСИТИ ПЕРЕДНІЙ ПЛАН. Яким сірим стане найближче: 1.0 = біле (вимкнено), 0.7 = машини/речі впритул до камери стають сірими. Нижче за ceiling_from — найближче стає ТЕМНІШИМ за героя"}),
                "ceiling_from": (F, {"default": 0.75, "min": 0.0, "max": 1.0, "step": 0.01,
                                 "tooltip": "з якого тону починається пригашування. Став трохи ВИЩЕ тону героя — тоді герой лишається як був, а все, що ближче, м'яко лягає під стелю"}),
                "hero_plane": (F, {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                               "tooltip": "🥇 на якому тоні стоїть твій герой (0 = найдальше, 1 = найближче). Підказка — гістограма планів у консолі: горб = план"}),
                "hero_pop": (F, {"default": 0.0, "min": 0.0, "max": 14.0, "step": 0.5,
                             "tooltip": "🥇 виділити героя: S-крива навколо hero_plane. Його тони розтягуються, решта м'яко стискається, але не обрізається. 4–8 робоче, 0 = вимкнено"}),
                "clarity": (F, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                            "tooltip": "як Clarity у Lightroom: ліплення форм усередині об'єкта (вилиці, складки), без ореолу по силуету. 0.3–0.5"}),
                "texture": (F, {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                            "tooltip": "як Texture у Lightroom: дрібні складочки й кромки. 0.3–0.5"}),
                "smooth_banding": (F, {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                   "tooltip": "прибрати сходинки-смуги в плавних переходах; силует лишається різким. 0.2–0.5"}),
                "banding_edge": (F, {"default": 0.12, "min": 0.01, "max": 0.5, "step": 0.01,
                                 "tooltip": "який перепад вважати смугою, а не межею. 0.06 — лише смуги · 0.12 звично · 0.25+ з'їдає й листя, але тупить силует"}),
                "trim_far": (F, {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                             "tooltip": "ігнорувати % найдальших пікселів, коли рахуються рівні (небо, викиди)"}),
                "trim_near": (F, {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                              "tooltip": "ігнорувати % найближчих пікселів, коли рахуються рівні (пилинка на об'єктиві, рука в кадрі)"}),
                "flip": ("BOOLEAN", {"default": False, "tooltip": "перевернути: близьке темне, далеке світле"}),
                "levels_scope": (["whole clip", "per frame"], {"default": "whole clip",
                                 "tooltip": "відео: whole clip = одні рівні на весь кліп (без стрибків яскравості). per frame = кожен кадр сам по собі (для фото)"}),
                "deflicker": (F, {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "🆕 ВІДЕО: гасить дрібне тремтіння поверхонь (стіни, асфальт, шкіра). 0 = вимкнено · 0.6 звично · 1.0 = максимум; якщо з'являються шлейфи — менше"}),
                "motion_guard": (F, {"default": 0.08, "min": 0.01, "max": 0.4, "step": 0.01,
                                 "tooltip": "ВІДЕО: який стрибок тону вважати справжнім рухом, а не мерехтінням. Менше = нуль шлейфів, але й гасить слабше; більше = гасить сильніше, ризик шлейфу за рухом"}),
            },
            "optional": {
                "motion_guide": ("IMAGE", {"tooltip": "ВІДЕО: підключи СВОЄ оригінальне відео (ті самі кадри). Рух тоді рахується по картинці, а не по депсу — точніше, менше шлейфів"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("depth", "depth_mask", "info")
    FUNCTION = "grade"
    CATEGORY = "StepVisionKit/Depth"

    def grade(self, depth, depth_is, look, foreground_pop, blacks, whites, midtones,
              foreground_ceiling, ceiling_from, hero_plane, hero_pop, clarity, texture,
              smooth_banding, banding_edge, trim_far, trim_near, flip, levels_scope,
              deflicker, motion_guard, motion_guide=None, unique_id=None):
        x = depth.cpu().numpy().astype(np.float32)
        if x.ndim == 4:
            x = x[..., :3].mean(axis=3)
        n = x.shape[0]
        video = n > 1
        preview = _preview_payload(x)
        preview["cache_key"] = str(unique_id) if unique_id is not None else ""
        preview["frames_total"] = int(n)

        motion = None
        if video and motion_guide is not None and motion_guide.shape[0] == n:
            g = motion_guide.cpu().numpy().astype(np.float32)
            motion = g[..., :3].mean(axis=3) if g.ndim == 4 else g

        progress = None
        try:
            from comfy.utils import ProgressBar
            pbar = ProgressBar(_grade_steps(n, video, deflicker))

            def progress(k, _total):
                try:
                    pbar.update_absolute(k)
                except Exception:
                    pass
        except Exception:
            progress = None

        p = dict(depth_is=depth_is, look=look, foreground_pop=foreground_pop, blacks=blacks,
                 whites=whites, midtones=midtones, foreground_ceiling=foreground_ceiling,
                 ceiling_from=ceiling_from, hero_plane=hero_plane, hero_pop=hero_pop,
                 clarity=clarity, texture=texture, smooth_banding=smooth_banding,
                 banding_edge=banding_edge, trim_far=trim_far, trim_near=trim_near,
                 flip=flip, levels_scope=levels_scope, deflicker=deflicker, motion_guard=motion_guard)
        out, info = _grade_array(x, p, motion=motion, progress=progress)

        if unique_id is not None:
            try:
                _cache_put(str(unique_id), x, motion, n)
            except Exception as e:
                print(f"[DepthGrade] кеш для живого прев'ю не записано: {e}")

        tt = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
        return {"ui": {"svk_depth_preview": [preview]},
                "result": (tt.unsqueeze(-1).repeat(1, 1, 1, 3), tt, info)}


class SVKDepthGradeView:
    """Калібрування кадру. Під'єднай вхід до виходу depth ноди Depth Grade: вихід віддає той самий депс далі,
    тож ланцюжок не рветься. Панель у ноді — піпетки, До/після і криві пишуть у повзунки Depth Grade."""
    DESCRIPTION = ("Калібрування кадру: під'єднай до виходу depth ноди Depth Grade. Вхід іде на вихід без змін, "
                   "тож ланцюжок не рветься. Піпетки й криві пишуть у ті самі повзунки Depth Grade.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"depth": ("IMAGE", {"tooltip": "вихід depth ноди Depth Grade"})}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/Depth"

    def view(self, depth):
        return {"ui": {}, "result": (depth,)}


class SVKDepthGradeVideoView:
    """Живе прев'ю відео: фрагмент кліпу перераховується сам при зміні повзунків Depth Grade."""
    DESCRIPTION = ("Живе прев'ю відео: під'єднай до виходу depth ноди Depth Grade, натисни Run один раз — "
                   "далі прев'ю перераховується саме при зміні повзунків, у зменшеному розмірі. "
                   "Clarity, texture і smooth_banding на малому розмірі видно слабше — точний вигляд дає Run.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"depth": ("IMAGE", {"tooltip": "вихід depth ноди Depth Grade"})},
            "optional": {"fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                                           "tooltip": "швидкість відтворення прев'ю"})},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/Depth"

    def view(self, depth, fps=24.0):
        return {"ui": {}, "result": (depth,)}


class SVKDepthGradeLive(SVKDepthGrade):
    """Стара назва (16.10.2026): пульт переїхав у SVKDepthGradeView. Лишено, щоб
    збережені воркфлоу не червоніли; у пошуку не видно."""
    DEPRECATED = True



# ───────────────────────── живе прев'ю: обчислення одного запиту ─────────────────────────

_PREVIEW_SIZES = (256, 384, 512, 768)
_SPECS = SVKDepthGrade.INPUT_TYPES()["required"]
_PARAM_NAMES = [name for name in _SPECS if name != "depth"]
_CHOICES = {name: tuple(_SPECS[name][0]) for name in _PARAM_NAMES if isinstance(_SPECS[name][0], list)}
_DEFAULTS = {name: _SPECS[name][1]["default"] for name in _PARAM_NAMES}


def _coerce_params(raw):
    """Параметри з JSON → типи віджетів. Відсутній ключ = дефолт віджета."""
    p = {}
    for name in _PARAM_NAMES:
        v = raw.get(name, _DEFAULTS[name])
        if name in _CHOICES:
            if v not in _CHOICES[name]:
                raise ValueError(f"{name}: неочікуване значення {v!r}")
            p[name] = v
        elif name == "flip":
            p[name] = bool(v)
        else:
            p[name] = float(v)
    return p


def _jpeg_b64(f):
    from PIL import Image

    u8 = (np.clip(f, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(u8)).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _preview_job(body):
    """Один запит живого прев'ю: фрагмент з кешу → грейд тим самим ядром → до/після JPEG.

    start/count — у кадрах оригінального кліпу. Відповідь: start — оригінальний індекс першого
    повернутого кадру, count — скільки кадрів повернуто (при stride > 1 це кадри кешу, кожен = stride
    оригінальних). Викликається в executor, тому синхронна.
    """
    t0 = time.perf_counter()
    try:
        key = str(body.get("key", ""))
        gen = int(body.get("gen", 0))
    except Exception as e:
        return {"ok": False, "gen": 0, "reason": "error", "message": f"key/gen: {e}"}

    def fail(reason, msg):
        return {"ok": False, "gen": gen, "reason": reason, "message": msg}

    try:
        size = int(body.get("size", 384))
        start = int(body.get("start", 0))
        count = int(body.get("count", 48))
        if size not in _PREVIEW_SIZES:
            return fail("error", "size має бути 256, 384, 512 або 768")
        if not 1 <= count <= 96:
            return fail("error", "count має бути від 1 до 96")
        raw = body.get("params")
        if not isinstance(raw, dict):
            return fail("error", "params має бути обʼєктом")
        p = _coerce_params(raw)

        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None:
                _CACHE.move_to_end(key)
                cur = _LATEST.get(key)
                now = time.monotonic()
                if cur is not None and gen < cur[0] and now - cur[1] < _STALE_SEC:
                    return fail("superseded", "новіший запит уже йде")
                _LATEST[key] = (gen, now)
        if entry is None:
            return fail("no_cache", "немає кадрів у кеші: натисни Run на Depth Grade один раз")

        x16, m16, n_orig, stride = entry
        nc = int(x16.shape[0])
        start = min(max(start, 0), n_orig - 1)
        end = min(n_orig, start + count) - 1
        c0 = start // stride
        c1 = max(c0 + 1, min(nc, end // stride + 1))
        lo, hi = max(0, c0 - _WARM), min(nc, c1 + _WARM)

        xs = _resize_stack(x16[lo:hi].astype(np.float32), size)
        ms = _resize_stack(m16[lo:hi].astype(np.float32), size) if m16 is not None else None
        clip = _cache_clip_lohi(x16, p) if (p["levels_scope"] == "whole clip" and n_orig > 1) else None

        def cb(_k, _t):
            if _is_superseded(key, gen):
                raise _Superseded()

        out, _info = _grade_array(xs, p, motion=ms, progress=cb, clip_lohi=clip)

        k0 = c0 - lo
        k1 = k0 + (c1 - c0)
        after = out[k0:k1]
        frag = xs[k0:k1]
        lo_p, hi_p = np.percentile(frag, [1.0, 99.0])
        before = np.clip((frag - lo_p) / max(float(hi_p - lo_p), 1e-6), 0.0, 1.0)
        return {"ok": True, "gen": gen, "w": int(out.shape[2]), "h": int(out.shape[1]),
                "start": int(c0 * stride), "count": int(c1 - c0), "frames_total": int(n_orig),
                "after": [_jpeg_b64(f) for f in after],
                "before": [_jpeg_b64(f) for f in before],
                "ms": int(round((time.perf_counter() - t0) * 1000))}
    except _Superseded:
        return fail("superseded", "новіший запит уже йде")
    except Exception as e:
        return fail("error", str(e) or type(e).__name__)


def _frame_job(body):
    """Один кадр для режиму «Статика»: кадр кешу → uint16, без грейду. Тон рахує браузер.

    i — оригінальний індекс кадру (затискається в межі кліпу). Відповідь має формат _preview_payload
    з одним ключем, тож браузер розкодує її тим самим M.decodePreview. Викликається в executor.
    """
    try:
        key = str(body.get("key", ""))
        i = int(body.get("i", 0))
        size = int(body.get("size", 384))
    except Exception as e:
        return {"ok": False, "reason": "error", "message": f"key/i/size: {e}"}

    try:
        if size not in _PREVIEW_SIZES:
            return {"ok": False, "reason": "error", "message": "size має бути 256, 384, 512 або 768"}
        with _LOCK:
            entry = _CACHE.get(key)
            if entry is not None:
                _CACHE.move_to_end(key)
        if entry is None:
            return {"ok": False, "reason": "no_cache",
                    "message": "немає кадрів у кеші: натисни Run на Depth Grade один раз"}

        x16, _m16, n_orig, stride = entry
        nc = int(x16.shape[0])
        i = min(max(i, 0), n_orig - 1)
        ci = min(max(i // stride, 0), nc - 1)
        f = x16[ci].astype(np.float32)
        sm = _resize_stack(f[None], size)[0]
        flo, fhi = _qtable(f)
        clo, chi = _clip_qtable(key, entry, x16)
        x0, x1, data = _encode_u16(sm)
        return {"ok": True, "w": int(sm.shape[1]), "h": int(sm.shape[0]), "frames": int(n_orig),
                "clip_lo": clo, "clip_hi": chi,
                "keys": [{"i": int(ci * stride), "x0": x0, "x1": x1,
                          "frame_lo": flo, "frame_hi": fhi, "data": data}]}
    except Exception as e:
        return {"ok": False, "reason": "error", "message": str(e) or type(e).__name__}


# ───────────────────────── маршрут ─────────────────────────

try:
    from server import PromptServer
    from aiohttp import web as _web

    @PromptServer.instance.routes.post("/svk/depth_grade/preview")
    async def _svk_depth_grade_preview(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("тіло запиту має бути обʼєктом")
        except Exception as e:
            return _web.json_response({"ok": False, "gen": 0, "reason": "error", "message": str(e)})
        loop = asyncio.get_running_loop()
        return _web.json_response(await loop.run_in_executor(None, _preview_job, body))

    @PromptServer.instance.routes.post("/svk/depth_grade/frame")
    async def _svk_depth_grade_frame(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("тіло запиту має бути обʼєктом")
        except Exception as e:
            return _web.json_response({"ok": False, "reason": "error", "message": str(e)})
        loop = asyncio.get_running_loop()
        return _web.json_response(await loop.run_in_executor(None, _frame_job, body))
except Exception as _e:
    print(f"[DepthGrade] маршрут живого прев'ю не зареєстровано: {_e}")
