"""LexDepthCurve — тонова крива для депсмепів (щоб DA3 віддавав як Depth Anything V2).

Нащо (розібрано 10.08.2026, ядро `comfy/ldm/depth_anything_3/preprocess.py:93`):
Depth Anything **V2** віддає disparity (≈1/відстань) — нелінійність, яка розтягує
ближній план. Depth Anything **V3** віддає ВІДСТАНЬ, а режим `v2_style` у DA3Render
розтягує її ЛІНІЙНО між квантилями 1%/99% і просто інвертує. Тому на портреті
(обличчя 1.0–1.3 м, фон 3 м) обличчя займає ~15% тонального діапазону замість ~34%,
і V3 виглядає «пласкіше», хоч модель і точніша, і швидша.

Ця нода — чиста арифметика над УЖЕ порахованим депсмепом. Моделі не запускає,
ваги не вантажить. Ціна = один прохід по тензору.
"""

import numpy as np
import torch


def _lohi(x, lo_p, hi_p):
    """Квантилі по вибірці — швидше за повний сорт і практично те саме."""
    flat = x.reshape(-1)
    if flat.size > 200_000:
        flat = flat[np.random.default_rng(0).integers(0, flat.size, 200_000)]
    return float(np.quantile(flat, lo_p / 100.0)), float(np.quantile(flat, hi_p / 100.0))


def _stretch(x, lo, hi):
    return np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def _pct_norm(x, lo_p, hi_p, lohi=None):
    """Нормалізація по квантилях. lohi задано ззовні -> спільне на весь батч (без мерехтіння)."""
    lo, hi = lohi if lohi is not None else _lohi(x, lo_p, hi_p)
    return _stretch(x, lo, hi)


def _edge_base(v, rad):
    """База для clarity, яка НЕ переступає межі об'єктів (звідси відсутність ореолу).

    guided filter із самим собою як гідом. Якщо opencv без contrib — падаємо назад
    на білатеральний по зменшеній копії (той самий сенс, дешевше за повний білатерал:
    1.4 с проти 0.015 с на кадрі 1080×1920, заміряно 10.08.2026).
    """
    import cv2

    v = np.ascontiguousarray(v, dtype=np.float32)   # cv2 бере тільки 8u/32f
    if hasattr(cv2, "ximgproc"):
        return cv2.ximgproc.guidedFilter(v, v, rad, 0.01)
    s = cv2.resize(v, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    b = cv2.bilateralFilter(s, -1, 0.15, max(3, rad // 4))
    return cv2.resize(b, (v.shape[1], v.shape[0]), interpolation=cv2.INTER_LINEAR)


class LexDepthCurve:
    """Перегнати депсмеп у V2-подібний вигляд (disparity) або покрутити гаму."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "depth": ("IMAGE", {"tooltip": "Depth map — output of DA3Render, Depth Receiver or any depth IMAGE"}),
                "source": (["near_is_white", "raw_distance"], {"default": "near_is_white",
                           "tooltip": "near_is_white = a regular depth map where near is bright (DA3Render v2_style / min_max, ControlNet depth). raw_distance = DA3Render in 'raw' mode, where values are distances and far is larger"}),
                "mode": (["disparity", "gamma", "none"], {"default": "disparity",
                         "tooltip": "disparity = like Depth Anything V2 (1/distance): stretches the foreground so faces look volumetric. gamma = simple curve. none = renormalize only"}),
                "near_bias": ("FLOAT", {"default": 0.25, "min": 0.02, "max": 2.0, "step": 0.01,
                              "tooltip": "strength of the non-linearity in disparity mode. Lower = sharper separation of the foreground (0.1 gives a very contrasty face), higher = closer to linear (1.0 ≈ unchanged)"}),
                "black_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01,
                                "tooltip": "LEFT Levels slider, as in Photoshop. Everything darker than this becomes black, the rest is STRETCHED. Raise it until the background drops to black"}),
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 1.0, "step": 0.01,
                                "tooltip": "RIGHT Levels slider. Everything brighter becomes white. Lower it until the nearest object blows out to white. The narrower the black↔white gap, the more gaps appear between depth planes"}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.05,
                           "tooltip": "MIDDLE Levels slider. <1 brightens the midtones, >1 darkens them"}),
                "focus": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                          "tooltip": "WHICH PLANE to emphasize: the tone your subject sits at (0 = farthest, 1 = nearest). Check the histogram in the console — three humps = three planes; put focus on the one you want"}),
                "focus_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 14.0, "step": 0.5,
                                   "tooltip": "how steep the curve is around focus. This is a Curves-style S-curve: tones AROUND the plane are stretched and the rest is softly compressed — unlike black/white point, which simply clip other planes to black/white. 4–8 works well"}),
                "clarity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                            "tooltip": "Clarity: local contrast (CLAHE). Sculpts form inside a shape — cloth folds, cheekbones. Global levels cannot do this. 0.3–0.5 works well"}),
                "detail": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.1,
                           "tooltip": "micro-contrast (unsharp mask with a small radius): fine folds and edges. 0.3–0.5"}),
                "deband": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05,
                           "tooltip": "remove stair-step BANDING without blurring the image. Bilateral filter: smooths small steps inside a plane and keeps the silhouette and plane boundaries sharp. 0.3–0.5. Runs BEFORE clarity, so clarity has nothing left to amplify"}),
                "deband_edge": ("FLOAT", {"default": 0.12, "min": 0.01, "max": 0.5, "step": 0.01,
                                "tooltip": "what step height counts as noise rather than a real edge. 0.06 — banding only · 0.12 default · 0.2–0.35 also catches ragged leaf clutter but starts to erode silhouette sharpness"}),
                "clip_low": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 20.0, "step": 0.5,
                             "tooltip": "lower clipping percentile, %"}),
                "clip_high": ("FLOAT", {"default": 99.0, "min": 80.0, "max": 100.0, "step": 0.5, "tooltip": "percentile taken as the far/near limit; lower clips more of the extreme values"}),
                "invert": ("BOOLEAN", {"default": False,
                           "tooltip": "invert the result if near came out dark"}),
                "normalize": (["batch", "per_frame"], {"default": "batch",
                              "tooltip": "VIDEO: 'batch' — black and white points are computed ONCE for the whole batch of frames. This is what removes depth brightness flicker. 'per_frame' — per frame separately (old behavior; fine for single stills)"}),
            },
        }

    DESCRIPTION = "Tone curve for a Depth Anything 3 map so it reads like Depth Anything V2: levels, gamma, focus band, clarity, deband."
    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("depth", "depth_mask", "info")
    FUNCTION = "curve"
    CATEGORY = "StepVisionKit"

    def curve(self, depth, source, mode, near_bias, black_point, white_point, gamma,
              clarity, detail, clip_low, clip_high, invert, deband=0.0, deband_edge=0.06,
              normalize="batch", focus=0.5, focus_strength=0.0):
        import cv2

        x = depth.cpu().numpy().astype(np.float32)
        if x.ndim == 4:
            x = x[..., :3].mean(axis=3)           # депсмеп сірий — беремо один канал
        n = x.shape[0]
        out = np.empty_like(x)

        # 🔴 МЕРЕХТІННЯ ЯСКРАВОСТІ на відео йшло звідси: квантилі рахувались НА КОЖНОМУ КАДРІ.
        #    Варто об'єкту зайти/вийти з кадру — чорна й біла точки їдуть, і вся картинка
        #    смикається по яскравості. На батчі рахуємо їх ОДИН раз (спіймано 10.08.2026).
        gl0 = _lohi(x, clip_low, clip_high) if normalize == "batch" and n > 1 else None

        for i in range(n):
            f = x[i]
            # 1) звести до d = ВІДСТАНЬ у [0,1], де 0 = найближче
            d = _pct_norm(f, clip_low, clip_high, gl0)
            if source == "near_is_white":
                d = 1.0 - d

            # 2) крива
            if mode == "disparity":
                v = 1.0 / (d + float(near_bias))   # близьке -> велике
                # межі disparity аналітичні, тому на батчі вони СПІЛЬНІ автоматично
                v = _stretch(v, 1.0 / (1.0 + float(near_bias)), 1.0 / float(near_bias))
            elif mode == "gamma":
                v = 1.0 - d
            else:
                v = 1.0 - d

            # 3) РІВНІ (Levels): чорна й біла точки. Саме це «розсуває провали»,
            #    на відміну від near_bias, який просто зсипає масу в чорне.
            bp, wp = float(black_point), float(white_point)
            if wp - bp < 1e-3:
                wp = bp + 1e-3
            v = np.clip((v - bp) / (wp - bp), 0.0, 1.0)

            if gamma != 1.0:
                v = np.power(v, float(gamma))

            # 3b) ФОКУС НА ПЛАНІ — S-крива (як Curves у фотошопі, а не Levels).
            #     black/white point інші плани ЗРІЗАЮТЬ у чорне й біле, тому й «об'єм
            #     не витягується»: щойно ти розтягнув середній план, передній і задній
            #     ставали пласкими плямами. Логістична крива навколо focus розтягує
            #     тони СВОГО плану, а решту м'яко стискає — вони лишаються читабельні.
            #     Кінці 0 і 1 крива тримає, тому загальний діапазон не з'їдається.
            if focus_strength > 0:
                s, p = float(focus_strength), float(focus)
                g0 = float(1.0 / (1.0 + np.exp(s * p)))
                g1 = float(1.0 / (1.0 + np.exp(-s * (1.0 - p))))
                v = ((1.0 / (1.0 + np.exp(-s * (v - p)))) - g0) / max(g1 - g0, 1e-6)
                # 🔴 np.exp повертає float64, і масив тихо ставав 64-бітним — а cv2
                #    нижче (bilateral, guidedFilter) приймає лише 8u і 32f. Звідси й
                #    виліт ноди (спіймано 10.08.2026). Тримаємо float32 явно.
                v = np.clip(v, 0.0, 1.0).astype(np.float32)

            # 4) ДЕБЕНДИНГ — до clarity, щоб та не мала що підсилювати.
            #    Білатеральний фільтр рівняє перепади НИЖЧЕ deband_edge (сходинки квантування),
            #    а перепади вище — силует, межа планів — лишає різкими. Тому це не «блюр картинки».
            if deband > 0:
                r = max(3, int(round(min(v.shape) / 220.0)) | 1)
                sm = cv2.bilateralFilter(v, -1, float(deband_edge), r)
                v = v * (1.0 - deband) + sm * deband

            # 5) Clarity — локальний контраст ВЕЛИКИМ радіусом, як однойменний повзунок
            #    у фотошопі: v + (v - сильно_розмите) * сила.
            # 🔴 Було CLAHE — і це була помилка (спіймано 10.08.2026 на кадрі 1080×1920).
            #    CLAHE вирівнює гістограму В КОЖНІЙ ПЛИТЦІ ОКРЕМО, тому на плитці, що лягла
            #    на рівний фон, він розтягував мікрошум на весь діапазон → плями по кущах
            #    і стінах. Плюс він адаптується покадрово, тобто ще й мерехтить на відео.
            #    Unsharp лінійний: плям не робить і між кадрами стабільний.
            # 🔴 А потім вилізло друге: гаусова база дає ОРЕОЛ по силуету (спіймано
            #    10.08.2026 — «край контурів шарпиться»). Гаус розмиває ЧЕРЕЗ межу,
            #    тому біля силуету різниця (v - база) величезна і малює світлий кант
            #    зовні й темний усередині. Guided filter межу не переступає -> канта нема.
            if clarity > 0:
                rad = int(max(4, min(v.shape) / 45))
                base = _edge_base(v, rad)
                v = np.clip(v + (v - base) * float(clarity) * 1.6, 0.0, 1.0)

            # 5) мікроконтраст: підняти дрібні складки
            if detail > 0:
                blur = cv2.GaussianBlur(v, (0, 0), 2.0)
                v = np.clip(v + (v - blur) * float(detail), 0.0, 1.0)

            if invert:
                v = 1.0 - v
            out[i] = v

        t = torch.from_numpy(out)
        # підказка, куди ставити рівні: де насправді лежить сюжет
        q = np.percentile(out[0], [2, 25, 50, 75, 98])
        info = (f"frames {n} · {mode}"
                + (f" · near_bias {near_bias}" if mode == "disparity" else "")
                + f" · levels {black_point}–{white_point} · gamma {gamma}"
                + (f" · focus {focus}×{focus_strength}" if focus_strength > 0 else "")
                + f" · clarity {clarity} · detail {detail} · norm. {normalize}"
                + " · output tones p2/25/50/75/98: "
                + "/".join(f"{v:.2f}" for v in q))
        # ГІСТОГРАМА планів: кожен горб = окремий план сцени. Куди ставити focus —
        # видно прямо тут, гадати по повзунку не треба.
        h = np.histogram(out[0], bins=10, range=(0.0, 1.0))[0] / out[0].size
        bars = "".join(" ▁▂▃▄▅▆▇█"[min(8, int(p * 8 / max(h.max(), 1e-6) + 0.5))] for p in h)
        print(f"[LexDepthCurve] {info}")
        print(f"[LexDepthCurve] planes 0.0→1.0: |{bars}|  (put focus on the hump you want)")
        return (t.unsqueeze(-1).repeat(1, 1, 1, 3), t, info)
