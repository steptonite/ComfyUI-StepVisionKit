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

import os

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


def _preview_payload(x, side=384, max_keys=8):
    """Мініатюри СИРОГО депсу (до грейду) для браузера: далі він грейдить сам, без Run.

    Шлемо саме вхід x, а не готовий результат — тоді depth_is, trim, рівні й крива
    перераховуються в браузері миттєво. Для відео — до max_keys рівномірних кадрів
    (менших, щоб повідомлення лишалось ~1 МБ), кожен зі своїми квантилями для
    levels_scope = per frame; таблиці кліпу — спільні.
    """
    import base64
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
        x0, x1 = float(sm.min()), float(sm.max())
        enc = np.round((sm - x0) / max(x1 - x0, 1e-9) * 65535.0).astype("<u2")
        flo, fhi = _qtable(f)
        keys.append({"i": int(i), "x0": x0, "x1": x1, "frame_lo": flo, "frame_hi": fhi,
                     "data": base64.b64encode(enc.tobytes()).decode("ascii")})
    clo, chi = _qtable(x)
    return {"w": int(sm.shape[1]), "h": int(sm.shape[0]), "frames": n,
            "clip_lo": clo, "clip_hi": chi, "keys": keys}


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
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("depth", "depth_mask", "info")
    FUNCTION = "grade"
    CATEGORY = "StepVisionKit/Depth"

    def grade(self, depth, depth_is, look, foreground_pop, blacks, whites, midtones,
              foreground_ceiling, ceiling_from, hero_plane, hero_pop, clarity, texture,
              smooth_banding, banding_edge, trim_far, trim_near, flip, levels_scope,
              deflicker, motion_guard, motion_guide=None):
        import cv2

        x = depth.cpu().numpy().astype(np.float32)
        if x.ndim == 4:
            x = x[..., :3].mean(axis=3)
        n, h, w = x.shape
        video = n > 1
        preview = _preview_payload(x)

        # 1) ВІДСТАНЬ: 0 = найближче
        dist = (1.0 - x) if depth_is == "near is white" else x

        sw, sh, sc = _small_side(h, w)
        small = np.stack([cv2.resize(dist[i], (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)])

        bw = fw = None
        if video and deflicker > 0:
            if motion_guide is not None and motion_guide.shape[0] == n:
                g = motion_guide.cpu().numpy().astype(np.float32)
                g = g[..., :3].mean(axis=3) if g.ndim == 4 else g
                gs = [cv2.resize(g[i], (sw, sh), interpolation=cv2.INTER_AREA) for i in range(n)]
            else:
                gs = [_stretch(s, *_lohi(s, 1.0, 99.0)) for s in small]   # кожен кадр у свій діапазон — потоку так легше
            bw, fw = _flows(gs)

        # «до»: той самий кліп у тій самій шкалі, але ще без замка й без deflicker
        jit_in = _jitter([_stretch(s, *_lohi(small, 1.0, 99.0)) for s in small], bw) if bw else None

        # 🔴 Тут був замок шкали (афінна підгонка кадру до сусіда уздовж потоку) — ПРИБРАНО
        #    16.10.2026 після двох замірів на depth_ant_v3_00001: і ланцюжок, і «стабілізатор
        #    траєкторії» ПОСИЛЮВАЛИ і тремтіння, і пульсацію (1.32 → 2.36). Глобального
        #    «дихання» в DA3 після спільних рівнів майже нема, а підгонка вносить свій шум ±0.02.
        #    Мерехтіння DA3 — ЛОКАЛЬНЕ, його гасить deflicker нижче.
        scale_note = ""

        # 3) РІВНІ по всьому кліпу або по кадру
        clip_lohi = _lohi(dist, trim_near, 100.0 - trim_far) if (levels_scope == "whole clip" and video) else None

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
                s, p = float(hero_pop), float(hero_plane)
                g0 = float(1.0 / (1.0 + np.exp(s * p)))
                g1 = float(1.0 / (1.0 + np.exp(-s * (1.0 - p))))
                v = ((1.0 / (1.0 + np.exp(-s * (v - p)))) - g0) / max(g1 - g0, 1e-6)
                v = np.clip(v, 0.0, 1.0).astype(np.float32)

            # СТЕЛЯ переднього плану: усе світліше за k м'яко лягає під c.
            # c > k: плече з нахилом 1 у точці k (без зламу), найближче = c.
            # c ≤ k: найближче ТЕМНІШЕ за героя — творчий «провал» переднього плану.
            if c < 1.0 and k < 1.0:
                m = v > k
                t = (v[m] - k) / (1.0 - k)
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

        # 4) DEFLICKER: вперед і назад уздовж руху, середнє — без запізнення й без шлейфу в один бік
        if video and deflicker > 0:
            st, gd = float(deflicker), float(motion_guard)
            fwd = np.empty_like(out)
            for t, acc in _temporal_pass(out, bw, st, gd, sc, range(n)):
                fwd[t] = acc
            for t, acc in _temporal_pass(out, fw, st, gd, sc, range(n - 1, -1, -1)):
                fwd[t] = 0.5 * (fwd[t] + acc)
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

        tt = torch.from_numpy(np.ascontiguousarray(out, dtype=np.float32))
        return {"ui": {"svk_depth_preview": [preview]},
                "result": (tt.unsqueeze(-1).repeat(1, 1, 1, 3), tt, info)}


class SVKDepthGradeView:
    """Живе прев'ю для Depth Grade. Під'єднай вхід до виходу depth ноди Depth Grade:
    після одного Run повзунки тієї ноди крутяться тут наживо, без перегенерації.
    Кнопка «Відео» в прев'ю показує кадри результату (deflicker) з швидкістю fps."""
    DESCRIPTION = ("Живе прев'ю Depth Grade: під'єднай до виходу depth, натисни Run один раз — "
                   "далі тягай повзунки Depth Grade і бач зміну одразу. Піпетки й криві пишуть у ті самі повзунки. "
                   "Режим «Відео» відтворює кадри результату з deflicker.")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {"depth": ("IMAGE", {"tooltip": "вихід depth ноди Depth Grade"})},
            "optional": {"fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 1.0,
                                           "tooltip": "швидкість відтворення в прев'ю"})},
        }

    RETURN_TYPES = ()
    FUNCTION = "view"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/Depth"

    def view(self, depth, fps=24.0):
        n = int(depth.shape[0])
        if n <= 1:
            # один кадр — саме прев'ю рахує браузер з мініатюри Depth Grade; тут нічого робити
            return {"ui": {}}

        import uuid
        from PIL import Image
        import folder_paths

        h, w = int(depth.shape[1]), int(depth.shape[2])
        k = max(1, -(-n // 400))                      # кожен k-й кадр, щоб лишилось ≤ 400
        x = depth[::k].detach().float().cpu()
        s = min(1.0, 512.0 / float(max(h, w)))        # довша сторона ≤ 512 — для прев'ю досить
        if s < 1.0:
            nh, nw = max(1, round(h * s)), max(1, round(w * s))
            x = torch.nn.functional.interpolate(x.permute(0, 3, 1, 2), size=(nh, nw), mode="area")
            x = x.permute(0, 2, 3, 1)
        arr = (x.clamp(0.0, 1.0) * 255.0 + 0.5).to(torch.uint8).numpy()

        out_dir = os.path.join(folder_paths.get_temp_directory(), "svk_view")
        os.makedirs(out_dir, exist_ok=True)
        prefix = uuid.uuid4().hex[:8]
        frames = []
        for j, fr in enumerate(arr):
            if fr.shape[-1] == 1:
                fr = np.repeat(fr, 3, axis=-1)
            name = f"{prefix}_{j:04d}.jpg"
            Image.fromarray(np.ascontiguousarray(fr[..., :3])).save(os.path.join(out_dir, name), quality=85)
            frames.append({"filename": name, "subfolder": "svk_view", "type": "temp"})
        return {"ui": {"svk_frames": frames, "svk_fps": [float(fps)], "svk_stride": [k]}}


class SVKDepthGradeLive(SVKDepthGrade):
    """Стара назва (16.10.2026): пульт переїхав у SVKDepthGradeView. Лишено, щоб
    збережені воркфлоу не червоніли; у пошуку не видно."""
    DEPRECATED = True
