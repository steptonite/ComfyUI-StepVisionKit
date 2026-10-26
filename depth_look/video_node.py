"""Depth Look · відео: бере `grade` (DL_GRADE) зі статичної/простої ноди й додає те, чого там нема:
час (fps, start_time), анімацію (look_params.ANIM_KEYS), фінальний розмір і швидке превʼю.
Повзунки анімації генеруються зі SPEC (look_params), у порядку GROUP_ORDER - нових ключів руками додавати не треба."""
import hashlib
import json
import os
import time

import cv2
import numpy as np

import look_api
import look_params
import depth_motion

from .core import (SIZE_ORIGINAL, OUTPUT_SIZES, GRADE_TYPE, PREVIEW_SIDE, PREVIEW_SIDES, parse_output_size, parse_preview_side, check_inputs,
                   legacy, validate_legacy, params_from_grade, out_shape, watermark)
from .render_proc import tensors_to_numpy, make_progress, render, to_result, stage
from ..media import color as C
try:
    from comfy_execution.graph_utils import ExecutionBlocker
except Exception:            # старий Comfy без блокера: лишаємо None (ланцюг упаде з явною помилкою, а не мовчки зарендерить)
    ExecutionBlocker = None

QUALITY_PREVIEW = "preview (fast)"
QUALITY_FINAL = "final"
HDR_OFF = "SDR or HLG/PQ as encoded"
HDR_LINEAR = "linear (keeps highlights)"
_DEF = None


def _anim_specs():
    """[(ключ, lo, hi, step, група, підпис, int?)] для ANIM_KEYS без `time` (його рахує нода: start_time + i/fps)."""
    rows = [s for s in look_params.SPEC if s[0] in look_params.ANIM_KEYS and s[0] != "time"]
    order = {g: i for i, g in enumerate(look_params.GROUP_ORDER)}
    rows.sort(key=lambda s: order.get(s[4], 99))                  # sort стабільний: усередині групи - порядок SPEC
    return [(s[0], s[1], s[2], s[3], s[4], s[5], (s[0] in look_params.INT_KEYS) or (len(s) > 6 and s[6] == "i"))
            for s in rows]


STRIP_MAX = 12          # скільки кадрів-превʼю показує стрічка в ноді
STRIP_SIDE = 960        # довга сторона кадру стрічки (на картці ~340 px; решта - для повноекранного перегляду)


def strip_path(unique_id, i):
    return os.path.join(look_api.CACHE_DIR, "comfy_strip_%s_%d.jpg" % (look_api._safe_id(unique_id), int(i)))


def strip_meta_path(unique_id):
    return os.path.join(look_api.CACHE_DIR, "comfy_strip_%s_meta.json" % look_api._safe_id(unique_id))


def save_strip(unique_id, arr):
    """Кадри-превʼю (B,H,W,3 float) -> до STRIP_MAX рівномірних jpg-мініатюр у кеші. Повертає індекси кадрів у arr.
    Лише локальний диск; сервер віддає їх маршрутом /depth_look/strip/<id>/<i>."""
    B = arr.shape[0]
    idx = sorted(set(int(round(v)) for v in np.linspace(0, B - 1, min(STRIP_MAX, B))))
    os.makedirs(look_api.CACHE_DIR, exist_ok=True)
    for n, k in enumerate(idx):
        im = np.clip(arr[k][..., :3] * 255.0 + 0.5, 0, 255).astype(np.uint8)
        h, w = im.shape[:2]
        f = STRIP_SIDE / float(max(h, w))
        if f < 1.0:
            im = cv2.resize(im, (max(2, int(round(w * f))), max(2, int(round(h * f)))), interpolation=cv2.INTER_AREA)
        cv2.imwrite(strip_path(unique_id, n), cv2.cvtColor(im, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 86])
    return idx


# ---- кеш готових кліпів (26.10.2026): повторний «Render final» з новим render_token, коли нічого, що впливає на кадри, не змінилось
# (змінили лише Save/даунстрім), не рахує кліп заново. По одному запису на (нода, превʼю|фінал). Ключ - хеш параметрів + відбиток тензорів.
CACHE_MAX_BYTES = 1 << 30      # масив у RAM (не mmap воркера) більший за це в кеш не кладемо
_CACHE = {}                    # (str(unique_id), "preview"|"final") -> (ключ, ndarray, play, ui)


def _tensor_sig(t):
    """Відбиток тензора без копії кліпу: форма, тип, пристрій, адреса й версія + сітка ~32x32 пікселів у ≤5 кадрах."""
    h = hashlib.blake2b(digest_size=16)
    try:
        ver = int(t._version)
    except Exception:                                      # inference-тензори (DA3 у Comfy) лічильника версій не мають
        ver = 0
    h.update(repr((tuple(t.shape), str(t.dtype), str(t.device), int(t.data_ptr()), ver)).encode())
    B = int(t.shape[0])
    for i in sorted({0, B // 4, B // 2, (3 * B) // 4, B - 1}):
        f = t[i]
        if f.dim() >= 2:
            f = f[::max(1, int(f.shape[0]) // 32), ::max(1, int(f.shape[1]) // 32)]
        h.update(f.detach().float().cpu().numpy().tobytes())
    return h.hexdigest()


def _job_key(image, depth, *parts):
    """Ключ завдання або None (не вдалось порахувати -> кешу немає, рахуємо як раніше)."""
    try:
        h = hashlib.blake2b(digest_size=20)
        h.update(json.dumps(parts, sort_keys=True, default=str, ensure_ascii=False).encode())
        h.update(_tensor_sig(image).encode())
        h.update(_tensor_sig(depth).encode())
        return h.hexdigest()
    except Exception as e:
        print("[DepthLookVideo] cache key not built (%s) - rendering without cache" % e)
        return None


def _cache_get(uid, kind, key):
    e = _CACHE.get((str(uid), kind))
    return e if (e is not None and key is not None and e[0] == key) else None


def _cache_put(uid, kind, key, arr, play, ui):
    k = (str(uid), kind)
    _CACHE.pop(k, None)
    if key is None or uid is None:
        return
    if getattr(arr, "filename", None) is None and arr.nbytes > CACHE_MAX_BYTES:      # не файлова: тримати в RAM задорого
        return
    _CACHE[k] = (key, arr, play, ui)


def _shrink(a, side):
    """(B,H,W[,C]) -> довга сторона <= side (INTER_AREA), float32; менше вже - як є (без копії)."""
    H, W = int(a.shape[1]), int(a.shape[2])
    s = float(side) / max(H, W)
    if s >= 1.0:
        return a
    w, h = max(1, round(W * s)), max(1, round(H * s))
    out = np.empty((a.shape[0], h, w) + tuple(a.shape[3:]), np.float32)
    for i in range(a.shape[0]):
        out[i] = cv2.resize(np.asarray(a[i], np.float32), (w, h), interpolation=cv2.INTER_AREA).reshape(out.shape[1:])
    return out


class DepthLookVideo:
    @classmethod
    def INPUT_TYPES(cls):
        global _DEF
        _DEF = _DEF or look_params.defaults()
        req = {
            "image": ("IMAGE", {"tooltip": "Video frames (B,H,W,3) — from Load Video · Pro or any image node."}),
            "depth": ("IMAGE", {"tooltip": "Depth, bright = near: one per frame (or one for the whole clip) — from Depth Grade, Depth Receiver or Video Depth · DA3 stream."}),
            "grade": (GRADE_TYPE, {"tooltip": "The grade output of Depth Look · Full or Depth Look · Simple. This node adds the animation below."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Video frame rate: frame i gets the time start_time + i/fps."}),
            "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "At which second to start the animation clock."}),
        }
        for key, lo, hi, step, grp, label, is_int in _anim_specs():       # анімація: ключі як у SPEC/пресетах
            d = _DEF.get(key, lo)
            tip = "%s · %s" % (grp, label)
            if is_int:
                req[key] = ("INT", {"default": int(round(d)), "min": int(lo), "max": int(hi), "step": 1, "tooltip": tip})
            else:
                req[key] = ("FLOAT", {"default": float(d), "min": float(lo), "max": float(hi), "step": float(step), "tooltip": tip})
        req.update({
            "output_size": (OUTPUT_SIZES, {"default": SIZE_ORIGINAL, "tooltip": "Final frame size (long side, px). The first option keeps the input size. Ignored in preview (it uses preview_side)."}),
            "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Working resolution: the look is computed on a frame with this long side and then upscaled to the final size. 0 = compute directly at final size. For long clips on 8 GB machines."}),
            "quality": ([QUALITY_PREVIEW, QUALITY_FINAL], {"default": QUALITY_PREVIEW, "tooltip": "Preview: long side = preview_side, every Nth frame, PREVIEW watermark - to check how the gradient moves. Final: size from output_size, all frames."}),
            "preview_every": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1, "tooltip": "Preview only: take every Nth frame. To watch at real speed, set the playback rate to fps/N (second output)."}),
        })
        opt = {"fps_in": ("FLOAT", {"forceInput": True, "min": 1.0, "max": 240.0, "step": 0.01,
                                    "tooltip": "Frame rate — connect the fps output of Load Video · Pro. When connected, it replaces the fps field."})}
        for key, kind, d, lo, hi, step, label in depth_motion.SPEC:     # «Рух по глибині» (depth_motion.py): сили за замовч. 0
            tip = "Depth motion · " + label
            if isinstance(kind, list):
                opt[key] = (kind, {"default": d, "tooltip": tip})
            elif kind == "c":
                opt[key] = ("STRING", {"default": d, "multiline": False, "tooltip": tip + " (hex, e.g. #7fd4ff)"})
            elif kind == "i":
                opt[key] = ("INT", {"default": int(d), "min": int(lo), "max": int(hi), "step": int(step), "tooltip": tip})
            else:
                opt[key] = ("FLOAT", {"default": float(d), "min": float(lo), "max": float(hi), "step": float(step), "tooltip": tip})
        # у кінці: значення віджетів у збереженому графі позиційні, нове поле в середині зсуває старі графи
        opt["hdr"] = ([HDR_OFF, HDR_LINEAR], {"default": HDR_OFF,
                              "tooltip": "HDR clip: Load Video (Pro) 'as encoded' needs nothing here - save with input_curve HLG/PQ (as encoded). "
                                                "Load Video (Pro) 'linear' -> pick linear: highlights above 1 go through the look instead of clipping; "
                                                "output stays linear, save with input_curve linear."})
        opt["final_out"] = ("BOOLEAN", {"default": False, "tooltip": "Set by the UI: the final output is connected, so a preview run also renders the final."})
        # render-gate (25.10.2026): теж лише в кінці, з тієї ж причини
        opt["auto_render"] = ("BOOLEAN", {"default": False, "tooltip": "Off: every run only renders the preview and the outputs stay blocked (Video Combine / Save do not run) - press 'Render final' on the node. "
                                                                       "On: every run also renders the final and passes it on, with the settings the node already has."})
        opt["render_token"] = ("STRING", {"default": "", "multiline": False, "tooltip": "Set by the 'Render final' button (one-shot). Do not edit."})
        # preview_side (26.10.2026): теж лише в кінці - значення віджетів у старих графах позиційні
        opt["preview_side"] = (PREVIEW_SIDES, {"default": str(PREVIEW_SIDE), "tooltip": "Preview only: long side of the preview frames, px. Smaller = faster, bigger = more detail."})
        return {"required": req, "optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

    _consumed = {}           # unique_id -> останній використаний render_token (one-shot: той самий токен фінал вдруге не запускає)

    DESCRIPTION = "Takes a grade from Depth Look and applies it to video: timing, animation, final size and a fast preview."
    RETURN_TYPES = ("IMAGE", "FLOAT", "IMAGE")
    RETURN_NAMES = ("preview", "play_fps", "final")
    OUTPUT_NODE = True       # стрічка превʼю в ноді має рахуватись і без підключеного виходу
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    @classmethod
    def VALIDATE_INPUTS(cls, quality=None, output_size=None):
        return validate_legacy(quality=(quality, [QUALITY_PREVIEW, QUALITY_FINAL]), output_size=(output_size, OUTPUT_SIZES))

    @classmethod
    def _gate_open(cls, unique_id, auto_render, render_token):
        """Фінал дозволено: ввімкнено auto_render або токен кнопки свіжий (≠ спожитому для цієї ноди). Токен споживається."""
        if auto_render:
            return True
        tok = str(render_token or "")
        key = str(unique_id)
        if tok and cls._consumed.get(key) != tok:
            cls._consumed[key] = tok
            return True
        return False

    def run(self, image, depth, grade, fps, start_time, output_size, max_side, quality, preview_every,
            unique_id=None, fps_in=None, hdr=HDR_OFF, final_out=False, auto_render=False, render_token="",
            preview_side=str(PREVIEW_SIDE), **anim):
        if fps_in is not None:                                 # сокет fps (напр. від Video Info) має перевагу над полем
            fps = float(fps_in)
        want_final = str(legacy(quality)).startswith("final") or bool(final_out)
        side = parse_preview_side(preview_side)
        job = (image, depth, grade, fps, start_time, output_size, max_side, preview_every, unique_id, hdr, anim)
        # (1) превʼю - завжди першим; стрічка йде в ноду одразу, ще до фіналу. Ті самі входи, що й минулого разу -> з кешу
        pkey = _job_key(image, depth, "preview", grade, float(fps), float(start_time), hdr, anim, int(preview_every), side)
        hit = _cache_get(unique_id, "preview", pkey)
        if hit and hit[3].get("depth_look_strip") and not os.path.exists(strip_path(unique_id, 0)):
            hit = None                                         # jpg стрічки з кешу диска зникли - перегенерувати
        if hit:
            prev_arr, play, ui = hit[1], hit[2], hit[3]
            stage(unique_id, "cached")
            print("[DepthLookVideo] preview unchanged - taken from cache")
        else:
            prev_arr, play, ui = self._render(*job, side, preview=True)
            _cache_put(unique_id, "preview", pkey, prev_arr, play, ui)
        self._push_strip(unique_id, ui)
        # (2) фінал - лише за кнопкою (свіжий токен) або з auto_render; інакше всі виходи блокуються
        if not self._gate_open(unique_id, auto_render, render_token):
            print("[DepthLookVideo] preview done; final not requested (press 'Render final' on the node or enable auto_render) - outputs blocked")
            blk = ExecutionBlocker(None) if ExecutionBlocker else None
            return {"ui": ui, "result": (blk, blk, blk)}
        out_prev = to_result(prev_arr, image)
        del prev_arr
        if not want_final:                                     # дозволено, але ні image=final, ні сокет final не потрібні: віддати превʼю
            return {"ui": ui, "result": (out_prev, play, out_prev)}
        # фінал не залежить від preview_every/preview_side/quality/render_token: змінилось лише це (або лише Save) - кліп той самий
        fkey = _job_key(image, depth, "final", grade, float(fps), float(start_time), parse_output_size(output_size),
                        int(max_side or 0), hdr, anim)
        fhit = _cache_get(unique_id, "final", fkey)
        if fhit:
            fin_arr = fhit[1]
            stage(unique_id, "cached")
            print("[DepthLookVideo] final unchanged - taken from cache (%d frame(s))" % fin_arr.shape[0])
        else:
            _CACHE.pop((str(unique_id), "final"), None)        # старий фінал вже не знадобиться: не тримати два кліпи разом
            with look_api.final_render():
                fin_arr = self._render(*job, side, preview=False)[0]
            _cache_put(unique_id, "final", fkey, fin_arr, None, None)
        fin = to_result(fin_arr, image)
        del fin_arr
        if str(legacy(quality)).startswith("preview"):         # image = превʼю, final = фінал (як і раніше)
            return {"ui": ui, "result": (out_prev, play, fin)}
        return {"ui": ui, "result": (fin, float(fps), fin)}

    @staticmethod
    def _push_strip(unique_id, ui):
        if unique_id is None or not ui.get("depth_look_strip"):
            return
        try:
            from server import PromptServer
            PromptServer.instance.send_sync("svk.strip", {"node": str(unique_id), "strip": ui["depth_look_strip"][0]})
        except Exception:
            pass

    def _render(self, image, depth, grade, fps, start_time, output_size, max_side, preview_every, unique_id, hdr, anim, preview_side, preview):
        g = grade
        params_from_grade(g)                                   # ранній зрозумілий виняток, якщо підключили не те
        stage(unique_id, "read", n=int(image.shape[0]))
        frames, dep = tensors_to_numpy(image, depth)
        step = max(1, int(preview_every)) if preview else 1
        if step > 1:                                           # прорідження ДО збереження завдання: воркер не бачить зайвого
            frames = frames[::step]
            if dep.shape[0] > 1:
                dep = dep[::step]
        out_size = preview_side if preview else parse_output_size(output_size)
        ms = 0 if preview else int(max_side or 0)
        if preview:                                            # превʼю: кадри й глибину зменшити ДО завдання воркеру - інакше на диск
            frames, dep = _shrink(frames, preview_side), _shrink(dep, preview_side)   # іде повний 1080p кліп (226 кадрів ≈ 5.6 ГБ)
        for w in check_inputs(frames, dep, ms, out_size):
            print("[DepthLookVideo] ⚠ " + w)
        kw = dict(grade=g, clean_depth=bool(g.get("clean_depth", True)), light_is_near=bool(g.get("light_is_near", True)),
                  depth_levels=g.get("depth_levels", "clip"), max_side=ms, output_size=out_size, fps=float(fps),
                  start_time=float(start_time), anim={k: anim[k] for k in look_params.ANIM_KEYS if k in anim},
                  frame_step=step, static=False)
        motion = {k: anim[k] for k in depth_motion.KEYS if k in anim}
        if depth_motion.active(motion):                        # старі графи без dm_*: ключа motion у завданні немає взагалі
            kw["motion"] = motion
        hdr_lin = hdr == HDR_LINEAR
        if hdr_lin:                                            # лук рахується на HLG-сигналі, назад у linear - наприкінці
            kw["hdr_linear"] = True
        t0 = time.time()
        arr = render(frames, dep, kw, make_progress(frames.shape[0], unique_id),
                     out_shape(frames.shape[0], frames.shape[1], frames.shape[2], out_size))
        mix = float(min(max(g.get("look_mix", 1.0), 0.0), 1.0))      # «сила луку» з простої ноди
        if mix < 1.0:
            from .simple import mix_with_original
            stage(unique_id, "mix")
            arr = mix_with_original(arr, frames, mix, C.hlg_encode if hdr_lin else None)
        ui = {}
        if preview and unique_id is not None:                  # стрічка кадрів у ноді (той самий стиль, що й плитки Simple)
            try:
                stage(unique_id, "strip")
                idx = save_strip(unique_id, arr)
                meta = {"n": len(idx), "frames": [int(k * step) for k in idx], "total": int(arr.shape[0]),
                        "play_fps": float(fps) / step, "every": step, "stamp": time.time_ns()}
                json.dump(meta, open(strip_meta_path(unique_id), "w"))      # щоб стрічка пережила перезавантаження сторінки
                ui = {"depth_look_strip": [meta]}
            except Exception as e:
                print("[DepthLookVideo] preview strip not written: %s" % e)
        if preview:
            watermark(arr, "PREVIEW %d, every %d - set quality=final" % (preview_side, step))
        if hdr_lin:                                            # стрічка вище - з сигналу (HLG дивиться і як SDR)
            for i in range(arr.shape[0]):
                arr[i] = C.hlg_decode(arr[i])
        play = float(fps) / step
        stage(unique_id, "done", secs=round(time.time() - t0, 1))
        print("[DepthLookVideo] %s: %d frame(s) %dx%d in %.1f s, play at %.2f fps, look strength %.2f" % (
            QUALITY_PREVIEW if preview else QUALITY_FINAL, arr.shape[0], arr.shape[2], arr.shape[1], time.time() - t0, play, mix))
        return arr, play, ui
