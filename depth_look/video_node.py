"""Depth Look · відео: бере `grade` (DL_GRADE) зі статичної/простої ноди й додає те, чого там нема:
час (fps, start_time), анімацію (look_params.ANIM_KEYS), фінальний розмір і швидке превʼю.
Повзунки анімації генеруються зі SPEC (look_params), у порядку GROUP_ORDER - нових ключів руками додавати не треба."""
import json
import os
import time

import cv2
import numpy as np

import look_api
import look_params
import depth_motion

from .core import (SIZE_ORIGINAL, OUTPUT_SIZES, GRADE_TYPE, PREVIEW_SIDE, parse_output_size, check_inputs,
                   legacy, validate_legacy, params_from_grade, out_shape, watermark)
from .render_proc import tensors_to_numpy, make_progress, render, to_result, stage
from ..media import color as C

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


class DepthLookVideo:
    @classmethod
    def INPUT_TYPES(cls):
        global _DEF
        _DEF = _DEF or look_params.defaults()
        req = {
            "image": ("IMAGE", {"tooltip": "Video frames (B,H,W,3)."}),
            "depth": ("IMAGE", {"tooltip": "Depth: one per frame (or one for the whole clip). Bright = near."}),
            "grade": (GRADE_TYPE, {"tooltip": "Grade from a static node (Depth Look) or Depth Look · Simple. This node adds the animation below."}),
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
            "output_size": (OUTPUT_SIZES, {"default": SIZE_ORIGINAL, "tooltip": "Final frame size (long side, px). The first option keeps the input size. Ignored in preview (it uses %d)." % PREVIEW_SIDE}),
            "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Working resolution: the look is computed on a frame with this long side and then upscaled to the final size. 0 = compute directly at final size. For long clips on 8 GB machines."}),
            "quality": ([QUALITY_PREVIEW, QUALITY_FINAL], {"default": QUALITY_PREVIEW, "tooltip": "Preview: long side %d, every Nth frame, PREVIEW watermark - to check how the gradient moves. Final: size from output_size, all frames." % PREVIEW_SIDE}),
            "preview_every": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1, "tooltip": "Preview only: take every Nth frame. To watch at real speed, set the playback rate to fps/N (second output)."}),
        })
        opt = {"fps_in": ("FLOAT", {"forceInput": True, "min": 1.0, "max": 240.0, "step": 0.01,
                                    "tooltip": "Frame rate from another node (e.g. fps from Video Info / Load Video). When connected, it replaces the fps field."}),
               "hdr": ([HDR_OFF, HDR_LINEAR], {"default": HDR_OFF,
                       "tooltip": "HDR clip: Load Video (Pro) 'as encoded' needs nothing here - save with input_curve HLG/PQ (as encoded). "
                                  "Load Video (Pro) 'linear' -> pick linear: highlights above 1 go through the look instead of clipping; "
                                  "output stays linear, save with input_curve linear."})}
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
        return {"required": req, "optional": opt, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "play_fps")
    OUTPUT_NODE = True       # стрічка превʼю в ноді має рахуватись і без підключеного виходу
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    @classmethod
    def VALIDATE_INPUTS(cls, quality=None, output_size=None):
        return validate_legacy(quality=(quality, [QUALITY_PREVIEW, QUALITY_FINAL]), output_size=(output_size, OUTPUT_SIZES))

    def run(self, image, depth, grade, fps, start_time, output_size, max_side, quality, preview_every,
            unique_id=None, fps_in=None, hdr=HDR_OFF, **anim):
        if fps_in is not None:                                 # сокет fps (напр. від Video Info) має перевагу над полем
            fps = float(fps_in)
        preview = str(legacy(quality)).startswith("preview")
        g = grade
        params_from_grade(g)                                   # ранній зрозумілий виняток, якщо підключили не те
        stage(unique_id, "read", n=int(image.shape[0]))
        frames, dep = tensors_to_numpy(image, depth)
        step = max(1, int(preview_every)) if preview else 1
        if step > 1:                                           # прорідження ДО збереження завдання: воркер не бачить зайвого
            frames = frames[::step]
            if dep.shape[0] > 1:
                dep = dep[::step]
        out_size = PREVIEW_SIDE if preview else parse_output_size(output_size)
        ms = 0 if preview else int(max_side or 0)
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
            watermark(arr, "PREVIEW %d, every %d - set quality=final" % (PREVIEW_SIDE, step))
        if hdr_lin:                                            # стрічка вище - з сигналу (HLG дивиться і як SDR)
            for i in range(arr.shape[0]):
                arr[i] = C.hlg_decode(arr[i])
        play = float(fps) / step
        stage(unique_id, "done", secs=round(time.time() - t0, 1))
        print("[DepthLookVideo] %s: %d frame(s) %dx%d in %.1f s, play at %.2f fps, look strength %.2f" % (
            QUALITY_PREVIEW if preview else QUALITY_FINAL, arr.shape[0], arr.shape[2], arr.shape[1], time.time() - t0, play, mix))
        return {"ui": ui, "result": (to_result(arr, image), play)}
