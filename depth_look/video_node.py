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

from .core import (SIZE_ORIGINAL, OUTPUT_SIZES, GRADE_TYPE, PREVIEW_SIDE, parse_output_size, check_inputs,
                   params_from_grade, out_shape, watermark)
from .render_proc import tensors_to_numpy, make_progress, render, to_result

QUALITY_PREVIEW = "превʼю (швидко)"
QUALITY_FINAL = "фінал"
_DEF = None


def _anim_specs():
    """[(ключ, lo, hi, step, група, підпис, int?)] для ANIM_KEYS без `time` (його рахує нода: start_time + i/fps)."""
    rows = [s for s in look_params.SPEC if s[0] in look_params.ANIM_KEYS and s[0] != "time"]
    order = {g: i for i, g in enumerate(look_params.GROUP_ORDER)}
    rows.sort(key=lambda s: order.get(s[4], 99))                  # sort стабільний: усередині групи - порядок SPEC
    return [(s[0], s[1], s[2], s[3], s[4], s[5], (s[0] in look_params.INT_KEYS) or (len(s) > 6 and s[6] == "i"))
            for s in rows]


STRIP_MAX = 12          # скільки кадрів-превʼю показує стрічка в ноді
STRIP_SIDE = 320        # довга сторона мініатюри стрічки


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
            "image": ("IMAGE", {"tooltip": "Кадри відео (B,H,W,3)."}),
            "depth": ("IMAGE", {"tooltip": "Глибина: по одній на кадр (або одна на весь кліп). Світле = близько."}),
            "grade": (GRADE_TYPE, {"tooltip": "Грейд зі статичної ноди (Depth Look) або Depth Look · простий. Анімацію нижче додає ця нода."}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.01, "tooltip": "Частота кадрів відео: кадр i отримує час start_time + i/fps."}),
            "start_time": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 100000.0, "step": 0.01, "tooltip": "З якої секунди почати відлік анімації."}),
        }
        for key, lo, hi, step, grp, label, is_int in _anim_specs():       # анімація: ключі як у SPEC/пресетах
            d = _DEF.get(key, lo)
            tip = "%s · %s" % (grp, label)
            if is_int:
                req[key] = ("INT", {"default": int(round(d)), "min": int(lo), "max": int(hi), "step": 1, "tooltip": tip})
            else:
                req[key] = ("FLOAT", {"default": float(d), "min": float(lo), "max": float(hi), "step": float(step), "tooltip": tip})
        req.update({
            "output_size": (OUTPUT_SIZES, {"default": SIZE_ORIGINAL, "tooltip": "Фінальний розмір кадру (довга сторона, px). «як оригінал» = розмір входу. У превʼю не діє (там %d)." % PREVIEW_SIDE}),
            "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Робоча роздільність: look рахується на кадрі з такою довгою стороною і росте до фінального. 0 = одразу у фінальному. Для довгих кліпів на 8 ГБ."}),
            "quality": ([QUALITY_PREVIEW, QUALITY_FINAL], {"default": QUALITY_PREVIEW, "tooltip": "Превʼю: довга сторона %d, кожен N-й кадр, напис PREVIEW - щоб глянути, як рухається градієнт. Фінал: розмір за output_size, усі кадри." % PREVIEW_SIDE}),
            "preview_every": ("INT", {"default": 3, "min": 1, "max": 60, "step": 1, "tooltip": "Лише в превʼю: брати кожен N-й кадр. Для перегляду в реальному темпі став частоту відтворення fps/N (другий вихід)."}),
        })
        return {"required": req, "hidden": {"unique_id": "UNIQUE_ID"}}

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "play_fps")
    OUTPUT_NODE = True       # стрічка превʼю в ноді має рахуватись і без підключеного виходу
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    def run(self, image, depth, grade, fps, start_time, output_size, max_side, quality, preview_every,
            unique_id=None, **anim):
        preview = str(quality).startswith("прев")
        g = grade
        params_from_grade(g)                                   # ранній зрозумілий виняток, якщо підключили не те
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
        t0 = time.time()
        arr = render(frames, dep, kw, make_progress(frames.shape[0]),
                     out_shape(frames.shape[0], frames.shape[1], frames.shape[2], out_size))
        mix = float(min(max(g.get("look_mix", 1.0), 0.0), 1.0))      # «сила луку» з простої ноди
        if mix < 1.0:
            from .simple import mix_with_original
            arr = mix_with_original(arr, frames, mix)
        ui = {}
        if preview and unique_id is not None:                  # стрічка кадрів у ноді (той самий стиль, що й плитки Simple)
            try:
                idx = save_strip(unique_id, arr)
                meta = {"n": len(idx), "frames": [int(k * step) for k in idx], "total": int(arr.shape[0]),
                        "play_fps": float(fps) / step, "every": step, "stamp": time.time_ns()}
                json.dump(meta, open(strip_meta_path(unique_id), "w"))      # щоб стрічка пережила перезавантаження сторінки
                ui = {"depth_look_strip": [meta]}
            except Exception as e:
                print("[DepthLookVideo] стрічку превʼю не записано: %s" % e)
        if preview:
            watermark(arr, "PREVIEW %d, every %d - set quality=final" % (PREVIEW_SIDE, step))
        play = float(fps) / step
        print("[DepthLookVideo] %s: %d кадр(ів) %dx%d за %.1f с, грати з %.2f fps, сила луку %.2f" % (
            QUALITY_PREVIEW if preview else QUALITY_FINAL, arr.shape[0], arr.shape[2], arr.shape[1], time.time() - t0, play, mix))
        return {"ui": ui, "result": (to_result(arr, image), play)}
