"""Depth Look · простий: пресет + для кожної групи ефектів «увімк + сила». Без вбудованого тюнера.
Групи й порядок - з look_params (GROUP_ORDER/GROUP_POWER/GROUP_HINT), тож нода сама підлаштовується під рушій.
Виходи: IMAGE і `grade` (DL_GRADE) - його приймають Depth Look (розширена) і Depth Look · відео."""
import json
import time

import cv2
import numpy as np

import look_api
import look_params

from .core import (DEFAULT_PRESET, SIZE_ORIGINAL, OUTPUT_SIZES, GRADE_TYPE, parse_output_size, check_inputs,
                   resolve_params, static_params, make_grade, out_shape)
from .render_proc import tensors_to_numpy, make_progress, render, to_result

# (група, ключ сили, id віджета). Сила групи = перший ключ GROUP_POWER; множник 0..2 лягає на нього
POWER_GROUPS = [(g, look_params.GROUP_POWER[g][0]) for g in look_params.GROUP_ORDER if g in look_params.GROUP_POWER]
_SPEC = {s[0]: s for s in look_params.SPEC}


# «типова» сила для групи, якої в пресеті нема (там 0): тап по чіпу вмикає її з цим значенням (extra_json).
# Без цього половина ефектів у більшості пресетів була б «мертвою» кнопкою.
ADD_DEFAULT = {"silhouette": 0.6, "warp": 0.12, "smear": 0.5, "trail": 0.5, "stretch_opacity": 0.5, "haze": 0.5,
               "glow": 0.6, "spark": 0.5, "halation_strength": 1.0, "grain_amount": 20.0}

# людські назви груп для чіпів (без жаргону); повна назва групи лишається підказкою
CHIP_LABEL = {"Силует": "Силует", "Деформація · хвиля": "Марево", "Деформація · розмаз": "Розмаз", "Шлейф": "Шлейф",
              "Копії в глибину": "Тунель", "Серпанок": "Серпанок", "Світло · глоу": "Сяйво",
              "Світло · точковий глоу": "Зірки", "Світло · halation": "Ореол", "Зерно": "Зерно"}


def parse_extra(s):
    """extra_json -> {ключ_сили: абсолютна сила}; сміття ігнорується (нода не падає через службове поле)."""
    try:
        j = json.loads(s) if (s or "").strip() else {}
    except Exception:
        return {}
    keys = {k for _g, k in POWER_GROUPS}
    return {k: float(v) for k, v in j.items() if k in keys and isinstance(v, (int, float))} if isinstance(j, dict) else {}


def apply_groups(p, switches, extra=None):
    """p - плоскі параметри пресету; switches {ключ_сили: (увімк, множник)} -> нові параметри.
    extra {ключ: сила} - групи, яких у пресеті нема, але користувач їх додав (база замість значення пресету).
    Вимкнено або множник 0 -> сила групи = 0 (рушій такі гілки пропускає); 1.0 -> без змін (бітово);
    інакше сила × множник у межах повзунка зі SPEC."""
    q = dict(p)
    extra = extra or {}
    for _g, key in POWER_GROUPS:
        on, mul = switches.get(key, (True, 1.0))
        if key in extra and float(q.get(key, 0.0)) == 0.0:
            q[key] = float(extra[key])
        if not on or float(mul) <= 0.0:
            q[key] = 0.0
        elif float(mul) != 1.0:
            lo, hi = _SPEC[key][1], _SPEC[key][2]
            q[key] = float(min(max(float(q[key]) * float(mul), lo), hi))
    return q


class DepthLookSimple:
    @classmethod
    def INPUT_TYPES(cls):
        presets = look_params.list_presets()
        req = {
            "image": ("IMAGE", {"tooltip": "Кадр або батч кадрів."}),
            "depth": ("IMAGE", {"tooltip": "Карта глибини: 1 на весь батч або по одній на кадр. Світле = близько."}),
            "preset": (presets, {"default": DEFAULT_PRESET if DEFAULT_PRESET in presets else presets[0],
                                 "tooltip": "Готовий лук. Нижче - кожну групу ефектів можна вимкнути або підсилити/послабити."}),
        }
        for g, key in POWER_GROUPS:
            hint = look_params.GROUP_HINT.get(g, "")
            req[key + "_on"] = ("BOOLEAN", {"default": True, "label": g + " · увімк", "tooltip": hint})
            req[key + "_power"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "label": g + " · сила ×",
                                             "tooltip": "Множник сили групи: 1 = як у пресеті, 0 = вимкнено (не рахується), 2 = вдвічі. " + hint})
        return {
            "required": req,
            "optional": {
                "output_size": (OUTPUT_SIZES, {"default": SIZE_ORIGINAL, "tooltip": "Фінальний розмір кадру (довга сторона, px). «як оригінал» = розмір входу."}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Робоча роздільність: рахується на кадрі з такою довгою стороною і росте до фінального. 0 = одразу у фінальному. Для довгих кліпів на 8 ГБ."}),
                "clean_depth": ("BOOLEAN", {"default": True, "tooltip": "Сира глибина чиститься (кадр - напрямна). Вимкни, якщо глибина вже чиста 0..1."}),
                "depth_light_is_near": ("BOOLEAN", {"default": True, "tooltip": "Світле = близько. Вимкни, якщо нода глибини віддає навпаки."}),
                # нові входи - в кінець, щоб widgets_values старих workflow не зсунулись
                "look_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "label": "сила луку",
                                       "tooltip": "Загальний міксер: 0 = оригінал, 1 = повний лук. Діє на вихід image (вихід grade його не несе)."}),
                "extra_json": ("STRING", {"multiline": False, "default": "{}"}),   # службове: ефекти, додані поверх пресету (пише js)
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", GRADE_TYPE)
    RETURN_NAMES = ("image", "grade")
    OUTPUT_NODE = True            # граф запускається й без Save/Preview; кадр для живого превʼю картки пишемо в кеш
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    def run(self, image, depth, preset, output_size=SIZE_ORIGINAL, max_side=0, clean_depth=True,
            depth_light_is_near=True, look_mix=1.0, extra_json="{}", unique_id=None, **switches):
        sw = {key: (bool(switches.get(key + "_on", True)), float(switches.get(key + "_power", 1.0)))
              for _g, key in POWER_GROUPS}
        p = static_params(apply_groups(resolve_params(preset), sw, parse_extra(extra_json)))
        g = make_grade(p, preset, clean_depth, depth_light_is_near, "clip")
        frames, dep = tensors_to_numpy(image, depth)
        ui = {}
        if unique_id is not None:          # кадри для живого превʼю картки (кеш під ключем s<id>: не плутається з DepthLook)
            try:
                meta = look_api.save_node_cache(node_key(unique_id), frames, dep, clean_depth, depth_light_is_near,
                                                depth_levels="clip")
                ui = {"depth_look_simple": [{"frames": meta["indices"], "stamp": meta["stamp"]}]}
            except Exception as e:
                print("[DepthLookSimple] кеш превʼю не записано: %s" % e)
        out_size = parse_output_size(output_size)
        for w in check_inputs(frames, dep, max_side, out_size):
            print("[DepthLookSimple] ⚠ " + w)
        kw = dict(grade=g, clean_depth=bool(clean_depth), light_is_near=bool(depth_light_is_near),
                  max_side=int(max_side or 0), depth_levels="clip", output_size=out_size, static=True)
        t0 = time.time()
        arr = render(frames, dep, kw, make_progress(frames.shape[0]),
                     out_shape(frames.shape[0], frames.shape[1], frames.shape[2], out_size))
        mix = float(min(max(look_mix, 0.0), 1.0))
        if mix < 1.0:                      # міксер оригінал↔лук: лінійно по кодованих значеннях (так само, як прозорість у превʼю картки)
            arr = mix_with_original(arr, frames, mix)
        print("[DepthLookSimple] %d кадр(ів) %dx%d -> %dx%d за %.1f с, сила луку %.2f" % (
            frames.shape[0], frames.shape[2], frames.shape[1], arr.shape[2], arr.shape[1], time.time() - t0, mix))
        g = dict(g, look_mix=mix)         # сила луку йде далі по grade: повна й відео-нода змішують так само
        return {"ui": ui, "result": (to_result(arr, image), g)}


def node_key(unique_id):
    """Ключ кешу/маршрутів картки Simple: окремий простір імен, id ноди з іншого workflow не підмінить кадр DepthLook."""
    return "s" + str(unique_id)


def mix_with_original(arr, frames, mix):
    """arr (B,H,W,3) результат луку; frames (B,h,w,3) оригінал. -> новий масив: оригінал*(1-mix) + лук*mix."""
    out = np.empty_like(arr)
    H, W = arr.shape[1:3]
    for i in range(arr.shape[0]):
        o = np.asarray(frames[i][..., :3], np.float32)
        if o.shape[:2] != (H, W):
            o = cv2.resize(o, (W, H), interpolation=cv2.INTER_AREA if o.shape[1] > W else cv2.INTER_CUBIC)
        out[i] = o * (1.0 - mix) + arr[i] * mix
    return out
