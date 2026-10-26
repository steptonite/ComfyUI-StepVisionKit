"""ComfyUI-ноди Depth Look (look v1): лінзовий блюр за глибиною + палітра + силует/шлейф/зерно.
Три ноди (категорія StepVisionKit/Depth):
  DepthLook       - статична, із вбудованим тюнером; віддає IMAGE і `grade` (DL_GRADE) для інших нод;
  DepthLookSimple - проста: пресет + «вкл/сила» по групах ефектів, без тюнера (simple.py);
  DepthLookVideo  - відео: приймає `grade`, додає анімацію, фінальний розмір, швидке превʼю (video_node.py).
Рушій — look.py / look_params.py / look_api.py / tune.html поруч, у цій теці. Лише numpy + cv2 (+ torch від Comfy).
Рендер іде в окремому процесі (worker.py): уся памʼять вертається ОС після кожного прогону.
Ядро (apply_numpy тощо) — у core.py (без torch), запуск воркера - у render_proc.py."""
import asyncio
import contextlib
import json
import os
import sys
import threading
import time

_PARENT = os.path.dirname(os.path.realpath(__file__))     # рушій лежить у цій же теці
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
_HERE = os.path.dirname(os.path.realpath(__file__))

import numpy as np  # noqa: E402,F401
try:
    from comfy_execution.graph_utils import ExecutionBlocker  # noqa: E402
except Exception:            # старий Comfy без блокера
    ExecutionBlocker = None
import look  # noqa: E402,F401
import look_params  # noqa: E402
import look_api  # noqa: E402
from .core import (DEFAULT_PRESET, _EXTRA_KEYS, parse_palette, build_preset, _params_from_json,  # noqa: E402,F401
                   _work_size, _Lazy, apply_numpy, check_inputs, SIZE_ORIGINAL, OUTPUT_SIZES,
                   GRADE_TYPE, parse_output_size, legacy, validate_legacy, resolve_params, static_params, make_grade, out_shape)
from .render_proc import (_interrupted, _render_subproc, tensors_to_numpy, make_progress, stage,  # noqa: E402,F401
                          render as _render, to_result)


MODE_TUNE = "tune (fast, 720)"
MODE_FULL = "full render"
TUNE_SIDE = 720


class DepthLook:
    @classmethod
    def INPUT_TYPES(cls):
        presets = look_params.list_presets()
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": "Frames (B,H,W,3)."}),
                "depth": ("IMAGE", {"tooltip": "Depth map: one for the whole batch or one per frame. Bright = near."}),
                "preset": (presets, {"default": DEFAULT_PRESET if DEFAULT_PRESET in presets else presets[0],
                                     "tooltip": "Preset from look_proto/presets. If grade is connected, it is used instead of the preset."}),
                "clean_depth": ("BOOLEAN", {"default": True, "tooltip": "Raw depth is cleaned up (the frame is the guide). Turn off if the depth is already clean 0..1."}),
                "depth_light_is_near": ("BOOLEAN", {"default": True, "tooltip": "Bright = near. Turn off if your depth node outputs the opposite."}),
            },
            # порядок: дані -> грейд -> розмір/якість -> службове. fps/start_time прибрано зі статики -
            # widgets_values старих workflow (11 значень) мігрує js у onConfigure
            "optional": {
                "grade": (GRADE_TYPE, {"tooltip": "Grade from another Depth Look node. It is the base; tuner edits/overrides are applied on top, and with no edits it passes through."}),
                "output_size": (OUTPUT_SIZES, {"default": SIZE_ORIGINAL, "tooltip": "Final frame size (long side, px). The first option keeps the input size. Only applies in the full-render mode; the tuner preview does not depend on it."}),
                "max_side": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 8, "tooltip": "Working resolution: the look is computed on a frame with this long side and then upscaled to the final size. 0 = compute directly at final size. For long clips on 8 GB machines."}),
                "depth_levels": (["clip", "frame"], {"default": "clip", "tooltip": "clip - shared depth levels across the whole clip (no flicker); frame - per-frame stretch. Only with clean_depth."}),
                # спершу крутиш тюнер на швидкому прогоні, потім перемикаєш на повний рендер і запускаєш
                "mode": ([MODE_TUNE, MODE_FULL], {"default": MODE_TUNE, "tooltip": "Tuning mode: fast, ≤720, the image outputs stay blocked. Full render: size from output_size and max_side."}),
                # службові (ховає js): стара текстова форма і значення вбудованого тюнера
                "overrides": ("STRING", {"multiline": True, "default": "{}"}),
                "palette": ("STRING", {"multiline": True, "default": ""}),
                "params_json": ("STRING", {"multiline": False, "default": ""}),   # пише вбудований тюнер; JS ховає
                # новий віджет - лише в кінець: widgets_values старих workflow лягають по порядку
                "auto_run": ("BOOLEAN", {"default": True, "tooltip": "On: every Run renders the final and passes it to the connected nodes (Save / video). Off: Run renders only the preview, the final output stays blocked until 'Render final'."}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE", GRADE_TYPE, "IMAGE")
    RETURN_NAMES = ("preview", "grade", "final")     # final - лише повний рендер: Save на ньому не пише превʼю з водяним знаком
    OUTPUT_NODE = True            # граф запускається й без Save-ноди (превʼю - у вбудованому тюнері)
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    @classmethod
    def VALIDATE_INPUTS(cls, preset=None, output_size=None, mode=None):
        return validate_legacy(preset=(preset, look_params.list_presets()), output_size=(output_size, OUTPUT_SIZES),
                               mode=(mode, [MODE_TUNE, MODE_FULL]))

    def run(self, image, depth, preset, clean_depth, depth_light_is_near, grade=None, output_size=SIZE_ORIGINAL,
            max_side=0, depth_levels="clip", mode=MODE_FULL, overrides="{}", palette="", params_json="",
            auto_run=True, fps=None, start_time=None, unique_id=None):
        # fps/start_time - лише щоб API-запити зі старих workflow не падали: у статичній ноді їх нема (time = 0)
        frames, dep = tensors_to_numpy(image, depth)
        preset, output_size, mode = legacy(preset), legacy(output_size), legacy(mode)
        full = (mode == MODE_FULL) or bool(auto_run)    # фінал: кнопка «Render final» (mode) або галочка auto
        if not full:                                       # швидкий прогін: look на ≤720, назад до рідного розміру
            max_side = min(int(max_side), TUNE_SIDE) if int(max_side or 0) > 0 else TUNE_SIDE
        out_size = parse_output_size(output_size) if full else 0     # превʼю і фінал не залежать одне від одного
        warns = check_inputs(frames, dep, max_side, out_size)        # явна халепа - виняток тут, до запуску воркера
        for w in warns:
            print("[DepthLook] ⚠ " + w)
        p = static_params(resolve_params(preset, overrides, palette, params_json, grade))
        g = make_grade(p, (grade or {}).get("preset") or preset, clean_depth, depth_light_is_near, depth_levels)
        ui = {}
        if unique_id is not None:                          # кадри батча (≤8, float16) для превʼю вбудованого тюнера
            try:
                stage(unique_id, "cache")
                meta = look_api.save_node_cache(unique_id, frames, dep, clean_depth, depth_light_is_near,
                                                depth_levels=depth_levels)
                ui = {"depth_look": [{"frames": meta["indices"], "stamp": meta["stamp"]}]}
            except Exception as e:
                print("[DepthLook] preview cache not written: %s" % e)
        if grade is not None:                              # js ініціалізує тюнер параметрами грейду, поки params_json порожній
            ui["depth_look_grade"] = [{"params": p, "preset": g["preset"]}]
        progress = make_progress(frames.shape[0], unique_id)   # смужка прогресу по кадрах у Comfy + рядок етапів на картці
        t0 = time.time()
        kw = dict(grade=g, clean_depth=bool(clean_depth), light_is_near=bool(depth_light_is_near),
                  max_side=int(max_side or 0), depth_levels=depth_levels, output_size=out_size, static=True)
        with (look_api.final_render() if full else contextlib.nullcontext()):
            arr = _render(frames, dep, kw, progress, out_shape(frames.shape[0], frames.shape[1], frames.shape[2], out_size))
        mix = float(min(max((grade or {}).get("look_mix", 1.0), 0.0), 1.0))     # «сила луку» з простої ноди
        if mix < 1.0:
            from .simple import mix_with_original
            stage(unique_id, "mix")
            arr = mix_with_original(arr, frames, mix)
            g = dict(g, look_mix=mix)
        ui["depth_look_warn"] = warns
        stage(unique_id, "done", secs=round(time.time() - t0, 1))
        print("[DepthLook] %s: %d frame(s) %dx%d -> %dx%d in %.1f s, look strength %.2f" % (
            mode, frames.shape[0], frames.shape[2], frames.shape[1], arr.shape[2], arr.shape[1], time.time() - t0, mix))
        out = to_result(arr, image)
        if not full:                                       # без auto_run / «Render final» далі по графу не йде нічого: превʼю живе в тюнері
            out = ExecutionBlocker(None) if ExecutionBlocker else None
        return {"ui": ui, "result": (out, g, out)}


from .simple import DepthLookSimple  # noqa: E402
from .video_node import DepthLookVideo  # noqa: E402

NODE_CLASS_MAPPINGS = {"DepthLook": DepthLook, "DepthLookSimple": DepthLookSimple, "DepthLookVideo": DepthLookVideo}
NODE_DISPLAY_NAME_MAPPINGS = {"DepthLook": "Depth Look · Full", "DepthLookSimple": "Depth Look · Simple",
                              "DepthLookVideo": "Depth Look · Video"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]   # js - у спільній web/ паку


# ---------------------------------------------------------------- маршрути /depth_look/... (вбудований тюнер)
_APIS = {}
_APIS_LOCK = threading.Lock()


def _api_for(node):
    key = look_api._safe_id(node)
    with _APIS_LOCK:
        if key not in _APIS:
            _APIS[key] = look_api.LookApi(node_id=key)
        return _APIS[key]


def _thumb_api_for(key):
    """Окремий LookApi для плиток: інший розмір кадру не скидає кеш головного превʼю (Renderer тримає один кадр)."""
    k = ("thumb", key)
    with _APIS_LOCK:
        if k not in _APIS:
            _APIS[k] = look_api.LookApi(node_id=key)
        return _APIS[k]


_THUMBS = {}            # (key, stamp, preset, size) -> (jpeg, ms); невеликий кеш плиток
_THUMBS_MAX = 64


def _simple_params(body):
    """Тіло запиту картки Simple -> повні параметри рушія (ті самі, що дасть Run: apply_groups + static_params)."""
    from .simple import apply_groups, POWER_GROUPS, parse_extra
    sw = {}
    for _g, key in POWER_GROUPS:
        v = (body.get("switches") or {}).get(key)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            sw[key] = (bool(v[0]), float(v[1]))
    return static_params(apply_groups(resolve_params(body.get("preset") or DEFAULT_PRESET), sw,
                                      parse_extra(json.dumps(body.get("extra") or {}))))


def _simple_info(key):
    """Усе, що картці треба знати: чи є кадр, пресети з базовою силою груп, групи з іконками й підписами."""
    from .simple import POWER_GROUPS, CHIP_LABEL, ADD_DEFAULT
    meta = look_api.read_node_meta(key)
    presets = []
    for n in look_params.list_presets():
        try:
            p = look_params.load_preset(n)
            presets.append({"name": n, "powers": {k: float(p.get(k, 0.0)) for _g, k in POWER_GROUPS}})
        except Exception as e:   # noqa: BLE001 - зіпсований пресет не валить картку
            print("[DepthLook] preset %s skipped: %s" % (n, e))
    groups = [{"key": k, "name": g, "label": CHIP_LABEL.get(g, g), "icon": look_params.GROUP_ICON.get(g, ""),
               "hint": look_params.GROUP_HINT.get(g, ""), "add": ADD_DEFAULT.get(k, 0.5),
               "max": [s for s in look_params.SPEC if s[0] == k][0][2]} for g, k in POWER_GROUPS]
    return {"node_frame": meta is not None, "orig": (meta or {}).get("orig"), "stamp": (meta or {}).get("stamp"),
            "frames": (meta or {}).get("indices"), "presets": presets, "groups": groups}


def register_routes(routes):
    """Реєструє aiohttp-маршрути на RouteTableDef-подібному обʼєкті (у Comfy — PromptServer.instance.routes)."""
    from aiohttp import web

    @routes.get("/depth_look/ui")
    async def depth_look_ui(request):
        html = open(os.path.join(_PARENT, "tune.html"), encoding="utf-8").read()
        return web.Response(text=html, content_type="text/html", headers={"Cache-Control": "no-store"})

    # ---- картка Depth Look · простий і стрічка відео-ноди (js/depth_look_simple.js) + спільні токени дизайну
    @routes.get("/depth_look/svk_ui.css")
    async def depth_look_css(request):
        return web.Response(text=open(os.path.join(os.path.dirname(_HERE), "web", "svk_ui.css"), encoding="utf-8").read(),
                            content_type="text/css", headers={"Cache-Control": "no-store"})

    def _jr(code, obj):
        return web.json_response(obj, status=code, headers={"Cache-Control": "no-store"})

    @routes.get("/depth_look/simple/{node}/info")
    async def dl_simple_info(request):
        loop = asyncio.get_running_loop()
        return _jr(200, await loop.run_in_executor(None, _simple_info, "s" + look_api._safe_id(request.match_info["node"])))

    @routes.post("/depth_look/simple/{node}/render")
    async def dl_simple_render(request):
        key = "s" + look_api._safe_id(request.match_info["node"])
        try:
            body = await request.json()
            req = json.dumps({"params": _simple_params(body), "size": int(body.get("size", 480)), "frame": int(body.get("frame", 0))})
        except Exception as e:   # noqa: BLE001
            return _jr(400, {"err": str(e)})
        loop = asyncio.get_running_loop()
        code, ctype, data, hdr = await loop.run_in_executor(None, _api_for(key).handle, "POST", "/api/render", req.encode())
        h = dict(hdr); h["Cache-Control"] = "no-store"
        return web.Response(status=code, body=data, content_type=ctype.split(";")[0], headers=h)

    @routes.get("/depth_look/simple/{node}/thumb")
    async def dl_simple_thumb(request):
        key = "s" + look_api._safe_id(request.match_info["node"])
        preset = request.query.get("preset", DEFAULT_PRESET)
        size = max(48, min(320, int(request.query.get("size", 160))))
        meta = look_api.read_node_meta(key)
        if meta is None:
            return _jr(409, {"err": "run it once first", "need_run": True})
        ck = (key, meta["stamp"], preset, size)

        def work():
            if ck in _THUMBS:
                return 200, "image/jpeg", _THUMBS[ck], {}
            req = json.dumps({"params": _simple_params({"preset": preset}), "size": size, "frame": 0}).encode()
            r = _thumb_api_for(key).handle("POST", "/api/render", req)
            if r[0] == 200:
                if len(_THUMBS) >= _THUMBS_MAX:
                    _THUMBS.pop(next(iter(_THUMBS)))
                _THUMBS[ck] = r[2]
            return r
        loop = asyncio.get_running_loop()
        code, ctype, data, hdr = await loop.run_in_executor(None, work)
        return web.Response(status=code, body=data, content_type=ctype.split(";")[0],
                            headers={"Cache-Control": "private, max-age=3600"} if code == 200 else {"Cache-Control": "no-store"})

    @routes.get("/depth_look/simple/{node}/original")
    async def dl_simple_original(request):
        key = "s" + look_api._safe_id(request.match_info["node"])
        loop = asyncio.get_running_loop()
        size = int(request.query.get("size", 480))
        code, ctype, data, hdr = await loop.run_in_executor(
            None, _api_for(key).handle, "GET", "/api/original.jpg?size=%d&frame=0" % size, b"")
        return web.Response(status=code, body=data, content_type=ctype.split(";")[0], headers={"Cache-Control": "no-store"})

    @routes.get("/depth_look/strip/{node}/{i}")
    async def dl_strip(request):
        from .video_node import strip_path, strip_meta_path
        try:
            if request.match_info["i"] == "meta":
                return web.Response(body=open(strip_meta_path(request.match_info["node"]), "rb").read(),
                                    content_type="application/json", headers={"Cache-Control": "no-store"})
            data = open(strip_path(request.match_info["node"], int(request.match_info["i"])), "rb").read()
        except Exception:
            return web.Response(status=404)
        return web.Response(body=data, content_type="image/jpeg", headers={"Cache-Control": "no-store"})

    # ---- пресети руху/анімації відео-ноди (video_presets.py): список / завантажити / зберегти / видалити
    @routes.get("/depth_look/video_presets")
    async def dl_vp_list(request):
        from . import video_presets as VP
        return _jr(200, {"presets": VP.list_presets()})

    @routes.get("/depth_look/video_presets/{name}")
    async def dl_vp_load(request):
        from . import video_presets as VP
        try:
            return _jr(200, {"name": request.match_info["name"], "values": VP.load_preset(request.match_info["name"])})
        except (ValueError, OSError) as e:
            return _jr(404, {"err": str(e)})

    @routes.post("/depth_look/video_presets/save")
    async def dl_vp_save(request):
        from . import video_presets as VP
        try:
            body = await request.json()
            return _jr(200, {"name": VP.save_preset(body.get("name"), body.get("values")), "presets": VP.list_presets()})
        except (ValueError, OSError, AttributeError) as e:
            return _jr(400, {"err": str(e)})

    @routes.post("/depth_look/video_presets/delete")
    async def dl_vp_delete(request):
        from . import video_presets as VP
        try:
            body = await request.json()
            VP.delete_preset(body.get("name"))
            return _jr(200, {"presets": VP.list_presets()})
        except (ValueError, OSError, AttributeError) as e:
            return _jr(400, {"err": str(e)})

    @routes.route("*", "/depth_look/api/{node}/{tail:.*}")
    async def depth_look_api(request):
        prefix = "/depth_look/api/"
        node_raw, _, tail = request.raw_path[len(prefix):].partition("/")
        from urllib.parse import unquote
        api = _api_for(unquote(node_raw))
        body = await request.read() if request.method == "POST" else b""
        loop = asyncio.get_running_loop()
        code, ctype, data, hdr = await loop.run_in_executor(None, api.handle, request.method, "/api/" + tail, body)
        h = dict(hdr)
        h["Cache-Control"] = "no-store"
        return web.Response(status=code, body=data, content_type=ctype.split(";")[0], headers=h)


try:   # у Comfy PromptServer.instance є; у простому import-тесті нема — тоді маршрути не реєструємо
    from server import PromptServer  # type: ignore
    _inst = getattr(PromptServer, "instance", None)
    if _inst is not None:
        register_routes(_inst.routes)
except Exception as _e:   # noqa: BLE001
    if "No module named 'server'" not in str(_e):
        print("[DepthLook] routes not registered: %s" % _e)
