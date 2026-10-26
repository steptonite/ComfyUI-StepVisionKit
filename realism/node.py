"""Нода SVKCameraRealism: «знято камерою» - сенсорний шум, байєр, ISP, оптика, глибина, тремтіння. Рушій - engine.py (numpy)."""
import numpy as np

from . import params as P
from . import preview
from .engine import apply_realism

_TIP = {
    "image": "Image or batch of frames (a render, a generation, a video). Put this node last, right before Save.",
    "depth": "Optional depth map (bright = near), one for the whole batch or one per frame. Enables aerial haze and near detail.",
    "profile": "Camera look: picks a ready-made set of all settings. Custom uses the sliders under «more»: moving a slider switches to Custom.",
    "strength": "Master strength 0..1: scales every effect strength of the profile (1 = the profile exactly, 0 = nothing is computed).",
    "seed": "Noise seed. Noise is new on every frame; the fixed sensor pattern depends only on the seed.",
}


class SVKCameraRealism:
    @classmethod
    def INPUT_TYPES(cls):
        opt = {"depth": ("IMAGE", {"tooltip": _TIP["depth"]})}
        for s in P.SPEC:
            name, lo, hi, step, _grp, label, tip = s[:7]
            is_int = len(s) > 7 and s[7] == "i"
            d = P.WIDGET_DEFAULTS[name]
            opt[name] = ("INT" if is_int else "FLOAT",
                         {"default": int(d) if is_int else float(d), "min": lo, "max": hi, "step": step,
                          "label": label, "tooltip": tip})
        return {
            "required": {
                "image": ("IMAGE", {"tooltip": _TIP["image"]}),
                # svk_profiles - значення профілів для картки: вибір плитки заповнює повзунки
                "profile": (P.PROFILE_NAMES, {"default": P.DEFAULT_PROFILE, "tooltip": _TIP["profile"],
                                              "svk_profiles": {n: P.profile_values(n) for n in P.PROFILES}}),
                "strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": _TIP["strength"]}),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFF, "control_after_generate": False,
                                 "tooltip": _TIP["seed"]}),
            },
            "optional": opt,
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Realism"

    def run(self, image, profile, strength, seed, depth=None, unique_id=None, **sliders):
        import torch
        frames = image.detach().cpu().numpy()[..., :3]
        dep = None
        if depth is not None:
            dep = depth.detach().cpu().numpy().astype(np.float32)
            dep = dep.mean(axis=-1) if dep.ndim == 4 else dep
        custom = {k: v for k, v in sliders.items() if k in P.NAMES}
        p = P.resolve(profile, strength, custom)
        off = int(sliders.get("frame_offset", 0))
        out = apply_realism(frames, dep, p, seed=int(seed), frame_offset=off)
        ui = []
        if unique_id is not None:                  # кадри для превʼю й лупи на картці
            try:
                ui = [preview.store(unique_id, frames, dep, off)]
            except Exception as e:   # noqa: BLE001 - превʼю не має валити рендер
                print("[CameraRealism] preview cache: %r" % e)
        if np.shares_memory(out, frames):          # нічого не активно: рушій віддав вхід як є
            res = image
        else:
            res = torch.from_numpy(np.ascontiguousarray(out)).to(image.dtype)
        return {"ui": {"svk_realism": ui}, "result": (res,)}


NODES = {"SVKCameraRealism": SVKCameraRealism}
DISPLAY = {"SVKCameraRealism": "Camera Realism"}
