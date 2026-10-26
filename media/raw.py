"""RAW-фото на вхід: LibRaw (rawpy) → float IMAGE без 8-бітної втрати.

Базова проявка, не Camera Raw: баланс білого (з камери / авто / денний + підстроювання R і B),
експозиція в EV, режим світлих ділянок LibRaw, демозаїк, колірний простір і крива виходу.
Експозиція й підстроювання БІ рахуються вже в лінійному float після LibRaw, тому вони точні
і не стискають світлі ділянки (exp_shift у LibRaw має межі й власне «зберегти світлі»).
"""
import os

import numpy as np
import torch

import folder_paths

from .color import srgb_oetf

RAW_EXT = (".nef", ".nrw", ".cr2", ".cr3", ".crw", ".arw", ".srf", ".sr2", ".dng", ".raf", ".orf",
           ".rw2", ".pef", ".srw", ".3fr", ".iiq", ".erf", ".kdc", ".mef", ".mos", ".x3f", ".rwl")

WB = ["as shot", "auto", "daylight"]
HIGHLIGHTS = ["clip", "blend", "reconstruct", "unclipped"]
DEMOSAIC = ["AHD", "DHT", "AAHD", "DCB", "VNG", "PPG", "linear"]
SPACES = ["sRGB / Rec.709", "Rec.2020", "Display P3", "ACES AP0", "ProPhoto", "Adobe RGB", "XYZ"]
CURVES = ["linear", "sRGB"]


def _raw_files():
    root = folder_paths.get_input_directory()
    out = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(RAW_EXT):
                out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out) or ["(put RAW files into ComfyUI/input)"]


def _resolve(file, path):
    p = path.strip().strip("'\"")
    p = os.path.abspath(os.path.expanduser(p)) if p else os.path.join(folder_paths.get_input_directory(), file)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"Load RAW: file not found: {p}")
    return p


def _develop(p, wb, highlights, demosaic, space, half, auto_bright):
    import rawpy

    hl = {"clip": rawpy.HighlightMode.Clip, "blend": rawpy.HighlightMode.Blend,
          "reconstruct": rawpy.HighlightMode.ReconstructDefault, "unclipped": rawpy.HighlightMode.Ignore}[highlights]
    dm = {"linear": "LINEAR"}.get(demosaic, demosaic)
    cs = {"sRGB / Rec.709": "sRGB", "Rec.2020": "Rec2020", "Display P3": "P3D65", "ACES AP0": "ACES",
          "ProPhoto": "ProPhoto", "Adobe RGB": "Adobe", "XYZ": "XYZ"}[space]
    with rawpy.imread(p) as raw:
        meta = {"camera_wb": [round(float(v), 4) for v in raw.camera_whitebalance],
                "black": int(np.mean(raw.black_level_per_channel)), "white": int(raw.white_level),
                "raw_size": [int(raw.sizes.raw_width), int(raw.sizes.raw_height)]}
        rgb = raw.postprocess(
            use_camera_wb=wb == "as shot", use_auto_wb=wb == "auto",
            highlight_mode=hl, demosaic_algorithm=getattr(rawpy.DemosaicAlgorithm, dm),
            output_color=getattr(rawpy.ColorSpace, cs), output_bps=16,
            gamma=(1, 1), no_auto_bright=not auto_bright, half_size=half, user_flip=None)
    return rgb.astype(np.float32) / 65535.0, meta


class SVKLoadRAW:
    DESCRIPTION = ("Loads a camera RAW (NEF, CR2, CR3, ARW, RAF, DNG, ORF, RW2…) through LibRaw and develops it "
                   "into a float IMAGE with no 8-bit loss. Linear output is scene-referred: feed it to Save Image HDR "
                   "as EXR, or pick the sRGB curve for regular nodes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "file": (_raw_files(), {"tooltip": "RAW file from ComfyUI/input (subfolders included)"}),
            "white_balance": (WB, {"default": "as shot",
                              "tooltip": "as shot = camera's white balance; auto = LibRaw gray-world; daylight = camera's daylight multipliers"}),
            "exposure": ("FLOAT", {"default": 0.0, "min": -6.0, "max": 6.0, "step": 0.05,
                         "tooltip": "exposure in EV stops, applied in linear light after development"}),
            "highlights": (HIGHLIGHTS, {"default": "blend",
                           "tooltip": "clip = hard white; blend = soft, no magenta; reconstruct = rebuild clipped channels; unclipped = keep raw channel ratios (may tint)"}),
            "output_curve": (CURVES, {"default": "sRGB",
                             "tooltip": "linear = scene-linear values for EXR / compositing; sRGB = display curve for regular image nodes"}),
        }, "optional": {
            "path": ("STRING", {"default": "", "tooltip": "absolute path to a RAW file anywhere on disk; overrides 'file' when set"}),
            "color_space": (SPACES, {"default": "sRGB / Rec.709", "tooltip": "output primaries; ComfyUI nodes assume sRGB / Rec.709"}),
            "wb_red": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01, "tooltip": "red multiplier on top of the white balance"}),
            "wb_blue": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.01, "tooltip": "blue multiplier on top of the white balance"}),
            "demosaic": (DEMOSAIC, {"default": "AHD", "tooltip": "demosaic algorithm; AHD is a safe default, DHT/AAHD are sharper"}),
            "auto_bright": ("BOOLEAN", {"default": False, "tooltip": "LibRaw auto brightness: scales so ~1% of pixels clip. Off = true camera exposure"}),
            "half_size": ("BOOLEAN", {"default": False, "tooltip": "half resolution, 4× faster: for previews and tuning"}),
        }}

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("image", "info")
    OUTPUT_TOOLTIPS = ("Developed photo, float RGB (values above 1.0 kept in linear mode) → Depth Look, Camera Realism, Save Image · HDR or any image node",
                       "File, size and LibRaw levels → Preview Any. Safe to leave unconnected")
    FUNCTION = "load"
    CATEGORY = "StepVisionKit/IO"

    @classmethod
    def IS_CHANGED(cls, file, path="", **kw):
        try:
            p = _resolve(file, path or "")
            return f"{p}:{os.path.getmtime(p)}:{sorted(kw.items())}"
        except Exception:
            return float("nan")

    @classmethod
    def VALIDATE_INPUTS(cls, file, path="", **kw):
        try:
            _resolve(file, path or "")
        except FileNotFoundError as e:
            return str(e)
        return True

    def load(self, file, white_balance, exposure, highlights, output_curve, path="", color_space="sRGB / Rec.709",
             wb_red=1.0, wb_blue=1.0, demosaic="AHD", half_size=False, auto_bright=False):
        p = _resolve(file, path)
        a, meta = _develop(p, white_balance, highlights, demosaic, color_space, half_size, auto_bright)
        gain = np.array([wb_red, 1.0, wb_blue], np.float32) * np.float32(2.0 ** exposure)
        a *= gain
        if output_curve == "sRGB":
            a = np.clip(srgb_oetf(a), 0.0, 1.0).astype(np.float32)
        h, w = a.shape[:2]
        info = (f"{os.path.basename(p)} · {w}×{h} · {white_balance} wb {meta['camera_wb']} · "
                f"black {meta['black']} white {meta['white']} · {color_space}, {output_curve}")
        print(f"[SVK Load RAW] {info}")
        return (torch.from_numpy(np.ascontiguousarray(a))[None], info)


NODES = {"SVKLoadRAW": SVKLoadRAW}
DISPLAY = {"SVKLoadRAW": "Load RAW · photo"}
