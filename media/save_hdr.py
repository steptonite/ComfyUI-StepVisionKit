"""Вихід без 8-бітної втрати: EXR half / float (лінійний) і PNG 16 біт (дисплейний sRGB).

Крива входу каже, у чому прийшов IMAGE: EXR завжди пишемо лінійним, PNG — у sRGB,
тож нода сама перетворює, якщо вхід не збігається. Маска (MASK) іде в альфу.
Партія кадрів = нумерована послідовність. Превʼю — 8-біт sRGB у temp, на сам файл не впливає.
"""
import os

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np

import folder_paths

from .color import srgb_eotf, srgb_oetf

FORMATS = ["exr half", "exr float", "png 16-bit"]
EXR_COMP = {"zip": cv2.IMWRITE_EXR_COMPRESSION_ZIP, "piz": cv2.IMWRITE_EXR_COMPRESSION_PIZ,
            "dwaa (lossy, small)": cv2.IMWRITE_EXR_COMPRESSION_DWAA, "none": cv2.IMWRITE_EXR_COMPRESSION_NO}
PREVIEW_MAX = 16


def out_dir(folder):
    f = folder.strip().strip("'\"")
    return os.path.abspath(os.path.expanduser(f)) if f else folder_paths.get_output_directory()


def write_exr(path, px, half, comp):
    flags = [cv2.IMWRITE_EXR_TYPE, cv2.IMWRITE_EXR_TYPE_HALF if half else cv2.IMWRITE_EXR_TYPE_FLOAT,
             cv2.IMWRITE_EXR_COMPRESSION, comp]
    if not cv2.imwrite(path, np.ascontiguousarray(px, dtype=np.float32), flags):
        raise RuntimeError(f"EXR not written: {path} (OpenCV built without OpenEXR?)")


def bgr(px):
    # RGB(A) → BGR(A) для OpenCV
    return px[..., [2, 1, 0, 3]] if px.shape[-1] == 4 else px[..., ::-1]


class SVKSaveImageHDR:
    DESCRIPTION = ("Saves images without 8-bit loss: EXR half/float (scene-linear) or 16-bit PNG (sRGB). "
                   "Converts the curve when the input doesn't match the format; an optional mask becomes alpha. "
                   "A batch becomes a numbered sequence.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE", {"tooltip": "images to save"}),
            "filename_prefix": ("STRING", {"default": "svk/image", "tooltip": "file name prefix; a/b makes subfolder a"}),
            "format": (FORMATS, {"default": "exr half",
                       "tooltip": "exr half = 16-bit float, standard for grading and compositing; exr float = 32-bit; png 16-bit = 65536 levels, display sRGB"}),
            "input_curve": (["linear", "sRGB"], {"default": "linear",
                            "tooltip": "what the incoming image is: linear (Load RAW linear, HDR decoders) or sRGB (regular ComfyUI images)"}),
        }, "optional": {
            "mask": ("MASK", {"tooltip": "written as alpha"}),
            "folder": ("STRING", {"default": "", "tooltip": "where to save; empty = ComfyUI output"}),
            "exr_compression": (list(EXR_COMP), {"default": "zip", "tooltip": "zip and piz are lossless; dwaa is lossy and small"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/IO"

    def save(self, images, filename_prefix, format, input_curve, mask=None, folder="", exr_compression="zip"):
        a = images.cpu().numpy().astype(np.float32)[..., :3]
        n, h, w = a.shape[:3]
        exr = format.startswith("exr")
        if exr and input_curve == "sRGB":
            a = srgb_eotf(a)
        elif not exr and input_curve == "linear":
            a = srgb_oetf(a)
        alpha = None
        if mask is not None:
            m = mask.cpu().numpy().astype(np.float32)
            if m.ndim == 2:
                m = m[None]
            if m.shape[1:] != (h, w):
                m = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_LINEAR) for x in m])
            alpha = np.broadcast_to(m, (n, h, w)) if m.shape[0] == 1 else m
        full, fname, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir(folder), w, h)
        os.makedirs(full, exist_ok=True)
        tmp = folder_paths.get_temp_directory()
        os.makedirs(tmp, exist_ok=True)
        ui = []
        for f in range(n):
            px = a[f] if alpha is None else np.concatenate([a[f], alpha[min(f, len(alpha) - 1)][..., None]], -1)
            if exr:
                file = f"{fname}_{counter:05}_.exr"
                write_exr(os.path.join(full, file), bgr(px), format == "exr half", EXR_COMP[exr_compression])
            else:
                file = f"{fname}_{counter:05}_.png"
                cv2.imwrite(os.path.join(full, file), bgr(np.clip(px * 65535.0 + 0.5, 0, 65535).astype(np.uint16)))
            if f < PREVIEW_MAX:
                pv = srgb_oetf(a[f]) if exr else a[f]
                pfile = f"svk_hdr_preview_{fname.replace(os.sep, '_')}_{counter:05}.png"
                cv2.imwrite(os.path.join(tmp, pfile), bgr(np.clip(pv * 255.0 + 0.5, 0, 255).astype(np.uint8)))
                ui.append({"filename": pfile, "subfolder": "", "type": "temp"})
            counter += 1
        print(f"[SVK Save HDR] {n} file(s) → {os.path.join(full, fname)}_*  ({format})")
        return {"ui": {"images": ui}}


NODES = {"SVKSaveImageHDR": SVKSaveImageHDR}
DISPLAY = {"SVKSaveImageHDR": "Save Image · HDR"}
