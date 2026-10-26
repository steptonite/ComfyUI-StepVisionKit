"""Професійне відео через ffmpeg: вхід ProRes / DNxHR / HEVC 10 біт / послідовності (EXR, PNG 16, TIFF, DPX),
вихід ProRes 422 HQ / 4444 / 4444 XQ, DNxHR HQX, HEVC 10 біт (SDR, PQ, HLG) і EXR-послідовність.

Кадри йдуть через трубу як rgb48 (16 біт на канал), тож 10–12-бітне джерело не ріжеться до 8 біт.
Матрицю YUV↔RGB бере swscale з метаданих файлу; криві й праймеріз рахуємо самі (color.py),
бо zscale є не в кожній збірці ffmpeg.
"""
import glob
import json
import os
import shutil
import subprocess

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

import folder_paths

from . import color as C
from .save_hdr import EXR_COMP, bgr, out_dir, write_exr

VIDEO_EXT = (".mov", ".mp4", ".m4v", ".mxf", ".mkv", ".avi", ".webm", ".mts", ".m2ts")
SEQ_EXT = (".exr", ".png", ".tif", ".tiff", ".dpx", ".jpg", ".jpeg")


def _ffmpeg(name="ffmpeg"):
    exe = shutil.which(name)
    if not exe:
        raise RuntimeError(f"{name} not found in PATH: install ffmpeg (brew install ffmpeg / apt install ffmpeg)")
    return exe


def _video_files():
    root = folder_paths.get_input_directory()
    out = []
    for d, _, files in os.walk(root):
        out += [os.path.relpath(os.path.join(d, f), root) for f in files if f.lower().endswith(VIDEO_EXT)]
    return sorted(out) or ["(put video files into ComfyUI/input)"]


def _path(file, path):
    p = path.strip().strip("'\"")
    return os.path.abspath(os.path.expanduser(p)) if p else os.path.join(folder_paths.get_input_directory(), file)


def probe(p):
    r = subprocess.run([_ffmpeg("ffprobe"), "-v", "error", "-select_streams", "v:0", "-count_packets",
                        "-show_entries", "stream=width,height,pix_fmt,r_frame_rate,nb_read_packets,codec_name,"
                        "profile,color_transfer,color_primaries,color_space,color_range",
                        "-of", "json", p], capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffprobe failed on {p}: {r.stderr.strip()[-300:]}")
    s = json.loads(r.stdout)["streams"][0]
    n, d = (s.get("r_frame_rate") or "24/1").split("/")
    s["fps"] = float(n) / float(d or 1)
    s["frames"] = int(s.get("nb_read_packets") or 0)
    return s


_MATRIX = {"bt2020nc": "bt2020", "bt2020c": "bt2020", "bt709": "bt709", "smpte170m": "smpte170m",
           "bt470bg": "bt470bg", "fcc": "fcc", "smpte240m": "smpte240m"}


def to_linear(a, transfer):
    if transfer == "smpte2084":
        return C.pq_decode(a)
    if transfer == "arib-std-b67":
        return C.hlg_decode(a)
    if transfer in ("linear",):
        return a
    return C.bt1886(a) if transfer in ("bt709", "smpte170m", "bt2020-10", "bt2020-12") else C.srgb_eotf(a)


def frames_cmd(p, s, start_frame=0, frame_count=0, every_nth=1, max_width=0):
    """ffmpeg -> rgb48le / rgba64le кадри в трубу. Спільна для Load Video Pro і потокової глибини (depth_look/depth_stream.py),
    щоб обидві декодували кадр байт-у-байт однаково. -> (cmd, w, h, ch)"""
    w, h = int(s["width"]), int(s["height"])
    if max_width and max_width < w:
        w, h = max_width, int(round(h * max_width / w / 2)) * 2
    alpha = "a" in s.get("pix_fmt", "").replace("gray", "") and s.get("pix_fmt", "").startswith(("yuva", "rgba", "bgra", "gbrap", "argb", "ya"))
    mat = _MATRIX.get(s.get("color_space", ""), "auto")
    rng = {"tv": "tv", "pc": "pc"}.get(s.get("color_range", ""), "auto")
    vf = [f"select='gte(n\\,{start_frame})*not(mod(n-{start_frame}\\,{every_nth}))'"]
    vf.append(f"scale={w}:{h}:flags=lanczos:in_color_matrix={mat}:in_range={rng}")
    pix = "rgba64le" if alpha else "rgb48le"
    cmd = [_ffmpeg(), "-v", "error", "-i", p, "-vf", ",".join(vf), "-fps_mode", "passthrough"]
    if frame_count:
        cmd += ["-frames:v", str(frame_count)]
    cmd += ["-f", "rawvideo", "-pix_fmt", pix, "-"]
    return cmd, w, h, 4 if alpha else 3


class SVKLoadVideoPro:
    DESCRIPTION = ("Loads ProRes, DNxHR/DNxHD, 10/12-bit HEVC, H.264 and other ffmpeg formats as 16-bit frames, "
                   "keeping HDR (PQ / HLG) and alpha (ProRes 4444). Can convert to scene-linear Rec.709 for compositing.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "file": (_video_files(), {"tooltip": "video from ComfyUI/input (subfolders included)"}),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "first frame to load"}),
            "frame_count": ("INT", {"default": 0, "min": 0, "max": 100_000, "tooltip": "how many frames; 0 = to the end (watch RAM: 1080p ≈ 25 MB per frame)"}),
            "output_curve": (["as encoded", "linear"], {"default": "as encoded",
                             "tooltip": "as encoded = code values as in the file (PQ/HLG stay HDR-encoded); linear = decode the file's transfer to scene-linear, 1.0 = diffuse white"}),
        }, "optional": {
            "path": ("STRING", {"default": "", "tooltip": "absolute path to a video anywhere on disk; overrides 'file' when set"}),
            "every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "tooltip": "take every Nth frame"}),
            "max_width": ("INT", {"default": 0, "min": 0, "max": 16384, "step": 2, "tooltip": "downscale to this width; 0 = original size"}),
            "gamut": (["keep source", "Rec.709 / sRGB"], {"default": "keep source",
                      "tooltip": "Rec.709 converts Rec.2020 HDR primaries into ComfyUI's usual gamut (linear output only)"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "FLOAT", "INT", "STRING")
    RETURN_NAMES = ("images", "alpha", "fps", "frame_count", "info")
    OUTPUT_TOOLTIPS = ("16-bit precision float frames", "alpha channel, 1 = opaque (ones when the file has none)",
                       "source frame rate divided by every_nth", "frames loaded", "codec, pixel format, color tags")
    FUNCTION = "load"
    CATEGORY = "StepVisionKit/IO"

    @classmethod
    def IS_CHANGED(cls, file, path="", **kw):
        p = _path(file, path or "")
        return f"{p}:{os.path.getmtime(p) if os.path.isfile(p) else 0}:{sorted(kw.items())}"

    @classmethod
    def VALIDATE_INPUTS(cls, file, path="", **kw):
        p = _path(file, path or "")
        return True if os.path.isfile(p) else f"Load Video (Pro): file not found: {p}"

    def load(self, file, start_frame, frame_count, output_curve, path="", every_nth=1, max_width=0, gamut="keep source"):
        p = _path(file, path)
        s = probe(p)
        cmd, w, h, ch = frames_cmd(p, s, start_frame, frame_count, every_nth, max_width)
        alpha = ch == 4
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        size = w * h * ch * 2
        bufs = []
        while True:
            buf = proc.stdout.read(size)
            if len(buf) < size:
                break
            bufs.append(buf)
        err = proc.stderr.read().decode(errors="ignore")
        proc.wait()
        if not bufs:
            raise RuntimeError(f"Load Video (Pro): no frames decoded from {p}. {err.strip()[-300:]}")
        # 16-біт кадри -> один float32 масив покадрово, звільняючи буфери: пік ≈ розмір результату, не список + stack (×2+)
        n = len(bufs)
        a = np.empty((n, h, w, ch), np.float32)
        for i in range(n):
            np.multiply(np.frombuffer(bufs[i], np.uint16).reshape(h, w, ch), 1.0 / 65535.0, out=a[i], dtype=np.float32)
            bufs[i] = None
        rgb = a[..., :3]
        tr = s.get("color_transfer", "")
        if output_curve == "linear":
            rgb = to_linear(rgb, tr)
            if gamut != "keep source" and s.get("color_primaries") == "bt2020":
                rgb = C.apply_matrix(rgb, C.M2020_TO_709)
        info = (f"{os.path.basename(p)} · {s.get('codec_name')} {s.get('profile', '')} · {s.get('pix_fmt')} · "
                f"{s['width']}×{s['height']} @ {s['fps']:.3f} · trc {tr or '?'} · prim {s.get('color_primaries', '?')} · "
                f"{n} frame(s) → {output_curve}")
        print(f"[SVK Load Video] {info}")
        # без альфи - одна площина одиниць на весь кліп (вʼю), а не B×H×W float32
        al = torch.from_numpy(np.ascontiguousarray(a[..., 3])) if alpha else torch.ones((1, h, w)).expand(n, h, w)
        return (torch.from_numpy(np.ascontiguousarray(rgb)), al, s["fps"] / every_nth, n, info)


def _read_seq_frame(f):
    if f.lower().endswith(".dpx"):
        r = subprocess.run([_ffmpeg(), "-v", "error", "-i", f, "-f", "rawvideo", "-pix_fmt", "rgb48le", "-"],
                           capture_output=True)
        s = probe(f)
        return np.frombuffer(r.stdout, np.uint16).reshape(int(s["height"]), int(s["width"]), 3).astype(np.float32) / 65535.0
    a = cv2.imread(f, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"Load Image Sequence: can't read {f}")
    scale = {np.dtype(np.uint8): 255.0, np.dtype(np.uint16): 65535.0}.get(a.dtype, 1.0)
    a = a.astype(np.float32) / scale
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, -1)
    return a[..., [2, 1, 0, 3]] if a.shape[-1] == 4 else a[..., ::-1]


class SVKLoadImageSequence:
    DESCRIPTION = ("Loads a numbered image sequence (EXR, 16-bit PNG/TIFF, DPX, JPG) from a folder as float frames, "
                   "values above 1.0 kept. Alpha goes to the alpha output.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "folder": ("STRING", {"default": "", "tooltip": "folder with the frames: absolute path or a subfolder of ComfyUI/input"}),
            "pattern": ("STRING", {"default": "*", "tooltip": "file name filter, e.g. shot_*.exr; files are sorted by name"}),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000}),
            "frame_count": ("INT", {"default": 0, "min": 0, "max": 100_000, "tooltip": "0 = all"}),
        }, "optional": {
            "every_nth": ("INT", {"default": 1, "min": 1, "max": 1000}),
            "input_curve": (["as stored", "sRGB → linear"], {"default": "as stored",
                            "tooltip": "EXR is normally linear already; use sRGB → linear for display-encoded PNG/TIFF/JPG"}),
        }}

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "STRING")
    RETURN_NAMES = ("images", "alpha", "frame_count", "info")
    FUNCTION = "load"
    CATEGORY = "StepVisionKit/IO"

    @staticmethod
    def _files(folder, pattern):
        d = folder.strip().strip("'\"")
        d = os.path.abspath(os.path.expanduser(d)) if os.path.isabs(os.path.expanduser(d)) else os.path.join(folder_paths.get_input_directory(), d)
        fs = sorted(f for f in glob.glob(os.path.join(d, pattern or "*")) if f.lower().endswith(SEQ_EXT))
        if not fs:
            raise FileNotFoundError(f"Load Image Sequence: no {'/'.join(SEQ_EXT)} files matching '{pattern}' in {d}")
        return fs

    @classmethod
    def IS_CHANGED(cls, folder, pattern, **kw):
        try:
            fs = cls._files(folder, pattern)
            return f"{len(fs)}:{max(os.path.getmtime(f) for f in fs)}:{sorted(kw.items())}"
        except Exception:
            return float("nan")

    def load(self, folder, pattern, start_frame, frame_count, every_nth=1, input_curve="as stored"):
        fs = self._files(folder, pattern)[start_frame::every_nth]
        if frame_count:
            fs = fs[:frame_count]
        if not fs:
            raise ValueError("Load Image Sequence: start_frame is past the last file")
        frames = [_read_seq_frame(f) for f in fs]
        h, w = frames[0].shape[:2]
        if any(f.shape[:2] != (h, w) for f in frames):
            raise ValueError("Load Image Sequence: frames have different sizes")
        has_a = any(f.shape[-1] == 4 for f in frames)
        rgb = np.stack([f[..., :3] for f in frames])
        al = np.stack([f[..., 3] if f.shape[-1] == 4 else np.ones((h, w), np.float32) for f in frames])
        if input_curve == "sRGB → linear":
            rgb = C.srgb_eotf(rgb)
        info = f"{len(fs)} frame(s) · {w}×{h} · {os.path.basename(fs[0])} … {os.path.basename(fs[-1])} · alpha {'yes' if has_a else 'no'}"
        print(f"[SVK Load Sequence] {info}")
        return (torch.from_numpy(rgb), torch.from_numpy(al), len(fs), info)


# формат → (кодек, профіль, піксельний формат, контейнер, чи вміє альфу)
VIDEO_FORMATS = {
    "ProRes 422 HQ": ("prores_ks", "3", "yuv422p10le", ".mov", False),
    "ProRes 4444": ("prores_ks", "4", "yuva444p10le", ".mov", True),
    "ProRes 4444 XQ": ("prores_ks", "5", "yuva444p10le", ".mov", True),
    "DNxHR HQX 10-bit": ("dnxhd", "dnxhr_hqx", "yuv422p10le", ".mov", False),
    "HEVC 10-bit": ("libx265", None, "yuv420p10le", ".mp4", False),
    "EXR sequence": (None, None, None, ".exr", True),
}
TRANSFERS = ["SDR Rec.709", "HDR PQ (HDR10)", "HDR HLG"]
INPUT_CURVES = ["sRGB", "linear", "HLG (as encoded)", "PQ (as encoded)"]
_TAGS = {"SDR Rec.709": ("bt709", "bt709", "bt709"),
         "HDR PQ (HDR10)": ("bt2020", "smpte2084", "bt2020nc"),
         "HDR HLG": ("bt2020", "arib-std-b67", "bt2020nc")}


class SVKSaveVideoHDR:
    DESCRIPTION = ("Encodes frames to ProRes 422 HQ / 4444 / 4444 XQ, DNxHR HQX or 10-bit HEVC with SDR Rec.709, "
                   "HDR PQ (HDR10) or HLG tags, or writes a linear EXR sequence. Linear 1.0 maps to 203-nit diffuse white in HDR.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "svk/video"}),
            "format": (list(VIDEO_FORMATS), {"default": "ProRes 422 HQ",
                       "tooltip": "ProRes/DNxHR = editing masters (.mov); HEVC 10-bit = delivery (.mp4, plays HDR on Apple devices); EXR = linear frames for grading/compositing"}),
            "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 240.0, "step": 0.001}),
            "transfer": (TRANSFERS, {"default": "SDR Rec.709",
                         "tooltip": "SDR = regular Rec.709; PQ = HDR10 (Rec.2020, absolute nits); HLG = broadcast HDR. Ignored for EXR (always linear)"}),
            "input_curve": (INPUT_CURVES, {"default": "sRGB",
                            "tooltip": "what the incoming frames are: sRGB = regular ComfyUI images; linear = Load RAW / Load Video linear / EXR; "
                                       "HLG / PQ (as encoded) = Load Video (Pro) 'as encoded' from an HDR file, also after Depth Look - "
                                       "written back untouched when transfer matches, so highlights survive"}),
        }, "optional": {
            "alpha": ("MASK", {"tooltip": "alpha for ProRes 4444 / XQ and EXR, 1 = opaque"}),
            "folder": ("STRING", {"default": "", "tooltip": "where to save; empty = ComfyUI output"}),
            "hevc_crf": ("INT", {"default": 16, "min": 0, "max": 51, "tooltip": "HEVC quality: lower = better and bigger; 12–18 is visually lossless"}),
            "max_nits": ("INT", {"default": 1000, "min": 203, "max": 10000, "step": 1,
                         "tooltip": "PQ only: brightest value written to HDR10 metadata (MaxCLL) and the clip point"}),
            "exr_compression": (list(EXR_COMP), {"default": "zip"}),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("path",)
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/IO"

    @staticmethod
    def _encode(rgb_lin, transfer, max_nits):
        if transfer == "HDR PQ (HDR10)":
            return C.pq_encode(np.minimum(C.apply_matrix(rgb_lin, C.M709_TO_2020), max_nits / C.REF_WHITE))
        if transfer == "HDR HLG":
            return C.hlg_encode(C.apply_matrix(rgb_lin, C.M709_TO_2020))
        return C.bt1886_inv(rgb_lin)

    @staticmethod
    def _passthrough(input_curve, transfer):
        """Код-значення вже в потрібній кривій (і в Rec.2020): пишемо як є, без декоду й матриці."""
        return (input_curve, transfer) in (("HLG (as encoded)", "HDR HLG"), ("PQ (as encoded)", "HDR PQ (HDR10)"))

    def save(self, images, filename_prefix, format, fps, transfer, input_curve, alpha=None, folder="",
             hevc_crf=16, max_nits=1000, exr_compression="zip"):
        codec, profile, pix, ext, can_alpha = VIDEO_FORMATS[format]
        n, h, w = images.shape[:3]
        full, fname, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir(folder), w, h)
        os.makedirs(full, exist_ok=True)
        al = None
        if alpha is not None and can_alpha:
            al = alpha.cpu().numpy().astype(np.float32)
            al = al[None] if al.ndim == 2 else al

        def frame(i):
            """-> scene-linear Rec.709 (1.0 = дифузний білий)."""
            a = images[i].cpu().numpy().astype(np.float32)[..., :3]
            if input_curve == "sRGB":
                return C.srgb_eotf(a)
            if input_curve == "HLG (as encoded)":
                return C.apply_matrix(C.hlg_decode(a), C.M2020_TO_709)
            if input_curve == "PQ (as encoded)":
                return C.apply_matrix(C.pq_decode(a), C.M2020_TO_709)
            return a

        if codec is None:
            d = os.path.join(full, f"{fname}_{counter:05}")
            os.makedirs(d, exist_ok=True)
            for i in range(n):
                px = frame(i)
                if al is not None:
                    px = np.concatenate([px, al[min(i, len(al) - 1)][..., None]], -1)
                write_exr(os.path.join(d, f"{fname}_{i:05}.exr"), bgr(px), True, EXR_COMP[exr_compression])
            print(f"[SVK Save Video] {n} EXR frame(s) → {d}")
            return {"ui": {"text": [f"{n} EXR frames → {d}"]}, "result": (d,)}

        if (w % 2 or h % 2) and pix.startswith(("yuv420", "yuv422")):
            raise ValueError(f"{format} needs even width and height, got {w}×{h}")
        prim, trc, mat = _TAGS[transfer]
        out = os.path.join(full, f"{fname}_{counter:05}{ext}")
        in_pix = "rgba64le" if al is not None else "rgb48le"
        cmd = [_ffmpeg(), "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", in_pix, "-s", f"{w}x{h}", "-r", f"{fps}", "-i", "-",
               "-vf", f"setparams=colorspace=gbr,scale=out_color_matrix={'bt2020' if mat == 'bt2020nc' else 'bt709'}:out_range=tv,format={pix if al is not None or not pix.startswith('yuva') else pix.replace('yuva', 'yuv')},setparams=color_primaries={prim}:color_trc={trc}:colorspace={mat}:range=tv",
               "-c:v", codec]
        if codec == "prores_ks":
            cmd += ["-profile:v", profile, "-vendor", "apl0"]
            if al is None and pix.startswith("yuva"):
                cmd += ["-alpha_bits", "0"]
        elif codec == "dnxhd":
            cmd += ["-profile:v", profile]
        else:
            x = f"colorprim={prim}:transfer={trc}:colormatrix={mat}:range=limited:repeat-headers=1"
            if transfer == "HDR PQ (HDR10)":
                x += f":hdr10=1:max-cll={max_nits},{min(max_nits, 400)}"
            cmd += ["-crf", str(hevc_crf), "-preset", "medium", "-tag:v", "hvc1", "-x265-params", x + ":log-level=error"]
        cmd += ["-color_primaries", prim, "-color_trc", trc, "-colorspace", mat, "-color_range", "tv"]
        cmd += (["-movflags", "+write_colr"] if ext == ".mov" else []) + [out]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for i in range(n):
                if self._passthrough(input_curve, transfer):
                    sig = np.clip(images[i].cpu().numpy().astype(np.float32)[..., :3], 0.0, 1.0)
                else:
                    sig = self._encode(frame(i), transfer, max_nits)
                if al is not None:
                    sig = np.concatenate([sig, al[min(i, len(al) - 1)][..., None]], -1)
                proc.stdin.write(np.clip(sig * 65535.0 + 0.5, 0, 65535).astype("<u2").tobytes())
            proc.stdin.close()
        except BrokenPipeError:
            pass
        err = proc.stderr.read().decode(errors="ignore")
        if proc.wait():
            raise RuntimeError(f"ffmpeg failed ({format}, {transfer}): {err.strip()[-500:]}")
        print(f"[SVK Save Video] {n} frame(s) → {out}  ({format}, {transfer})")
        return {"ui": {"text": [f"{n} frames → {out}"]}, "result": (out,)}


NODES = {"SVKLoadVideoPro": SVKLoadVideoPro, "SVKLoadImageSequence": SVKLoadImageSequence,
         "SVKSaveVideoHDR": SVKSaveVideoHDR}
DISPLAY = {"SVKLoadVideoPro": "Load Video (ProRes / DNxHR / HEVC 10-bit)",
           "SVKLoadImageSequence": "Load Image Sequence (EXR / PNG 16 / DPX)",
           "SVKSaveVideoHDR": "Save Video HDR (ProRes / DNxHR / HEVC / EXR)"}
