"""Сира глибина DA3 без 8-бітних проміжків (17.10.2026).

DA3Inference віддає DA3_GEOMETRY — float32 depth у мірилі моделі. Раніше її рендерили в
картинку, зберігали SaveImage (8 біт, 256 рівнів) і грейдили вже сходинки. Тут:
  Raw Save / Raw Load — геометрія на диск і назад float32 (EXR), DA3 рахується один раз;
  Receiver — геометрія → depth IMAGE для Depth Grade, float, без обрізання країв;
  Save 16/EXR — фінальний вихід у 16-біт PNG або float32 EXR.
"""
import asyncio
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2
import numpy as np
import torch

import folder_paths

RAW_DIR = "svk_raw"


def _raw_root():
    return os.path.join(folder_paths.get_output_directory(), RAW_DIR)


def _folder(path):
    # шлях із поля вводу: «Copy as Pathname» у Finder, лапки, ~ — усе приймаємо; порожньо = output/svk_raw
    path = path.strip().strip("'\"")
    return os.path.abspath(os.path.expanduser(path)) if path else _raw_root()


def _is_clip(d):
    return os.path.isfile(os.path.join(d, "meta.json"))


def _clip_dir(path):
    # тека кліпу з Raw Save, або тека, куди Raw Save пише, — тоді найсвіжіший кліп у ній
    d = _folder(path)
    if _is_clip(d):
        return d
    clips = [os.path.join(d, c) for c in os.listdir(d)] if os.path.isdir(d) else []
    clips = [c for c in clips if _is_clip(c)]
    if not clips:
        raise ValueError(f"Raw Load: no depth saved by Raw Save in {d}")
    return max(clips, key=lambda c: os.path.getmtime(os.path.join(c, "meta.json")))


def _write_exr(path, a):
    if not cv2.imwrite(path, np.ascontiguousarray(a, dtype=np.float32)):
        raise RuntimeError(f"EXR not written: {path} (OpenCV without OpenEXR?)")


def _read_exr(path):
    a = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if a is None:
        raise RuntimeError(f"EXR not read: {path}")
    return a.astype(np.float32)


class SVKRawDepthSave:
    DESCRIPTION = ("Saves raw DA3 geometry (float32 EXR per frame + source image) to a new subfolder of 'folder' "
                   "(empty = output/svk_raw). "
                   "Run DA3 once, then grade from Raw Load as many times as you want.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_geometry": ("DA3_GEOMETRY", {"tooltip": "output of DA3Inference"}),
            "name": ("STRING", {"default": "depth", "tooltip": "prefix of the new subfolder: name_00001, name_00002…; a/b/name makes subfolders a/b"}),
            "folder": ("STRING", {"default": "", "tooltip": "where to save; empty = output/svk_raw"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/Depth"

    def save(self, da3_geometry, name, folder):
        g = da3_geometry
        # a/b/name → підтеки a/b, у них name_00001 (як filename_prefix у Save Image)
        parts = ["".join(ch for ch in p.strip() if ch.isalnum() or ch in "-_.") for p in name.replace("\\", "/").split("/")]
        parts = [p for p in parts if p.strip(".")]
        base = parts.pop() if parts else "depth"
        root = os.path.join(_folder(folder), *parts)
        os.makedirs(root, exist_ok=True)
        i = 1
        while os.path.exists(os.path.join(root, f"{base}_{i:05d}")):
            i += 1
        clip = f"{base}_{i:05d}"
        d = os.path.join(root, clip)
        os.makedirs(d)

        depth = g["depth"].cpu().numpy()
        image = g["image"].cpu().numpy()
        n = depth.shape[0]
        extra = [k for k in ("sky", "confidence") if k in g]
        for f in range(n):
            _write_exr(os.path.join(d, f"depth_{f:05d}.exr"), depth[f])
            for k in extra:
                _write_exr(os.path.join(d, f"{k}_{f:05d}.exr"), g[k][f].cpu().numpy())
            rgb = np.clip(image[f, ..., :3] * 65535.0 + 0.5, 0, 65535).astype(np.uint16)
            cv2.imwrite(os.path.join(d, f"image_{f:05d}.png"), rgb[..., ::-1])

        meta = {"frames": n, "mode": g.get("mode", "mono"), "extra": extra,
                "height": int(depth.shape[1]), "width": int(depth.shape[2]),
                "depth_min": float(depth.min()), "depth_max": float(depth.max())}
        for k in ("extrinsics", "intrinsics"):
            if k in g:
                meta[k] = g[k].cpu().float().tolist()
        with open(os.path.join(d, "meta.json"), "w") as fh:
            json.dump(meta, fh)
        msg = f"{d} · {n} frame(s) {meta['width']}×{meta['height']} float32"
        print(f"[SVK Raw Save] {msg}")
        return {"ui": {"text": [msg]}}


class SVKRawDepthLoad:
    DESCRIPTION = ("Loads raw depth saved by Raw Save back as DA3_GEOMETRY (float32). "
                   "Point 'folder' at a saved clip, or at the folder Raw Save writes to — the newest clip there is taken. "
                   "Feed it to Depth Receiver, or to DA3 mesh / point cloud nodes.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"folder": ("STRING", {"default": "",
                                                   "tooltip": "clip folder, or the Raw Save folder (newest clip); empty = output/svk_raw"})}}

    RETURN_TYPES = ("DA3_GEOMETRY",)
    RETURN_NAMES = ("da3_geometry",)
    FUNCTION = "load"
    CATEGORY = "StepVisionKit/Depth"

    @classmethod
    def IS_CHANGED(cls, folder):
        try:
            d = _clip_dir(folder)
        except ValueError:
            return ""
        return d + str(os.path.getmtime(os.path.join(d, "meta.json")))

    def load(self, folder):
        d = _clip_dir(folder)
        print(f"[SVK Raw Load] {d}")
        with open(os.path.join(d, "meta.json")) as fh:
            meta = json.load(fh)
        n = meta["frames"]
        g = {"mode": meta["mode"],
             "depth": torch.from_numpy(np.stack([_read_exr(os.path.join(d, f"depth_{f:05d}.exr")) for f in range(n)]))}
        for k in meta["extra"]:
            g[k] = torch.from_numpy(np.stack([_read_exr(os.path.join(d, f"{k}_{f:05d}.exr")) for f in range(n)]))
        imgs = []
        for f in range(n):
            a = cv2.imread(os.path.join(d, f"image_{f:05d}.png"), cv2.IMREAD_UNCHANGED)
            imgs.append(a[..., ::-1].astype(np.float32) / 65535.0)
        g["image"] = torch.from_numpy(np.ascontiguousarray(np.stack(imgs)))
        for k in ("extrinsics", "intrinsics"):
            if k in meta:
                g[k] = torch.tensor(meta[k], dtype=torch.float32)
        return (g,)


class SVKDepthReceiver:
    DESCRIPTION = ("Raw DA3 geometry → depth image for Depth Grade, float32, near is white. "
                   "Full model range (min…max over the clip), nothing clipped — trim and levels belong to Depth Grade. "
                   "Use instead of DA3Render, which clips 1%/99% and is often saved as 8-bit.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_geometry": ("DA3_GEOMETRY", {"tooltip": "DA3Inference or Raw Load"}),
            "sky_to_far": ("BOOLEAN", {"default": True,
                                       "tooltip": "Mono/Metric models: sky pixels go to the far end and do not stretch the range"}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("depth",)
    FUNCTION = "receive"
    CATEGORY = "StepVisionKit/Depth"

    def receive(self, da3_geometry, sky_to_far):
        d = da3_geometry["depth"].float()
        sky = da3_geometry.get("sky") if sky_to_far else None
        lo, hi = float("inf"), float("-inf")
        if sky is not None:                                    # межі по землі покадрово: без маски-копії на весь кліп
            for i in range(d.shape[0]):
                g = d[i][sky[i] < 0.3]
                if g.numel():
                    lo, hi = min(lo, float(g.min())), max(hi, float(g.max()))
        if lo > hi:
            lo, hi = float(d.min()), float(d.max())
        v = d.sub(lo).div_(max(hi - lo, 1e-12)).clamp_(0.0, 1.0).neg_().add_(1.0)   # одна копія, далі на місці
        if sky is not None:
            v.masked_fill_(sky >= 0.3, 0.0)
        # 3 канали - вʼю (stride 0): кліп 1080p не множиться втричі в RAM; споживачі лише читають
        return (v.unsqueeze(-1).expand(-1, -1, -1, 3),)


class SVKSaveDepth16:
    DESCRIPTION = ("Saves images without 8-bit loss: 16-bit PNG (65536 levels) or float32 EXR. "
                   "A batch becomes a numbered sequence. SaveImage writes 8-bit — that is where banding came from.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "filename_prefix": ("STRING", {"default": "depth16/depth"}),
            "format": (["png 16-bit", "exr float32"], {"default": "png 16-bit"}),
            "folder": ("STRING", {"default": "", "tooltip": "where to save; empty = ComfyUI output"}),
        }}

    RETURN_TYPES = ()
    FUNCTION = "save"
    OUTPUT_NODE = True
    CATEGORY = "StepVisionKit/Depth"

    def save(self, images, filename_prefix, format, folder=""):
        if format not in ("png 16-bit", "exr float32"):
            raise ValueError(f"unknown format {format}")
        a = images.cpu().numpy().astype(np.float32)
        h, w = a.shape[1:3]
        out_dir = _folder(folder) if folder.strip() else folder_paths.get_output_directory()
        full, fname, counter, sub, _ = folder_paths.get_save_image_path(filename_prefix, out_dir, w, h)
        gray = a.shape[-1] == 1 or bool(np.all(a[..., 0] == a[..., 1]) and np.all(a[..., 0] == a[..., 2]))
        ui = []
        for f in range(a.shape[0]):
            px = a[f, ..., 0] if gray else a[f, ..., :3][..., ::-1]
            if format == "png 16-bit":
                file = f"{fname}_{counter:05}_.png"
                cv2.imwrite(os.path.join(full, file), np.clip(px * 65535.0 + 0.5, 0, 65535).astype(np.uint16))
                if not folder.strip():
                    ui.append({"filename": file, "subfolder": sub, "type": "output"})
            else:
                file = f"{fname}_{counter:05}_.exr"
                _write_exr(os.path.join(full, file), px)
            counter += 1
        print(f"[SVK Save 16] {a.shape[0]} file(s) → {os.path.join(full, fname)}_*  ({format})")
        return {"ui": {"images": ui}} if ui else {"ui": {"text": [f"{a.shape[0]} file(s) → {os.path.join(full, fname)}_*"]}}


# ───────────────────────── лупа 1:1 і карта сходинок ─────────────────────────
# Сходинка = тон стоїть пласко, стрибає на кілька відтінків і знову стоїть: слід 8-бітного джерела.
# Рівний float-градієнт (і білінійний апскейл DA3) дає сусідні перепади однієї величини — не сходинка.
# Справжня межа об'єкта крутіша за _STEP_HI. Розмиті сходинки ловимо на зменшених масштабах.
_STEP_LO, _STEP_HI = 0.25 / 255.0, 16.0 / 255.0


def _jumps(v):
    j = np.zeros(v.shape, bool)
    for ax in (0, 1):
        d = np.abs(np.diff(v, axis=ax))
        p = np.pad(d, [(2, 2) if a == ax else (0, 0) for a in (0, 1)], mode="edge")
        sl = lambda o: p[tuple(slice(2 + o, 2 + o + d.shape[a]) if a == ax else slice(None) for a in (0, 1))]
        side = np.maximum(np.maximum(sl(-2), sl(-1)), np.maximum(sl(1), sl(2)))
        hit = (d >= _STEP_LO) & (d <= _STEP_HI) & (side < 0.25 * d)
        if ax == 0:
            j[:-1] |= hit
        else:
            j[:, :-1] |= hit
    # сходинка — це лінія, а не поодинокий шум: лишаємо там, де стрибків багато поруч
    dens = cv2.boxFilter(j.astype(np.float32), -1, (9, 9))
    return j & (dens >= 4.0 / 81.0)


def step_map(v):
    """bool-карта сходинок на повній роздільності (масштаби 1…16: розмиті «чисткою» смуги ширші)."""
    h, w = v.shape
    m = _jumps(v)
    for k in (2, 4, 8, 16):
        if min(h, w) // k < 32:
            break
        s = cv2.resize(v, (w // k, h // k), interpolation=cv2.INTER_AREA)
        m |= cv2.resize(_jumps(s).astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return m


def relief(v):
    """Рельєф: мілкі перепади тону ×12 над плавною основою. Розмиті «чисткою» смуги тут видно оку
    (детектор їх уже не ловить), рівний float-градієнт лишається рівним сірим."""
    base = cv2.GaussianBlur(v, (0, 0), max(8.0, min(v.shape) / 150.0))
    return 0.5 + (v - base) * 12.0


def _dither_u8(v, rng):
    return np.clip(np.floor(v * 255.0 + rng.random(v.shape, dtype=np.float32)), 0, 255).astype(np.uint8)


def loupe_payload(depth, uid, max_frames):
    """Кадри повної роздільності для лупи: 8-біт PNG з дизерингом (щоб смуги показу монітора
    не плутати зі смугами даних), червона маска сходинок і рельєф. Лише для показу, у ланцюг не йде."""
    a = depth.cpu().numpy()
    a = a[..., :3].mean(axis=3) if a.ndim == 4 else a
    n = a.shape[0]
    idx = sorted(set(np.linspace(0, n - 1, min(n, max_frames)).round().astype(int).tolist()))
    tmp = folder_paths.get_temp_directory()
    os.makedirs(tmp, exist_ok=True)
    tag = f"svk_loupe_{uid}_{int(time.time() * 1000) % 100000000}"
    rng = np.random.default_rng(0)
    frames = []
    for f in idx:
        v = np.ascontiguousarray(a[f], dtype=np.float32)
        u8 = _dither_u8(v, rng)
        m = step_map(v)
        rgba = np.zeros(v.shape + (4,), np.uint8)
        md = cv2.dilate(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        rgba[md] = (40, 40, 255, 230)
        cv2.imwrite(os.path.join(tmp, f"{tag}_{f:05d}.png"), u8)
        cv2.imwrite(os.path.join(tmp, f"{tag}_{f:05d}_steps.png"), rgba)
        cv2.imwrite(os.path.join(tmp, f"{tag}_{f:05d}_relief.png"), _dither_u8(relief(v), rng))
        pct = float(m.mean() * 100.0)
        frames.append({"frame": int(f), "steps_pct": round(pct, 3),
                       "image": {"filename": f"{tag}_{f:05d}.png", "subfolder": "", "type": "temp"},
                       "steps": {"filename": f"{tag}_{f:05d}_steps.png", "subfolder": "", "type": "temp"},
                       "relief": {"filename": f"{tag}_{f:05d}_relief.png", "subfolder": "", "type": "temp"}})
        print(f"[SVK Loupe] frame {f}: steps {pct:.3f}% of pixels")
    return {"width": int(a.shape[2]), "height": int(a.shape[1]), "frames_total": int(n), "frames": frames}


def _pick_folder(start):
    if sys.platform != "darwin":
        return {"ok": False, "message": "Folder picker is macOS only — paste the path into 'folder'"}
    script = ["-e", "activate"]
    if start and os.path.isdir(start):
        script += ["-e", f'set p to choose folder with prompt "Raw depth folder" default location (POSIX file {json.dumps(start)})']
    else:
        script += ["-e", 'set p to choose folder with prompt "Raw depth folder"']
    script += ["-e", "POSIX path of p"]
    r = subprocess.run(["osascript", *script], capture_output=True, text=True)
    if r.returncode:
        return {"ok": False, "cancelled": "-128" in r.stderr, "message": r.stderr.strip()}
    return {"ok": True, "path": r.stdout.strip().rstrip("/")}


try:
    from server import PromptServer
    from aiohttp import web as _web

    @PromptServer.instance.routes.post("/svk/raw/pick_folder")
    async def _svk_raw_pick_folder(request):
        body = await request.json()
        start = _folder(str(body.get("path", "")))
        loop = asyncio.get_running_loop()
        return _web.json_response(await loop.run_in_executor(None, _pick_folder, start))
except Exception as _e:
    print(f"[SVK Raw] folder picker route not registered: {_e}")
