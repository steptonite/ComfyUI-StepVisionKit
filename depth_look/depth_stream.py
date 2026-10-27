"""Глибина DA3 для відео з кешем (26.10.2026). Звіт: docs/depth_render_efficiency.md.

Замінює ланцюг ImageScaleBy → DA3Inference → Depth Receiver: кадри з входу images масштабуються, проходять DA3
пачками по batch, порядок операцій як у старому ланцюгу, тож вихід збігається з Receiver
(перевірка: depth_look/tests/depth_stream_check.py).
27.10.2026: вибору файлу в ноді немає - вхід лише images. Потокове читання файлу (кліп не в RAM)
прибрано разом із ним; воно є в історії git (2ded716), якщо знадобиться для довгих 2K-кліпів.

Кеш (output/svk_depth_cache/<ключ>/):
  depth_lr.f32 / sky_lr.f32 — сирий вихід моделі на її роздільності (518×294 для 16:9), float32, кадр за кадром;
                              перерваний прохід продовжується з останнього записаного кадру;
  depth_<H>x<W>_<sky>.npy   — нормалізована глибина, як у Receiver (near = 1), float32, mmap.
Ключ = відбиток кадрів + діапазон + масштаб + модель + роздільність + пристрій. Той самий ключ → DA3 не рахується.
Анти-мерехтіння і рівні тут свідомо немає: це Depth Grade.
"""
import hashlib
import json
import os
import queue
import threading

import numpy as np
import torch

import comfy.model_management as mm
import comfy.utils
import folder_paths
from comfy.ldm.depth_anything_3 import preprocess as da3_preprocess


CACHE_DIR = "svk_depth_cache"



def _cache_root(folder):
    p = folder.strip().strip("'\"")
    return os.path.abspath(os.path.expanduser(p)) if p else os.path.join(folder_paths.get_output_directory(), CACHE_DIR)


def _model_sig(model_patcher):
    """Відбиток моделі без хешу всіх ваг: тип, dtype, кількість тензорів і суми трьох тензорів."""
    m = model_patcher.model.diffusion_model
    sd = m.state_dict()
    keys = sorted(sd)
    h = hashlib.sha1(f"{type(m).__name__}|{getattr(m, 'dtype', None)}|{len(keys)}|{getattr(m, 'has_sky', '?')}".encode())
    for k in (keys[0], keys[len(keys) // 2], keys[-1]):
        t = sd[k].detach()
        h.update(f"{k}|{tuple(t.shape)}|{float(t.flatten()[:4096].double().sum()):.10e}".encode())
    return h.hexdigest()[:16]


def _image_frames(images, start_frame, frame_count, every_nth):
    """Джерело кадрів «вхід images»: зріз кліпу, який уже лежить у RAM (кадри не копіюються, лише читаються)."""
    sel = images[int(start_frame)::int(every_nth)]
    if frame_count:
        sel = sel[:int(frame_count)]

    def frames(skip=0):
        for i in range(skip, sel.shape[0]):
            yield sel[i].float().cpu().numpy()
    return sel, frames


def _images_sig(sel):
    """Відбиток кліпу з входу без хешу всіх пікселів: форма + кожен 16-й піксель кожного кадру."""
    h = hashlib.sha1(f"{tuple(sel.shape)}|{sel.dtype}".encode())
    h.update(np.ascontiguousarray(sel[:, ::16, ::16, :].float().cpu().numpy()).tobytes())
    return h.hexdigest()[:16]


def _prefetch(gen, depth=3):
    """Декодування в окремому потоці, поки GPU рахує попередній кадр. Кадри - копії, тож буфер генератора вільний."""
    q, stop = queue.Queue(depth), threading.Event()

    def work():
        try:
            for f in gen:
                if stop.is_set():
                    break
                q.put(f.copy())
        except BaseException as e:   # noqa: BLE001 - помилку декодера віддаємо в головний потік
            q.put(e)
        else:
            q.put(None)
        finally:
            gen.close()
    t = threading.Thread(target=work, daemon=True)
    t.start()
    try:
        while True:
            f = q.get()
            if f is None:
                return
            if isinstance(f, BaseException):
                raise f
            yield f
    finally:
        stop.set()
        while t.is_alive():                                    # розблокувати put, якщо черга повна
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            t.join(0.05)


def _batches(gen, b):
    """Пачки по b кадрів (остання може бути меншою)."""
    buf = []
    for f in gen:
        buf.append(f)
        if len(buf) == b:
            yield buf
            buf = []
    if buf:
        yield buf


class SVKVideoDepthStream:
    DESCRIPTION = ("Depth for a whole video from frames on 'images' (from any loader, e.g. Load Video Pro): frames are scaled and "
                   "run through Depth Anything 3 in passes of 'batch' frames. Same output as Image Scale By → Run DA3 → Depth Receiver. "
                   "Cached: the same frames and settings do not run DA3 again. Runs where ComfyUI runs: CUDA, Apple MPS or CPU. "
                   "Temporal flicker and levels are not touched here: put Depth Grade (grade + anti-flicker) after this node - "
                   "it re-grades without re-running DA3.")

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "da3_model": ("DA3_MODEL", {"tooltip": "Load Depth Anything 3 (Mono-Large gives the sky mask)"}),
            "images": ("IMAGE", {"tooltip": "video frames from any loader (Load Video Pro, image sequence)"}),
            "start_frame": ("INT", {"default": 0, "min": 0, "max": 10_000_000, "tooltip": "first frame to read"}),
            "frame_count": ("INT", {"default": 0, "min": 0, "max": 100_000, "tooltip": "0 = all frames from the first one"}),
            "pre_scale": ("FLOAT", {"default": 0.5, "min": 0.05, "max": 1.0, "step": 0.05,
                                    "tooltip": "frame scale before DA3, like Image Scale By (area). The model works at 'resolution' anyway; "
                                               "this sets the size of the depth map"}),
            "resolution": ("INT", {"default": 518, "min": 140, "max": 2520, "step": 14, "tooltip": "DA3 resolution, longest side"}),
            "sky_to_far": ("BOOLEAN", {"default": True, "tooltip": "as in Depth Receiver: sky goes far and does not stretch the range"}),
            "reuse_cache": ("BOOLEAN", {"default": True, "tooltip": "off = run DA3 again and overwrite the cache"}),
            "batch": ("INT", {"default": 1, "min": 1, "max": 64,
                              "tooltip": "ceiling of frames per DA3 pass. 1 = exactly the old Run DA3 result. 8 was ~13% faster on an RTX 3060 "
                                         "(decode, not GPU, is the limit); depth then differs by fp16 rounding (<=0.004, "
                                         "rare sky-edge pixels flip). Costs VRAM. Keep 1 on an 8 GB Mac"}),
            "ram_budget_gb": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 256.0, "step": 0.5,
                                        "tooltip": "max RAM for the output depth. If the clip fits, depth stays in RAM (fastest); "
                                                   "if not, it goes to a file on disk (mmap). 0 = always disk. Raw DA3 cache is on disk anyway (~1 MB/frame)"}),
        }, "optional": {
            "every_nth": ("INT", {"default": 1, "min": 1, "max": 1000, "tooltip": "take every N-th frame; 1 = all"}),
            "scale_method": (["area", "bilinear", "bicubic", "lanczos", "nearest-exact"], {"default": "area", "tooltip": "filter for pre_scale"}),
            "resize_method": (["upper_bound_resize", "lower_bound_resize"], {"default": "upper_bound_resize", "tooltip": "how DA3 fits the frame to its resolution: upper bound = long side, lower bound = short side"}),
            "cache_folder": ("STRING", {"default": "", "tooltip": "empty = output/svk_depth_cache"}),
        }}

    RETURN_TYPES = ("IMAGE", "INT", "STRING")
    RETURN_NAMES = ("depth", "frame_count", "info")
    OUTPUT_TOOLTIPS = ("float32 depth, near is white, 3 channels (view); lives on disk, not in RAM",
                       "frames", "cache folder and timings")
    FUNCTION = "run"
    CATEGORY = "StepVisionKit/Depth"

    def run(self, da3_model, images, start_frame, frame_count, pre_scale, resolution, sky_to_far, reuse_cache,
            batch=1, ram_budget_gb=2.0, every_nth=1, scale_method="area", resize_method="upper_bound_resize", cache_folder=""):
        device = mm.get_torch_device()
        sel, frames = _image_frames(images, start_frame, frame_count, every_nth)
        if sel.shape[0] == 0:
            raise ValueError(f"Video Depth: no frames on 'images' after start {start_frame} / every {every_nth}.")
        key_src = {"v": VERSION, "images": _images_sig(sel), "start": int(start_frame), "count": int(frame_count),
                   "nth": int(every_nth), "pre_scale": float(pre_scale), "scale_method": scale_method, "res": int(resolution),
                   "resize": resize_method, "model": _model_sig(da3_model), "device": device.type, "batch": int(batch),
                   "torch": torch.__version__.split("+")[0]}
        name, est = "images", int(sel.shape[0])
        key = hashlib.sha1(json.dumps(key_src, sort_keys=True).encode()).hexdigest()[:16]
        d = os.path.join(_cache_root(cache_folder), f"{name}_{key}")
        os.makedirs(d, exist_ok=True)
        meta_p = os.path.join(d, "meta.json")
        meta = {}
        if reuse_cache and os.path.isfile(meta_p):
            with open(meta_p) as fh:
                meta = json.load(fh)
        if not reuse_cache:
            for f in os.listdir(d):
                os.remove(os.path.join(d, f))
        meta["key"] = key_src

        if not meta.get("done"):
            self._infer(da3_model, frames, est, d, meta, meta_p, device, pre_scale, scale_method, resolution,
                        resize_method, int(batch))
        out = self._normalize(d, meta, meta_p, device, sky_to_far, ram_budget_gb)
        n = out.shape[0]
        where = "RAM" if isinstance(out, torch.Tensor) else "disk mmap"
        info = (f"{d} · {n} frame(s) {out.shape[2]}×{out.shape[1]} float32 in {where} · DA3 {meta['lr_w']}×{meta['lr_h']} "
                f"on {device.type} {meta.get('dtype', '')}")
        print(f"[SVK Video Depth] {info}")
        t = out if isinstance(out, torch.Tensor) else torch.from_numpy(out)
        return (t.unsqueeze(-1).expand(-1, -1, -1, 3), n, info)

    # ---------------------------------------------------------------- прохід 1: DA3 кадр за кадром → сирий кеш
    def _infer(self, model, frames, est, d, meta, meta_p, device, pre_scale, scale_method, resolution,
               resize_method, batch=1):
        mm.load_model_gpu(model)
        diffusion = model.model.diffusion_model
        dtype = diffusion.dtype if diffusion.dtype is not None else torch.float32
        files = {"depth": os.path.join(d, "depth_lr.f32"), "sky": os.path.join(d, "sky_lr.f32")}
        done = 0
        if meta.get("lr_h"):                                   # продовження перерваного проходу
            fb = meta["lr_h"] * meta["lr_w"] * 4
            sizes = [os.path.getsize(f) // fb for k, f in files.items() if os.path.isfile(f) and (k == "depth" or meta.get("has_sky"))]
            done = min(sizes) if sizes else 0
            for k, f in files.items():
                if os.path.isfile(f):
                    with open(f, "r+b") as fh:
                        fh.truncate(done * fb)
            if done:
                print(f"[SVK Video Depth] resuming after {done} cached frame(s)")
        pbar = comfy.utils.ProgressBar(max(est, 1))
        fhs = {}
        n = done
        acc = None if done else {}                            # межі кліпу на льоту: [ground lo, hi, all lo, hi] на пристрої
        try:
            last = n
            for rgbs in _batches(_prefetch(frames(skip=done), max(3, batch + 1)), batch):
                mm.throw_exception_if_processing_interrupted()
                x = torch.from_numpy(np.stack(rgbs))                            # (B,H,W,3), як IMAGE з Load Video Pro
                if pre_scale != 1.0:                                            # як ImageScaleBy.upscale
                    smp = x.movedim(-1, 1)
                    x = comfy.utils.common_upscale(smp, round(smp.shape[3] * pre_scale), round(smp.shape[2] * pre_scale),
                                                   scale_method, "disabled").movedim(1, -1)
                single = x.to(device)                                           # далі як _run_da3 у ядрі Comfy
                xin = da3_preprocess.preprocess_image(single, process_res=resolution, method=resize_method).to(dtype=dtype)
                with torch.no_grad():
                    o = diffusion(xin)
                dl = o["depth"].float()
                if not bool(torch.isfinite(dl).all()):                 # fp16/bf16 на MPS чи старих картах може дати NaN/inf
                    raise RuntimeError(f"Video Depth: DA3 returned NaN/inf near frame {n} ({device.type}, {dtype}). "
                                       "Load the model with weight_dtype fp32 (Load Depth Anything 3) and run again.")
                if not fhs:
                    meta.update({"dtype": str(dtype).replace("torch.", ""), "h": int(x.shape[1]), "w": int(x.shape[2]), "lr_h": int(dl.shape[-2]),
                                 "lr_w": int(dl.shape[-1]), "has_sky": "sky" in o})
                    fhs = {k: open(f, "ab") for k, f in files.items() if k == "depth" or meta["has_sky"]}
                fhs["depth"].write(dl.reshape(-1).cpu().numpy().tobytes())
                if meta["has_sky"]:
                    fhs["sky"].write(o["sky"].float().reshape(-1).cpu().numpy().tobytes())
                if acc is not None:                            # той самий розтяг, що в _upsampled / _run_da3
                    H, W = meta["h"], meta["w"]
                    full = torch.nn.functional.interpolate(dl.unsqueeze(1), size=(H, W), mode="bilinear", align_corners=False).squeeze(1)
                    mins = [full.min(), full.max()]
                    if meta["has_sky"]:
                        sf = torch.nn.functional.interpolate(o["sky"].float().unsqueeze(1), size=(H, W), mode="bilinear",
                                                             align_corners=False).squeeze(1)
                        gr = sf < SKY_T
                        mins = [torch.where(gr, full, float("inf")).min(), torch.where(gr, full, float("-inf")).max()] + mins
                    cur = torch.stack(mins)
                    acc["v"] = cur if "v" not in acc else torch.stack([torch.minimum(acc["v"][k], cur[k]) if k % 2 == 0
                                                                        else torch.maximum(acc["v"][k], cur[k]) for k in range(len(mins))])
                n += len(rgbs)
                if n - last >= 16:
                    last = n
                    for fh in fhs.values():
                        fh.flush()
                    meta["frames"] = n
                    with open(meta_p, "w") as fh:
                        json.dump(meta, fh)
                pbar.update_absolute(min(n, max(est, 1)), max(est, n, 1))
        finally:
            for fh in fhs.values():
                fh.close()
        meta.update({"frames": n, "done": True})
        meta.pop("stats", None)
        if acc and "v" in acc:
            v = [float(x) for x in acc["v"].cpu()]
            meta["stats"] = v if meta["has_sky"] else [float("inf"), float("-inf")] + v
        with open(meta_p, "w") as fh:
            json.dump(meta, fh)

    # ---------------------------------------------------------------- прохід 2: межі кліпу і нормалізація як у Receiver
    def _upsampled(self, d, meta, device):
        n, lh, lw, H, W = meta["frames"], meta["lr_h"], meta["lr_w"], meta["h"], meta["w"]
        dep = np.memmap(os.path.join(d, "depth_lr.f32"), np.float32, "r", shape=(n, lh, lw))
        sky = np.memmap(os.path.join(d, "sky_lr.f32"), np.float32, "r", shape=(n, lh, lw)) if meta["has_sky"] else None

        def up(a):                                             # той самий білінійний розтяг, що в _run_da3, на тому ж пристрої
            t = torch.from_numpy(np.array(a)).to(device)[None]
            return torch.nn.functional.interpolate(t.unsqueeze(1), size=(H, W), mode="bilinear",
                                                   align_corners=False).squeeze(1).cpu()[0]
        for i in range(n):
            yield up(dep[i]), (up(sky[i]) if sky is not None else None)

    def _normalize(self, d, meta, meta_p, device, sky_to_far, ram_budget_gb=0.0):
        use_sky = bool(sky_to_far and meta["has_sky"])
        n, H, W = meta["frames"], meta["h"], meta["w"]
        out_p = os.path.join(d, f"depth_{W}x{H}_{'sky' if use_sky else 'nosky'}.npy")
        if os.path.basename(out_p) in meta.get("outs", []) and os.path.isfile(out_p):
            return np.load(out_p, mmap_mode="r+")
        st = meta.get("stats")
        if not st:                                             # межі: по землі (sky < 0.3) і по всьому кліпу - для обох режимів
            glo, ghi, alo, ahi = float("inf"), float("-inf"), float("inf"), float("-inf")
            for dd, ss in self._upsampled(d, meta, device):
                alo, ahi = min(alo, float(dd.min())), max(ahi, float(dd.max()))
                if ss is not None:
                    g = dd[ss < SKY_T]
                    if g.numel():
                        glo, ghi = min(glo, float(g.min())), max(ghi, float(g.max()))
            st = meta["stats"] = [glo, ghi, alo, ahi]
        lo, hi = (st[0], st[1]) if use_sky else (float("inf"), float("-inf"))
        if lo > hi:
            lo, hi = st[2], st[3]
        in_ram = n * H * W * 4 <= ram_budget_gb * 2**30                # влазить у стелю - тримаємо в RAM, інакше файл
        out = torch.empty((n, H, W)) if in_ram else np.lib.format.open_memmap(out_p, mode="w+", dtype=np.float32, shape=(n, H, W))
        pbar = comfy.utils.ProgressBar(n)
        for i, (dd, ss) in enumerate(self._upsampled(d, meta, device)):
            v = dd.sub(lo).div_(max(hi - lo, 1e-12)).clamp_(0.0, 1.0).neg_().add_(1.0)     # рядок у рядок як Receiver
            if use_sky:
                v.masked_fill_(ss >= SKY_T, 0.0)
            out[i] = v if in_ram else v.numpy()
            pbar.update_absolute(i + 1, n)
        if in_ram:
            with open(meta_p, "w") as fh:
                json.dump(meta, fh)
            return out
        out.flush()
        meta["outs"] = sorted(set(meta.get("outs", [])) | {os.path.basename(out_p)})
        with open(meta_p, "w") as fh:
            json.dump(meta, fh)
        return out


NODE_CLASS_MAPPINGS = {"SVKVideoDepthStream": SVKVideoDepthStream}
NODE_DISPLAY_NAME_MAPPINGS = {"SVKVideoDepthStream": "Video Depth · DA3 stream"}
