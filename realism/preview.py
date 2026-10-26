"""Превʼю картки Camera Realism. Run кладе вхідні кадри в памʼять під id ноди (до 24 поспіль із середини кліпу);
картка просить малі кадри (весь шматок - анімація) і лупу 1:1 (повнороздільні кадри, кеш на останній набір параметрів)."""
import base64
import hashlib
import json
import threading
import time

import cv2
import numpy as np

from . import params as P
from .engine import apply_realism

MAX_FRAMES = 24          # шматок відео в памʼяті для анімації
LOUPE_FRAMES = 8         # кадрів у лупі (відео): повна роздільність, тому менше
MEM_BUDGET = 300e6       # байт на кадри однієї ноди
_CACHE = {}              # node -> {"stamp","frames"(u8 B,H,W,3),"depth"(f32|None),"start"}
_LOUPE = {}              # node -> (ключ параметрів, кадри u8)
_LOCK = threading.Lock()


class NeedRun(Exception):
    """Кадрів цієї ноди нема в памʼяті: її треба запустити (Run) хоч раз."""


def _key(node):
    k = "".join(ch for ch in str(node) if ch.isalnum() or ch in ":_-")[:64]
    if not k:
        raise ValueError("bad node id")
    return k


def store(node, frames, depth, frame_offset=0):
    """Після Run: frames float (B,H,W,3) 0..1, depth (B|1,H,W) або None. Повертає meta для ui."""
    B, H, W = frames.shape[:3]
    m = max(1, min(MAX_FRAMES, int(MEM_BUDGET // (H * W * 3))))      # 4K: ~12 кадрів, 1080p: усі 24
    s = max(0, (B - m) // 2)
    fr = np.clip(frames[s:s + m] * 255.0 + 0.5, 0, 255).astype(np.uint8)
    dep = None
    if depth is not None:
        dep = depth[s:s + m] if len(depth) == B and B > 1 else depth[:1]
        dep = np.ascontiguousarray(dep, dtype=np.float32)
    stamp = "%x" % int(time.time() * 1000)
    k = _key(node)
    with _LOCK:
        _CACHE[k] = {"stamp": stamp, "frames": fr, "depth": dep, "start": int(frame_offset) + s}
        _LOUPE.pop(k, None)
    return {"stamp": stamp, "n": int(len(fr)), "total": int(B), "w": int(fr.shape[2]), "h": int(fr.shape[1])}


def _get(node):
    c = _CACHE.get(_key(node))
    if c is None:
        raise NeedRun()
    return c


def resolve_body(body):
    """Тіло запиту картки -> (params, seed). Значення повзунків - як у Run: профіль, сила, Custom/власне зерно з values."""
    vals = {k: v for k, v in (body.get("values") or {}).items() if k in P.NAMES and isinstance(v, (int, float))}
    prof = body.get("profile") if body.get("profile") in P.PROFILE_NAMES else P.DEFAULT_PROFILE
    return P.resolve(prof, float(body.get("strength", 0.5)), vals), int(body.get("seed", 1))


def _jpg(a, q=88):
    return base64.b64encode(cv2.imencode(".jpg", a[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()).decode()


def _png(a):
    return base64.b64encode(cv2.imencode(".png", a[..., ::-1])[1].tobytes()).decode()


def _resize(a, w, h, interp=cv2.INTER_AREA):
    return np.stack([cv2.resize(x, (w, h), interpolation=interp) for x in a])


def _run(frames_u8, depth, p, seed, start):
    out = apply_realism(frames_u8.astype(np.float32) / 255.0, depth, p, seed=seed, frame_offset=start)
    return np.clip(out * 255.0 + 0.5, 0, 255).astype(np.uint8)


def render(node, body):
    """Малі кадри всього шматка (≤ size по довшій стороні). orig=true - ще й оригінали (картка бере їх раз на Run)."""
    c = _get(node)
    p, seed = resolve_body(body)
    fr = c["frames"]
    B, H, W = fr.shape[:3]
    size = max(160, min(768, int(body.get("size", 512))))
    sc = min(1.0, size / max(H, W))
    w, h = max(2, int(round(W * sc))), max(2, int(round(H * sc)))
    small = _resize(fr, w, h) if sc < 1.0 else fr
    dep = c["depth"]
    if dep is not None and sc < 1.0:
        dep = _resize(dep, w, h)
    t = time.time()
    with _LOCK:
        out = _run(small, dep, p, seed, c["start"])
    res = {"stamp": c["stamp"], "w": w, "h": h, "fw": W, "fh": H, "ms": int((time.time() - t) * 1000), "frames": [_jpg(x) for x in out]}
    if body.get("orig"):
        res["orig"] = [_jpg(x) for x in small]
    return res


def loupe(node, body):
    """Кроп crop×crop навколо (x, y) з повнороздільних кадрів [frame, frame+count): оригінал і результат, PNG (зерно без JPEG)."""
    c = _get(node)
    p, seed = resolve_body(body)
    fr = c["frames"]
    B, H, W = fr.shape[:3]
    f0 = max(0, min(B - 1, int(body.get("frame", 0))))
    n = max(1, min(LOUPE_FRAMES, int(body.get("count", 1)), B - f0))
    crop = max(16, min(512, int(body.get("crop", 160))))
    ck = hashlib.sha1(json.dumps([c["stamp"], p, seed, f0, n], sort_keys=True).encode()).hexdigest()
    k = _key(node)
    hit = _LOUPE.get(k)
    if hit is None or hit[0] != ck:
        dep = c["depth"]
        if dep is not None and len(dep) == B and B > 1:
            dep = dep[f0:f0 + n]
        with _LOCK:
            out = _run(fr[f0:f0 + n], dep, p, seed, c["start"] + f0)
            _LOUPE[k] = hit = (ck, out)
    cx = int(round(float(body.get("x", 0.5)) * W)); cy = int(round(float(body.get("y", 0.5)) * H))
    cw, chh = min(crop, W), min(crop, H)
    x0 = max(0, min(W - cw, cx - cw // 2)); y0 = max(0, min(H - chh, cy - chh // 2))
    sl = (slice(y0, y0 + chh), slice(x0, x0 + cw))
    return {"stamp": c["stamp"], "x0": x0, "y0": y0, "w": cw, "h": chh, "frame": f0,
            "orig": [_png(x[sl]) for x in fr[f0:f0 + n]], "proc": [_png(x[sl]) for x in hit[1]]}
