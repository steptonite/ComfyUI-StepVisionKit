"""look_api — спільна логіка API тюнера look (без HTTP-фреймворку).

Один код для двох оболонок:
  * look_tool.py tune  — stdlib http.server на 127.0.0.1:8777 (автономний тюнер);
  * comfy_depth_look   — aiohttp-маршрути /depth_look/... усередині ComfyUI (тюнер у ноді).
LookApi.handle(method, raw_path, body) -> (status, content_type, bytes, extra_headers).

Режим ноди (node_id != None): джерело кадру — кеш, який нода пише при виконанні графа
(_cache/comfy_node_<id>_{img,depth}.npy + _meta.json): до 8 кадрів з батча, float16.
Без кешу кадра нема: рендер віддає 409 «запусти граф раз…» (жодних фото-фолбеків).

Лупа (POST /api/render_crop): віконце 100% — ділянка кадру в робочій роздільності. Рушій поки не вміє рахувати
ділянку окремо (блюр, копії, шлейф, хвиля, серпанок залежать від усього кадру), тому точний шлях такий:
повний кадр у робочій роздільності рахується ОДИН раз на набір параметрів (окремий рендерер, кеш шарів превʼю
не чіпає), а далі кожна ділянка — миттєвий зріз. Нові параметри = новий розрахунок лише коли лупа ввімкнена.
"""
import io, json, os, sys, threading, time
from urllib.parse import urlparse, parse_qs, unquote
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")   # EXR-глибина: до імпорту cv2
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import look, look_params as LP

PREVIEW = 720
IMG_EXT = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".exr")
CODE_DIR = os.path.dirname(os.path.realpath(__file__))    # StepVisionKit/depth_look (і з look_proto через симлінк)
CACHE_DIR = os.path.join(CODE_DIR, "_cache")
GRAD_DIR = os.path.join(CODE_DIR, "gradients")
NODE_MAX_FRAMES = 8          # скільки кадрів батча тримає кеш ноди
NODE_LONG_SIDE = 2144        # робоча роздільність рушія; кеш не більший
NEED_RUN_MSG = "run the graph once so the preview picks up your frame"


class NeedRun(Exception):
    pass


# ---------------------------------------------------------------- глибина кадру (спільне для ноди й превʼю)
def prepare_frame_depth(rgb, d, clean=True, light_is_near=True, levels=None):
    """Глибина кадру -> float32 HxW під rgb. Єдиний шлях і для apply_numpy, і для превʼю в ноді.
    levels=None: покадровий розтяг (як було); levels=(lo,hi): рівні з усього кліпу (look.clip_levels), без мерехтіння."""
    H, W = rgb.shape[:2]
    if clean:
        if levels is None:
            return look.prepare_depth(d, rgb, light_is_near=light_is_near)
        return look.apply_levels(look.prepare_depth(d, rgb, light_is_near=light_is_near, stretch=False), *levels)
    d = np.clip(d, 0, 1)
    if d.shape != (H, W):
        d = cv2.resize(d, (W, H), interpolation=cv2.INTER_CUBIC)
    if not light_is_near:
        d = 1.0 - d
    return np.ascontiguousarray(d, np.float32)


# ---------------------------------------------------------------- кеш кадрів ноди
def _safe_id(node_id):
    return "".join(ch for ch in str(node_id) if ch.isalnum() or ch in "_-") or "0"


def node_paths(node_id):
    b = os.path.join(CACHE_DIR, "comfy_node_%s_" % _safe_id(node_id))
    return b + "img.npy", b + "depth.npy", b + "meta.json"


def _cap(im, long_side=NODE_LONG_SIDE):
    h, w = im.shape[:2]
    if max(h, w) <= long_side:
        return im
    f = long_side / max(h, w)
    return cv2.resize(im, (max(2, int(round(w * f))), max(2, int(round(h * f)))), interpolation=cv2.INTER_AREA)


def _atomic_save(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def save_node_cache(node_id, frames_np, depths_np, clean_depth, light_is_near, *legacy,
                    fps=None, start_time=None, depth_levels="clip"):
    """frames_np (B,H,W,C), depths_np (B|1,h,w) -> до 8 рівномірних кадрів батча у float16. Повертає meta.
    fps/start_time потрібні лише відео-ноді (у статичній їх нема). Стара позиційна форма теж працює:
    (..., fps, start_time[, depth_levels]) або (..., depth_levels)."""
    if legacy:
        if isinstance(legacy[0], str):
            depth_levels = legacy[0]
        else:
            fps = legacy[0]
            if len(legacy) > 1:
                start_time = legacy[1]
            if len(legacy) > 2:
                depth_levels = legacy[2]
    B = frames_np.shape[0]
    orig_wh = [int(frames_np.shape[2]), int(frames_np.shape[1])]       # до зменшення під кеш: лупа чесно підпише масштаб
    K = min(NODE_MAX_FRAMES, B)
    idx = sorted(set(int(round(v)) for v in np.linspace(0, B - 1, K)))
    imgs = [_cap(np.clip(np.asarray(frames_np[i][..., :3], np.float32), 0, 1)).astype(np.float16) for i in idx]
    dB = depths_np.shape[0]
    didx = idx if dB == B else [0]
    deps = [_cap(np.asarray(depths_np[i], np.float32)).astype(np.float16) for i in didx]
    ip, dp, mp = node_paths(node_id)
    os.makedirs(CACHE_DIR, exist_ok=True)
    _atomic_save(ip, np.stack(imgs)); del imgs
    _atomic_save(dp, np.stack(deps)); del deps
    meta = dict(indices=idx, batch=int(B), depth_per_frame=(dB == B), clean_depth=bool(clean_depth),
                light_is_near=bool(light_is_near), depth_levels=str(depth_levels), orig=orig_wh, stamp=time.time_ns())
    if fps is not None:
        meta["fps"] = float(fps)
    if start_time is not None:
        meta["start_time"] = float(start_time)
    tmp = mp + ".tmp"
    json.dump(meta, open(tmp, "w"))
    os.replace(tmp, mp)
    return meta


def read_node_meta(node_id):
    mp = node_paths(node_id)[2]
    if not os.path.isfile(mp):
        return None
    try:
        return json.load(open(mp))
    except Exception:
        return None


_LEVELS = {}


def _node_levels(node_id, meta, imgs, deps):
    """Рівні глибини кліпу для превʼю: ті самі ≤8 кадрів, що й clip_levels бачить у run() (у ≤1080 для швидкості)."""
    key = (node_id, meta["stamp"])
    if key not in _LEVELS:
        _LEVELS.clear()
        preps = []
        for k in range(imgs.shape[0]):
            rgb = _cap(np.clip(np.asarray(imgs[k], np.float32), 0, 1), 1080)
            d = np.asarray(deps[k if deps.shape[0] > 1 else 0], np.float32)
            preps.append(look.prepare_depth(d, rgb, light_is_near=meta["light_is_near"], stretch=False))
        _LEVELS[key] = look.clip_levels(preps)
    return _LEVELS[key]


def load_node_frame(node_id, k):
    """-> (rgb f32, depth_clean f32) k-го кешованого кадру. Читає лише потрібний кадр (mmap)."""
    meta = read_node_meta(node_id)
    if meta is None:
        raise NeedRun(NEED_RUN_MSG)
    ip, dp, _ = node_paths(node_id)
    imgs = np.load(ip, mmap_mode="r")
    deps = np.load(dp, mmap_mode="r")
    k = max(0, min(int(k), imgs.shape[0] - 1))
    rgb = np.clip(np.asarray(imgs[k], np.float32), 0, 1)
    d = np.asarray(deps[k if deps.shape[0] > 1 else 0], np.float32)
    levels = None
    if meta["clean_depth"] and meta.get("depth_levels", "clip") == "clip":
        levels = _node_levels(node_id, meta, imgs, deps)
    return rgb, prepare_frame_depth(rgb, d, meta["clean_depth"], meta["light_is_near"], levels)


# ---------------------------------------------------------------- рендерер
class Renderer:
    """Тримає кадр і кеш лінзових шарів: кольорові повзунки не перераховують блюр."""

    def __init__(self):
        self.frames, self.cache_key, self.cache = {}, None, None
        self.full = {}      # src -> (W, H) робочої роздільності кадру до зменшення під превʼю

    def reset(self):
        self.frames.clear(); self.full.clear(); self.cache_key, self.cache = None, None

    def frame(self, src, long_side):
        key = (src, long_side)
        if key not in self.frames:
            self.frames.clear()
            kind, a, b = src
            if kind == "photo":
                rgb, d = LP.load_frame(photo=a)
            elif kind == "npy":
                rgb, d = load_node_frame(a, b[1])
            else:
                rgb, d = LP.load_frame(image=a, depth=b)
            self.full[src] = (d.shape[1], d.shape[0])
            self.frames[key] = LP.fit(rgb, d, long_side)
        return self.frames[key]

    def render(self, src, p, long_side=PREVIEW):
        rgb, d = self.frame(src, long_side)
        kw = LP.kwargs(p)
        lk = (src, long_side) + tuple(kw[k] for k in LP.LENS_KEYS)
        if lk != self.cache_key:
            self.cache = look.prepare_lens(rgb, d, **{k: kw[k] for k in LP.LENS_KEYS})
            self.cache_key = lk
        return look.depth_look_lens(rgb, d, cache=self.cache, **kw)


# ---------------------------------------------------------------- бібліотека градієнтів
def _hexpal(lst):
    return [[t, [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]] for h, t in lst]


def builtin_grads():
    out = []
    try:
        pal = LP.load_preset("look_v1")["palette"]
        out.append(("look v1", [[float(t), [float(x) for x in c]] for t, c in pal]))
    except Exception:
        pass
    out += [
        ("Ember", _hexpal([("#0b0604", 0), ("#7c0303", .35), ("#c54202", .65), ("#ffc9a0", 1)])),
        ("Abyss", _hexpal([("#01091c", 0), ("#03035b", .35), ("#0025ac", .65), ("#5fe0e0", 1)])),
        ("Spectrum", _hexpal([("#000000", 0), ("#3a0ca3", .2), ("#0077ff", .4), ("#00d4a0", .55),
                            ("#ffd000", .75), ("#ff3c00", .9), ("#ffffff", 1)])),
        ("Black→red", _hexpal([("#000000", 0), ("#5a0000", .5), ("#ff2a00", 1)])),
    ]
    try:
        import variants
        for i in range(1, 8):
            out.append(("Palette %d" % i, variants.jsonable_pal(variants.PALETTES[i])))
    except Exception:
        pass
    return out


def safe_grad_name(name):
    n = "".join(ch for ch in str(name) if ch.isalnum() or ch in "_- ").strip()
    if not n:
        raise ValueError("empty name")
    return n


def clean_pal(pal):
    out = sorted(([min(1.0, max(0.0, float(t))), [min(1.0, max(0.0, float(x))) for x in c[:3]]] for t, c in pal),
                 key=lambda s: s[0])
    if len(out) < 2 or any(len(s[1]) != 3 for s in out):
        raise ValueError("palette: at least 2 RGB stops are required")
    return out


def list_grads():
    res = [dict(name=n, palette=p, builtin=True) for n, p in builtin_grads()]
    if os.path.isdir(GRAD_DIR):
        for f in sorted(os.listdir(GRAD_DIR)):
            if not f.endswith(".json"):
                continue
            try:
                j = json.load(open(os.path.join(GRAD_DIR, f), encoding="utf-8"))
                res.append(dict(name=j.get("name") or f[:-5], palette=clean_pal(j["palette"]), builtin=False))
            except Exception:
                pass
    return res


def _jpeg(img8, q=90):
    return cv2.imencode(".jpg", img8, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


# ---------------------------------------------------------------- API
class LookApi:
    def __init__(self, photo=None, image=None, depth=None, preset="look_v1", node_id=None, anim=False):
        self.R = Renderer()
        self.node_id = node_id
        self.anim = bool(anim)           # False = статична нода: відео-групи (LP.ANIM_GROUPS) у тюнері не показуємо
        self.lock = threading.RLock()
        self.crop_lock = threading.Lock()   # лупа рахує окремо від превʼю: повзунки не чекають її повного кадру
        self.RL, self.full_img, self.full_key = None, None, None
        if node_id is not None:
            src = None
        else:
            src = ("photo", photo, None) if photo else ("file", image, depth)
        self.state = dict(src=src, preset=preset, size=PREVIEW, frame=0)

    # ---- допоміжне
    def _src_name(self, src):
        if src is None or src[0] == "npy":
            return "frame from the graph"
        return src[1] if src[0] == "photo" else os.path.splitext(os.path.basename(src[1]))[0]

    def _src(self):
        """Поточне джерело. У ноді — з кешу графа (stamp у ключі: новий запуск = новий кадр)."""
        if self.node_id is None:
            return self.state["src"]
        meta = read_node_meta(self.node_id)
        if meta is None:
            raise NeedRun(NEED_RUN_MSG)
        k = max(0, min(int(self.state["frame"]), len(meta["indices"]) - 1))
        self.state["frame"] = k
        return ("npy", self.node_id, (meta["stamp"], k))

    def _cur(self, size=None):
        sz = int(size) if size else self.state["size"]
        return self.R.frame(self._src(), sz or PREVIEW)

    def _node_info(self):
        meta = read_node_meta(self.node_id)
        if meta is None:
            return dict(embed=True, node_frame=False, need_run=NEED_RUN_MSG)
        info = dict(embed=True, node_frame=True, frames=meta["indices"], batch=meta["batch"], frame=self.state["frame"],
                    orig=meta.get("orig"))
        for k in ("fps", "start_time"):      # лише відео-нода
            if meta.get(k) is not None:
                info[k] = meta[k]
        return info

    def _params(self, p):
        """У ноді зерно живе: seed + номер кадра батча (як у run()); так превʼю кадру N = кадр N у відео."""
        if self.node_id is None:
            return p
        meta = read_node_meta(self.node_id)
        if not meta:
            return p
        k = max(0, min(int(self.state["frame"]), len(meta["indices"]) - 1))
        p = dict(p)
        p["seed"] = int(round(float(p.get("seed", 7)))) + int(meta["indices"][k])
        return p

    def _set_frame(self, v):
        if self.node_id is not None and v is not None:
            try:
                self.state["frame"] = max(0, int(v))
            except Exception:
                pass

    @staticmethod
    def _j(code, obj, headers=None):
        return code, "application/json", json.dumps(obj, ensure_ascii=False).encode(), headers or {}

    # ---- лупа
    def _render_crop(self, q):
        """{params, cx, cy, size, frame, [free]} -> PNG ділянки size×size пікселів кадру (1 піксель = 1 піксель
        робочої роздільності), центр у (cx, cy) ∈ 0..1. Біля краю вікно зсувається всередину кадру;
        заголовки X-Crop-X/Y повертають його реальний кут, щоб сторінка вирівняла центр."""
        with self.crop_lock:
            if q.get("free"):                                    # лупу вимкнено: віддати памʼять
                self.RL, self.full_img, self.full_key = None, None, None
                return self._j(200, {"ok": True})
            self._set_frame(q.get("frame"))
            src = self._src()
            params = self._params(q["params"])
            key = (src, json.dumps(params, sort_keys=True, default=str))
            t0, cached = time.time(), True
            if self.full_key != key:
                cached = False
                if self.RL is None:
                    self.RL = Renderer()
                self.full_img = LP.to8(self.RL.render(src, params, 0))     # 0 = робоча роздільність кадру
                self.full_key = key
            img = self.full_img
            H, W = img.shape[:2]
            size = max(64, min(1024, int(q.get("size", 480))))
            w, h = min(size, W), min(size, H)
            x0 = int(round(min(max(float(q.get("cx", 0.5)) * W - w / 2, 0), W - w)))
            y0 = int(round(min(max(float(q.get("cy", 0.5)) * H - h / 2, 0), H - h)))
            png = cv2.imencode(".png", img[y0:y0 + h, x0:x0 + w], [cv2.IMWRITE_PNG_COMPRESSION, 1])[1].tobytes()
            orig = (read_node_meta(self.node_id) or {}).get("orig") if self.node_id is not None else None
            ow, oh = (orig or [W, H])[:2]
            hdr = {"X-Crop-X": str(x0), "X-Crop-Y": str(y0), "X-Crop-W": str(w), "X-Crop-H": str(h),
                   "X-Full-W": str(W), "X-Full-H": str(H), "X-Orig-W": str(ow), "X-Orig-H": str(oh),
                   "X-Cached": "1" if cached else "0", "X-Render-Ms": str(int((time.time() - t0) * 1000))}
            return 200, "image/png", png, hdr

    # ---- вхід
    def handle(self, method, raw_path, body=b""):
        u = urlparse(raw_path)
        path, qs = u.path, parse_qs(u.query)
        if method == "POST" and path == "/api/render_crop":      # лупа: свій замок, без блокування превʼю
            try:
                return self._render_crop(json.loads(body or b"{}"))
            except NeedRun as e:
                return self._j(409, {"err": str(e), "need_run": True})
            except Exception as e:
                return self._j(400, {"err": str(e)})
        with self.lock:
            try:
                if method == "GET":
                    r = self._get(path, qs)
                else:
                    r = self._post(path, json.loads(body or b"{}"))
            except NeedRun as e:
                return self._j(409, {"err": str(e), "need_run": True})
            except Exception as e:
                return self._j(400, {"err": str(e)})
        return r if r is not None else self._j(404, {"err": "no"})

    def _get(self, path, qs):
        if path == "/api/state":
            st = self.state
            d = dict(spec=LP.spec_json(self.anim), groups=LP.groups_json(self.anim), anim=self.anim,
                     extra_groups={"Preview": dict(icon=LP.GROUP_ICON.get("Preview", ""), hint=LP.GROUP_HINT.get("Preview", ""))},
                     presets=LP.list_presets(), locked=sorted(LP.LOCKED),
                     preset_name=st["preset"], preset=LP.load_preset(st["preset"]))
            if self.node_id is not None:
                d.update(photos=[], photo="", name="frame from the graph", image="", depth="", **self._node_info())
            else:
                src = st["src"]
                d.update(photos=LP.photo_list(), photo=src[1] if src[0] == "photo" else "",
                         name=self._src_name(src), image=src[1] if src[0] == "file" else "",
                         depth=src[2] if src[0] == "file" else "")
            return self._j(200, d)
        if path == "/api/gradients":
            return self._j(200, list_grads())
        if path.startswith("/api/preset/"):
            return self._j(200, LP.load_preset(unquote(path.split("/api/preset/", 1)[1])))
        if path in ("/api/depth_stats", "/api/depth.jpg", "/api/original.jpg"):
            self._set_frame(qs.get("frame", [None])[0])
            rgb, d = self._cur(qs.get("size", [None])[0])
            if path == "/api/depth_stats":
                p5, p50, p95 = (float(v) for v in np.percentile(d, [5, 50, 95]))
                return self._j(200, dict(p5=p5, p50=p50, p95=p95, w=d.shape[1], h=d.shape[0]))
            if path == "/api/depth.jpg":
                img = cv2.cvtColor((np.clip(d, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_GRAY2BGR)
            else:
                img = LP.to8(rgb)
            return 200, "image/jpeg", _jpeg(img, 92), {}
        return None

    def _post(self, path, q):
        st, R = self.state, self.R
        if path == "/api/render":
            if self.node_id is None and q.get("photo") and q["photo"] in LP.photo_list():
                st["src"] = ("photo", q["photo"], None)
            self._set_frame(q.get("frame"))
            size = int(q.get("size", PREVIEW))
            t0 = time.time()
            src = self._src()
            out = R.render(src, self._params(q["params"]), size)
            st["size"] = size
            buf = _jpeg(LP.to8(out))
            fw, fh = R.full.get(src, (0, 0))
            hdr = {"X-Render-Ms": str(int((time.time() - t0) * 1000)), "X-Frame-W": str(fw), "X-Frame-H": str(fh),
                   "X-Preview-W": str(out.shape[1]), "X-Preview-H": str(out.shape[0])}
            return 200, "image/jpeg", buf, hdr
        if path == "/api/depth_at":
            self._set_frame(q.get("frame"))
            _, d = self._cur(q.get("size"))
            H, W = d.shape
            x = min(W - 1, max(0, int(float(q["x"]) * W)))
            y = min(H - 1, max(0, int(float(q["y"]) * H)))
            win = d[max(0, y - 2):y + 3, max(0, x - 2):x + 3]
            return self._j(200, dict(depth=float(np.median(win)), x=x, y=y, w=W, h=H))
        if path == "/api/load":
            if self.node_id is not None:
                raise ValueError("in the node, the frame comes from the image/depth inputs")

            def _chk(p, what):
                p = os.path.abspath(os.path.expanduser(str(p or "").strip()))
                if not os.path.isfile(p):
                    raise ValueError("%s: file not found: %s" % (what, p))
                if not p.lower().endswith(IMG_EXT):
                    raise ValueError("%s: extension must be %s" % (what, "/".join(e[1:] for e in IMG_EXT)))
                return p
            image, depth = _chk(q.get("image"), "frame"), _chk(q.get("depth"), "depth")
            new = ("file", image, depth)
            R.reset()
            try:
                R.frame(new, st["size"] or PREVIEW)
            except Exception:
                R.reset()
                raise
            st["src"] = new
            fw, fh = R.full[new]
            return self._j(200, dict(w=fw, h=fh, name=self._src_name(new)))
        if path == "/api/save":
            return self._j(200, {"path": LP.save_preset(q["name"], q["params"])})
        if path == "/api/gradients/save":
            name = safe_grad_name(q["name"])
            if name.lower() in [n.lower() for n, _ in builtin_grads()]:
                raise ValueError("\"%s\" is a built-in gradient, save it under a different name" % name)
            os.makedirs(GRAD_DIR, exist_ok=True)
            p = os.path.join(GRAD_DIR, name + ".json")
            json.dump(dict(name=name, palette=clean_pal(q["palette"])), open(p, "w", encoding="utf-8"),
                      indent=1, ensure_ascii=False)
            return self._j(200, {"path": p, "name": name})
        if path == "/api/gradients/delete":
            name = safe_grad_name(q["name"])
            if name.lower() in [n.lower() for n, _ in builtin_grads()]:
                raise ValueError("a built-in gradient cannot be deleted")
            p = os.path.join(GRAD_DIR, name + ".json")
            if not os.path.isfile(p):
                raise ValueError("no such custom gradient")
            os.remove(p)
            return self._j(200, {"ok": True})
        if path == "/api/full":
            self._set_frame(q.get("frame"))
            src = self._src()
            out = R.render(src, self._params(q["params"]), 0)
            name = self._src_name(src)
            od = os.path.join(HERE, "renders"); os.makedirs(od, exist_ok=True)
            p = os.path.join(od, "%s_%s.jpg" % (name.replace(" ", "_"), time.strftime("%H%M%S")))
            cv2.imwrite(p, LP.to8(out), [cv2.IMWRITE_JPEG_QUALITY, 94])
            return self._j(200, {"path": p})
        return None
