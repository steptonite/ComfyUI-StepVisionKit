"""Ядро рендеру ноди Depth Look: apply_numpy + хелпери. БЕЗ torch і без сервера Comfy —
імпортується і з ноди (__init__.py), і з воркера (worker.py), і з пакетних скриптів.
Рушій — look.py / look_params.py / look_api.py поруч, у цій теці. Лише numpy + cv2."""
import json
import os
import re
import sys

_PARENT = os.path.dirname(os.path.realpath(__file__))     # рушій лежить у цій же теці
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

import numpy as np  # noqa: E402
import cv2  # noqa: E402
import look  # noqa: E402
import look_params  # noqa: E402
import look_api  # noqa: E402
import depth_motion  # noqa: E402


def _color():
    """media/color.py без пакета media (його __init__ тягне rawpy і ноди Comfy, а воркер - голий скрипт)."""
    global _C
    if _C is None:
        import importlib.util
        spec = importlib.util.spec_from_file_location("svk_color", os.path.join(os.path.dirname(_PARENT), "media", "color.py"))
        _C = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_C)
    return _C


_C = None

DEFAULT_PRESET = "look_v1"
_EXTRA_KEYS = {"vp_x", "vp_y", "shift_x", "shift_y", "palette"}

# фінальний розмір рендеру (довга сторона, px). "original" = розмір входу; max_side - окремо, робоча роздільність
SIZE_ORIGINAL = "original"
OUTPUT_SIZES = [SIZE_ORIGINAL, "3840", "2560", "1920", "1440", "1080", "720"]
PREVIEW_SIDE = 480            # довга сторона превʼю відео-ноди (за замовч.)
PREVIEW_SIDES = ["320", "480", "640", "960"]      # що можна обрати в preview_side; PREVIEW_SIDE має бути серед них


def parse_preview_side(v):
    """'640' / 640 -> довга сторона превʼю в px; усе недопустиме -> PREVIEW_SIDE."""
    try:
        v = int(float(v))
    except (TypeError, ValueError):
        return PREVIEW_SIDE
    return v if str(v) in PREVIEW_SIDES else PREVIEW_SIDE
GRADE_TYPE = "DL_GRADE"       # тип сокета між нодами (звичайний dict)

# значення combo до перекладу (жовтень 2026): старі workflow і API-запити несуть їх далі -> мапимо на нові
LEGACY = {"як оригінал": SIZE_ORIGINAL, "налаштування (швидко, 720)": "tune (fast, 720)", "повний рендер": "full render",
          "превʼю (швидко)": "preview (fast)", "фінал": "final", **look_params.LEGACY_PRESETS}


def legacy(v):
    return look_params.preset_alias(LEGACY.get(v, v)) if isinstance(v, str) else v


def validate_legacy(**fields):
    """Для VALIDATE_INPUTS: name=(значення, дозволені). Старе значення з LEGACY - теж ок."""
    for name, (v, allowed) in fields.items():
        if v is not None and v not in allowed and legacy(v) not in allowed:
            return "%s: %r is not one of %s" % (name, v, ", ".join(map(str, allowed)))
    return True


def parse_palette(s):
    """JSON [[t,[r,g,b]],...] (rgb 0..1) або 't:#hex, t:#hex' -> [[t,[r,g,b]],...]. Порожньо -> None."""
    s = (s or "").strip()
    if not s:
        return None
    if s[0] in "[{":
        pal = json.loads(s)
    else:
        pal = []
        for part in re.split(r"[,;\n]+", s):
            part = part.strip()
            if not part:
                continue
            t, h = part.split(":", 1)
            h = h.strip().lstrip("#")
            if len(h) == 3:
                h = "".join(c * 2 for c in h)
            if len(h) != 6:
                raise ValueError("palette: bad color %r" % part)
            pal.append([float(t), [int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]])
    if len(pal) < 2:
        raise ValueError("palette: at least 2 stops are required")
    return [[float(t), [float(x) for x in c]] for t, c in pal]


def build_preset(preset_name, overrides="{}", palette=""):
    p = look_params.load_preset(preset_name)
    ov = json.loads(overrides) if (overrides or "").strip() else {}
    if not isinstance(ov, dict):
        raise ValueError("overrides must be a JSON object")
    bad = [k for k in ov if k not in look_params.NAMES and k not in _EXTRA_KEYS]
    if bad:
        raise ValueError("overrides: unknown keys %s" % bad)
    p.update(ov)
    pal = parse_palette(palette)
    if pal is not None:
        p["palette"] = pal
    elif isinstance(p.get("palette"), str):
        p["palette"] = parse_palette(p["palette"])
    return p


def _params_from_json(params_json):
    """Повні плоскі параметри з вбудованого тюнера (params_json) поверх дефолтів рушія -> dict пресету."""
    p = json.loads(params_json)
    if not isinstance(p, dict):
        raise ValueError("params_json must be a JSON object")
    base = look_params.defaults()
    base.update(p)
    if isinstance(base.get("palette"), str):
        base["palette"] = parse_palette(base["palette"])
    return base


_DEFAULTS = {}


def _defaults():
    """Дефолти рушія (кеш: inspect.signature щоразу - зайве)."""
    if not _DEFAULTS:
        _DEFAULTS.update(look_params.defaults())
    return dict(_DEFAULTS)


def parse_output_size(v):
    """"original" / '1920' / 1920 -> довга сторона в px (0 = як вхід)."""
    v = legacy(v)
    if v is None or v == SIZE_ORIGINAL:
        return 0
    try:
        return max(0, int(float(v)))
    except (TypeError, ValueError):
        return 0


def static_params(p):
    """Статична нода: анімаційні ключі (look_params.ANIM_KEYS) ігноруються - беруться дефолти рушія (time=0, без руху)."""
    d = _defaults()
    q = dict(p)
    for k in look_params.ANIM_KEYS:
        if k in d:
            q[k] = d[k]
    return q


def make_grade(p, preset="", clean_depth=True, light_is_near=True, depth_levels="clip"):
    """Параметри грейду для виходу `grade` (тип DL_GRADE): плоскі параметри рушія + палітра + preset + прапорці глибини.
    Це звичайний JSON-сумісний dict (копія), тож його можна класти у файл і перечитувати."""
    g = dict(kind="depth_look_grade", version=1, preset=str(preset or ""), params=dict(p), palette=p.get("palette"),
             clean_depth=bool(clean_depth), light_is_near=bool(light_is_near), depth_levels=str(depth_levels))
    return json.loads(json.dumps(g))


def params_from_grade(g):
    """grade -> плоскі параметри рушія поверх дефолтів (палітра - список [[t,[r,g,b]],..])."""
    if not isinstance(g, dict) or not isinstance(g.get("params"), dict):
        raise ValueError("grade must be the output of a Depth Look / Depth Look · Simple node (dict with params)")
    base = _defaults()
    base.update(g["params"])
    if g.get("palette") is not None and "palette" not in g["params"]:
        base["palette"] = g["palette"]
    if isinstance(base.get("palette"), str):
        base["palette"] = parse_palette(base["palette"])
    return json.loads(json.dumps(base))


def resolve_params(preset_name=DEFAULT_PRESET, overrides="{}", palette="", params_json="", grade=None):
    """Один шлях збирання параметрів для всіх нод. Без grade: params_json (повні з тюнера) або preset+overrides+palette.
    З grade: він - база; params_json (повний) лягає зверху, інакше зверху overrides/palette. Пусті правки = grade як є (транзит)."""
    if grade is None:
        if (params_json or "").strip():
            return _params_from_json(params_json)
        return build_preset(preset_name, overrides, palette)
    p = params_from_grade(grade)
    if (params_json or "").strip():
        j = json.loads(params_json)
        if not isinstance(j, dict):
            raise ValueError("params_json must be a JSON object")
        p.update(j)
    else:
        ov = json.loads(overrides) if (overrides or "").strip() else {}
        if not isinstance(ov, dict):
            raise ValueError("overrides must be a JSON object")
        bad = [k for k in ov if k not in look_params.NAMES and k not in _EXTRA_KEYS]
        if bad:
            raise ValueError("overrides: unknown keys %s" % bad)
        p.update(ov)
        pal = parse_palette(palette)
        if pal is not None:
            p["palette"] = pal
    if isinstance(p.get("palette"), str):
        p["palette"] = parse_palette(p["palette"])
    return p


def _out_size(H, W, output_size):
    """(Wo, Ho) фінального кадру: довга сторона = output_size; 0 = як вхід."""
    if output_size and output_size > 0 and max(H, W) != output_size:
        f = output_size / float(max(H, W))
        return max(2, int(round(W * f))), max(2, int(round(H * f)))
    return W, H


def out_shape(B, H, W, output_size=0):
    """Форма результату (B,Ho,Wo,3) для заданого фінального розміру."""
    Wo, Ho = _out_size(H, W, int(output_size or 0))
    return (B, Ho, Wo, 3)


def _work_size(H, W, max_side):
    if max_side and max(H, W) > max_side:
        f = max_side / float(max(H, W))
        return max(2, int(round(W * f))), max(2, int(round(H * f)))
    return W, H


class _Lazy:
    """Послідовність, що рахує елемент на вимогу (look.camera_offsets індексує її, не тримаючи весь батч у памʼяті)."""

    def __init__(self, n, fn):
        self.n, self.fn = n, fn

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.fn(i)


EDGE_MIN = 0.12        # кореляція країв фото↔глибина: своя фотка 0.25, чужа (дзеркальна) 0.07
GB_PER_MP = 0.6        # пік воркера на мегапіксель робочого кадру (заміри 23.10: 3 Мп ≈ 1.2-2.4 ГБ, грубо)


def _edges(x, side=256):
    h, w = x.shape[:2]
    f = side / float(max(h, w))
    x = cv2.resize(x, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    x = cv2.GaussianBlur(x, (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(x, cv2.CV_32F, 1, 0), cv2.Sobel(x, cv2.CV_32F, 0, 1))


def edge_match(rgb, depth):
    """Кореляція модулів градієнтів яскравості кадру і глибини (глибину спершу під пропорції кадру)."""
    g = np.asarray(rgb[..., :3], np.float32).mean(-1)
    d = np.asarray(depth, np.float32)
    d = cv2.resize(d, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_AREA)
    a, b = _edges(g).ravel(), _edges(d).ravel()
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / den) if den > 0 else 0.0


def _ram():
    """(всього, доступно) байт; доступно = None, якщо нема psutil."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.total, vm.available
    except Exception:
        try:
            return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"), None
        except Exception:
            return None, None


def check_inputs(frames_np, depths_np, max_side=0, output_size=0):
    """Запобіжники перед рендером. Явна халепа -> ValueError/MemoryError; підозріле -> список попереджень."""
    warns = []
    B, H, W = frames_np.shape[:3]
    dB = depths_np.shape[0]
    if dB not in (1, B):
        raise ValueError("depth: %d frame(s) but image: %d. Provide 1 depth for everything or one per frame" % (dB, B))
    dh, dw = depths_np.shape[1:3]
    ra, rd = W / float(H), dw / float(dh)
    if abs(rd - ra) / ra > 0.02:
        warns.append("depth aspect %dx%d does not match the image %dx%d - depth will be stretched and the layers will misalign" % (dw, dh, W, H))
    d0 = np.asarray(depths_np[0], np.float32)
    if d0.ndim == 3:
        d0 = d0[..., 0]
    lv = np.unique(d0[::2, ::2]).size
    if float(d0.max() - d0.min()) < 1e-4:
        warns.append("depth is flat (a single level) - the look does not work without the 3D part")
    else:
        if lv <= 256:
            warns.append("depth is 8-bit (%d levels) - the blur will show banding. Connect DA3 directly, or keep depth in float: "
                         "Raw Depth Save (EXR) -> Raw Depth Load -> Depth Receiver -> depth. An 8-bit PNG through Load Image loses levels" % lv)
        for i in sorted({0, B // 2}):
            di = depths_np[i if dB > 1 else 0]
            di = di[..., 0] if di.ndim == 3 else di
            c = edge_match(frames_np[i], di)
            if c < EDGE_MIN:
                warns.append("frame %d: the depth does not seem to belong to this image (edge match %.2f, expected at least %.2f)" % (i, c, EDGE_MIN))
                break
    Wo, Ho = _out_size(H, W, int(output_size or 0))
    Ww, Hw = _work_size(Ho, Wo, int(max_side or 0))
    need = GB_PER_MP * Ww * Hw / 1e6 * 1e9 + B * (H * W + Ho * Wo) * 3 * 4   # пік воркера + вхід/вихід батча
    total, avail = _ram()
    if total and need > 0.9 * total:
        raise MemoryError("needs ≈%.1f GB but the machine only has %.0f GB of memory. Set max_side (e.g. 2048) or use fewer frames"
                          % (need / 1e9, total / 1e9))
    if avail and need > max(avail, 0.5 * total):        # macOS занижує «вільно» (кеш, компресор) - не шуміти на звичайних фото
        warns.append("needs ≈%.1f GB, %.1f GB free - it will swap and be slow. Close heavy apps or lower max_side"
                     % (need / 1e9, avail / 1e9))
    return warns


def watermark(arr, text="PREVIEW 720 - mode: full render for final"):
    """Напис у лівому нижньому куті кожного кадру (B,H,W,3) float, на місці. Лише ASCII (cv2 не вміє кирилицю)."""
    B, H, W = arr.shape[:3]
    sc = max(0.4, W / 1400.0)
    th = max(1, int(round(sc * 2)))
    (tw, tht), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, sc, th)
    x, y = int(12 * sc), H - int(14 * sc)
    y0, y1 = max(0, y - tht - int(8 * sc)), min(H, y + int(8 * sc))
    x0, x1 = max(0, x - int(8 * sc)), min(W, x + tw + int(8 * sc))
    # штамп у uint8 і змішування як альфа: cv2.putText з OpenCV 5 (Vast) приймає лише CV_8U
    stamp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.putText(stamp, text, (x - x0, y - y0), cv2.FONT_HERSHEY_SIMPLEX, sc, 255, th, cv2.LINE_AA)
    a = (stamp.astype(np.float32) / 255.0)[:, :, None]
    for i in range(B):
        roi = arr[i, y0:y1, x0:x1]
        roi *= 0.45 * (1.0 - a)
        roi += a
    return arr


# ───────────── паралельний рендер кадрів ─────────────
class _NoFork(Exception):
    """ОС без fork - рендеримо послідовно."""


def _cpus():
    """Ядра, які реально дасть планувальник: affinity, урізане квотою cgroup (Vast: контейнер бачить 96, квота 23)."""
    try:
        n = len(os.sched_getaffinity(0))
    except Exception:
        n = os.cpu_count() or 1
    q = None
    try:
        a, b = open("/sys/fs/cgroup/cpu.max").read().split()[:2]
        q = None if a == "max" else int(a) / int(b)
    except Exception:
        try:
            a = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            b = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            q = a / b if a > 0 and b > 0 else None
        except Exception:
            pass
    return max(1, min(n, int(q))) if q else n


def _mem_budget():
    """Скільки байт RAM можна віддати воркерам: min(доступно, ліміт cgroup/контейнера); None = невідомо."""
    total, avail = _ram()
    lim = None
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            if v.isdigit() and int(v) < (1 << 50):
                lim = int(v)
                break
        except Exception:
            pass
    cands = [x for x in (avail, lim, total and total * 0.6) if x]
    return min(cands) if cands else None


def pick_workers(B, work_px, requested=None):
    """Кількість процесів: requested -> env SVK_WORKERS -> ядра; ≤ кадрів; ≤ RAM (GB_PER_MP на Мп робочого кадру)."""
    if requested is None:
        requested = _cpus()
        env = os.environ.get("SVK_WORKERS", "").strip()
        if env.isdigit() and int(env) > 0:
            requested = min(requested, int(env))
    n = max(1, min(int(requested), int(B)))
    budget = _mem_budget()
    if n > 1 and budget:
        per = GB_PER_MP * max(work_px, 1) / 1e6 * 1e9
        n = max(1, min(n, int(0.7 * budget / per)))
    return n


_PJOB = {}


def _pinit(fn, out):
    _PJOB["fn"], _PJOB["out"] = fn, out
    try:
        cv2.setNumThreads(1)             # N процесів по 1 потоку замість N x (всі ядра)
    except Exception:
        pass


def _pjob(i):
    res = _PJOB["fn"](i)
    out = _PJOB["out"]
    if isinstance(out, np.memmap):       # спільна memmap-сторінка: пишемо прямо, назад нічого не шлемо
        out[i] = res
        return i, None
    return i, res


def _render_parallel(render_frame, out, B, n_workers, progress):
    import multiprocessing as mp
    if "fork" not in mp.get_all_start_methods():
        raise _NoFork()
    ctx = mp.get_context("fork")         # діти успадковують масиви/mmap без серіалізації
    print("[DepthLook] %d frame(s) in %d parallel process(es)" % (B, n_workers), flush=True)
    done = 0
    # fork при живому пулі потоків cv2 (Linux, pthreads) = діти навічно висять на успадкованому замку
    # (24.10 Vast: 90 процесів по 0% CPU). Гасимо пул у батька до fork, повертаємо після.
    prev = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        with ctx.Pool(n_workers, initializer=_pinit, initargs=(render_frame, out)) as pool:
            for i, res in pool.imap_unordered(_pjob, range(B)):
                if res is not None:
                    out[i] = res
                done += 1
                if progress is not None:
                    progress(done, B)
            if isinstance(out, np.memmap):
                out.flush()
    finally:
        cv2.setNumThreads(prev)
    return out


def _torch_device(device):
    """None -> CPU numpy-шлях (дефолт). Рядок ("cuda"/"cpu"/"mps"/"auto") або env SVK_TORCH (1/cuda/cpu/mps) -> цей пристрій."""
    if device is None or device == "":
        device = os.environ.get("SVK_TORCH", "").strip()
    if isinstance(device, str) and device.lower() in ("", "0", "false", "no", "off", "none"):
        return None
    return device


def _render_torch(device, prep_frame, post_frame, out, B, progress):
    """Кадри йдуть по одному на пристрої (look_torch); CPU-підготовка (глибина, зерно) наперед у потоках: cv2 і numpy RNG
    відпускають GIL, тож GPU не чекає. Порядок і результат не залежать від кількості потоків."""
    import look_torch                                              # імпорт torch лише тут
    from concurrent.futures import ThreadPoolExecutor
    nth = max(1, min(6, _cpus()))
    ahead = nth + 2

    def job(i):
        rgb, d, k = prep_frame(i)
        return rgb, d, k, look_torch.make_noise_for(rgb.shape[0], rgb.shape[1], k)

    with ThreadPoolExecutor(nth) as ex:
        futs = {i: ex.submit(job, i) for i in range(min(B, ahead))}
        for i in range(B):
            rgb, d, k, nz = futs.pop(i).result()
            if i + ahead < B:
                futs[i + ahead] = ex.submit(job, i + ahead)
            res = look_torch.depth_look_lens(rgb, np.ascontiguousarray(d, np.float32), device=device, noise=nz, **k)
            out[i] = post_frame(res, d, k)
            if progress is not None:
                progress(i + 1, B)
            if (i + 1) % 10 == 0:
                print("[DepthLook] %d/%d (torch)" % (i + 1, B))
    return out


def apply_numpy(frames_np, depths_np, preset_name=DEFAULT_PRESET, clean_depth=True,
                light_is_near=True, overrides="{}", palette="", fps=24.0, start_time=0.0,
                params_json="", max_side=0, depth_levels="clip", out=None, progress=None,
                output_size=0, static=False, grade=None, anim=None, frame_step=1, workers=None, motion=None,
                hdr_linear=False, device=None):
    """frames_np: (B,H,W,3) float 0..1; depths_np: (B|1,h,w) або (B|1,h,w,C) float (канал 0).
    Кадр i отримує time = start_time + i/fps (анімаційні повзунки: хвиля, пульс глоу, серпанок, шлейф) і
    seed зерна = seed + i (зерно живе; warp_seed стала). Кадри йдуть по одному, без великих проміжних стеків.
    params_json (непорожній) = повні параметри з тюнера в ноді; інакше preset + overrides + palette.
    grade (dict з виходу ноди, див. make_grade) = готові параметри: preset/overrides/palette/params_json тоді не беруться.
    anim {ключ: значення} лягає поверх ANIM_KEYS (відео-нода). static=True: ANIM_KEYS = дефолти рушія, time = 0.
    output_size > 0: фінальна довга сторона (0 = як вхід). max_side > 0: look рахується на зменшеному кадрі
    (довга сторона = max_side від ФІНАЛЬНОГО розміру) і росте назад до фінального.
    frame_step > 1: кадри вже проріджені викликачем (кожен frame_step-й) - time і seed рахуються по справжньому номеру.
    depth_levels="clip": рівні глибини з усього кліпу (look.clip_levels), "frame": покадровий розтяг.
    workers: скільки процесів рахують кадри паралельно (None = авто: ядра за sched_getaffinity, стеля env SVK_WORKERS і RAM;
    результат побітово той самий, що й послідовний - кожен кадр залежить лише від свого номера).
    motion {dm_*: значення}: «Рух по глибині» (depth_motion.py) - пост-прохід після look у кожному кадрі, на ТОМУ Ж часі
    кадру, що й анімація; None / усі сили 0 = вимкнено (результат побітово той самий, що без параметра).
    hdr_linear=True: кадри scene-linear (1.0 = дифузний білий, світло > 1). Кожен кадр кодується в HLG-сигнал 0..1
    (запас до ~3.8x білого) замість обрізання на 1, лук рахується на сигналі, і ВИХІД теж HLG-сигнал - назад у linear
    його переводить викликач (video_node), після превʼю-стрічки.
    device (етап 1 GPU-порту, 25.10.2026): None / "" = звичайний CPU-шлях (numpy+cv2, байт-в-байт як раніше);
    "cuda" / "cpu" / "mps" / "auto" = look.depth_look_lens рахується в look_torch.py на пристрої (опційно; паритет <= 1/255).
    Те саме вмикає змінна середовища SVK_TORCH=1 (або =cuda/cpu/mps), якщо device не задано. Глибина, motion, ресайз - як було (CPU).
    out: готовий (B,Ho,Wo,3) float32 масив для результату (напр. numpy-вʼю torch-тензора). -> out."""
    if not isinstance(frames_np, np.ndarray):
        frames_np = np.asarray(frames_np)
    if not isinstance(depths_np, np.ndarray):
        depths_np = np.asarray(depths_np)
    B, H, W = frames_np.shape[:3]
    dB = depths_np.shape[0]
    if dB not in (1, B):
        raise ValueError("depth: %d frames but image: %d" % (dB, B))
    if grade is not None:
        p = params_from_grade(grade)
    else:
        p = resolve_params(preset_name, overrides, palette, params_json)
    if static:
        p = static_params(p)
    if anim:
        p.update({k: v for k, v in anim.items() if k in look_params.ANIM_KEYS})
    kw = look_params.kwargs(p)
    base_seed = int(kw.get("seed", 7))
    step = max(1, int(frame_step or 1))
    Wo, Ho = _out_size(H, W, int(output_size or 0))
    if out is None:
        out = np.empty((B, Ho, Wo, 3), np.float32)
    Ww, Hw = _work_size(Ho, Wo, int(max_side or 0))

    def get_rgb(i):
        rgb = np.asarray(frames_np[i][..., :3], np.float32)
        if not hdr_linear:
            rgb = np.clip(rgb, 0, 1)
        if (Ww, Hw) != (W, H):
            if Ww * Hw < W * H:
                rgb = cv2.resize(rgb, (Ww, Hw), interpolation=cv2.INTER_AREA)
            else:                                                  # робоча роздільність більша за вхід: апскейл
                rgb = cv2.resize(rgb, (Ww, Hw), interpolation=cv2.INTER_CUBIC)
                if not hdr_linear:
                    rgb = np.clip(rgb, 0, 1)
        if hdr_linear:                                             # світло над 1 - у верхню чверть сигналу, не в стелю
            rgb = _color().hlg_encode(rgb)
        return rgb

    def get_depth(i):
        d = depths_np[i if dB > 1 else 0]
        if d.ndim == 3:
            d = d[..., 0]
        return np.asarray(d, np.float32)          # float до кінця, без uint8

    levels = None
    if clean_depth and depth_levels == "clip":
        idx = sorted(set(int(round(v)) for v in np.linspace(0, B - 1, min(8, B))))
        preps = [look.prepare_depth(get_depth(i), get_rgb(i), light_is_near=light_is_near, stretch=False) for i in idx]
        levels = look.clip_levels(preps)
        del preps
    offs = None
    if int(round(float(kw.get("grad_mode", 0)))) == 1 and B > 1:
        print("STAGE camera", flush=True)                 # воркер -> рядок стану ноди «аналіз руху камери»
        # режим «камера»: рух фону між кадрами -> grad_offset_x/y (раз на батч; глибина для маски далекого — на 720)
        def small_depth(i):
            rgb = get_rgb(i)
            f = min(1.0, 720.0 / max(rgb.shape[:2]))
            if f < 1.0:
                rgb = cv2.resize(rgb, (max(8, int(round(rgb.shape[1] * f))), max(8, int(round(rgb.shape[0] * f)))),
                                 interpolation=cv2.INTER_AREA)
            return look_api.prepare_frame_depth(rgb, get_depth(i), clean_depth, light_is_near, levels)
        offs = look.camera_offsets(_Lazy(B, get_rgb), _Lazy(B, small_depth), normalize=True)
    base_ox, base_oy = float(kw.get("grad_offset_x", 0.0)), float(kw.get("grad_offset_y", 0.0))

    motion_on = bool(motion) and not static and depth_motion.active(motion)

    def prep_frame(i):
        """Кадр i -> (rgb, d, k): вхід look. Залежить ЛИШЕ від i (time, seed, офсети камери) і від спільних levels/offs."""
        rgb = get_rgb(i)
        d = look_api.prepare_frame_depth(rgb, get_depth(i), clean_depth, light_is_near, levels)
        si = i * step                                              # справжній номер кадру в кліпі
        k = dict(kw)
        k["time"] = 0.0 if static else float(start_time) + si / max(float(fps), 1e-6)
        k["seed"] = base_seed + si
        if offs is not None:
            k["grad_offset_x"] = base_ox + float(offs[i][0])
            k["grad_offset_y"] = base_oy + float(offs[i][1])
        return rgb, d, k

    def post_frame(res, d, k):
        if (Ww, Hw) != (Wo, Ho):
            res = np.clip(cv2.resize(res, (Wo, Ho), interpolation=cv2.INTER_CUBIC), 0, 1)
        if motion_on:                                              # рух по глибині: глибина - під розмір ФІНАЛЬНОГО кадру
            dm_d = d if d.shape == (Ho, Wo) else cv2.resize(d, (Wo, Ho), interpolation=cv2.INTER_CUBIC)
            res = depth_motion.apply(res, np.clip(dm_d, 0, 1), k["time"], motion)
        return res

    def render_frame(i):
        """Кадр i -> (Ho,Wo,3) float32. Залежить ЛИШЕ від i (time, seed, офсети камери) і від спільних levels/offs - тож
        порядок і кількість процесів не змінюють результат."""
        rgb, d, k = prep_frame(i)
        res = look.depth_look_lens(rgb, np.ascontiguousarray(d, np.float32), **k)
        return post_frame(res, d, k)

    tdev = _torch_device(device)
    if tdev is not None:
        try:
            return _render_torch(tdev, prep_frame, post_frame, out, B, progress)
        except ImportError as e:
            print("[DepthLook] torch-шлях недоступний (%s) - рахую на CPU" % e)

    n_workers = pick_workers(B, Ww * Hw, workers)
    if n_workers > 1:
        try:
            return _render_parallel(render_frame, out, B, n_workers, progress)
        except _NoFork:
            pass
    for i in range(B):
        out[i] = render_frame(i)
        if progress is not None:
            progress(i + 1, B)
        if (i + 1) % 10 == 0:
            print("[DepthLook] %d/%d" % (i + 1, B))
    return out
