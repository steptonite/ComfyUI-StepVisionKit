"""Depth Look · відео: іменовані пресети руху й анімації (Save as… / завантажити / видалити).
Один файл на пресет у depth_look/video_presets/<name>.json = {ім'я віджета: значення}. Набір ключів: рух по глибині (dm_*),
анімація (look_params.ANIM_KEYS без `time`), preview_every, hdr. НЕ входять: fps, start_time, розмір, якість, службові поля.
Вбудовані пресети (BUILTIN) живуть у коді: їх не можна ні перезаписати, ні видалити, ні підмінити файлом на диску.
Лише look_params + depth_motion (numpy) - без torch/comfy, тож модуль тестується окремо від ноди."""
import json
import math
import os

import look_params
import depth_motion

CODE_DIR = os.path.dirname(os.path.realpath(__file__))
PRESETS_DIR = os.path.join(CODE_DIR, "video_presets")
HDR_OFF = "SDR or HLG/PQ as encoded"        # = video_node.HDR_OFF (там імпорт тягне torch, тож тут дубль)
PREVIEW_EVERY = 3
NAME_MAX = 60


def keys():
    """Впорядкований список ключів, що входять у пресет."""
    anim = [k for k in look_params.ANIM_KEYS if k != "time"]
    return anim + list(depth_motion.KEYS) + ["preview_every", "hdr"]


def defaults():
    d = look_params.defaults()
    out = {}
    for k in keys():
        if k in depth_motion.DEFAULTS:
            out[k] = depth_motion.DEFAULTS[k]
        elif k == "preview_every":
            out[k] = PREVIEW_EVERY
        elif k == "hdr":
            out[k] = HDR_OFF
        else:
            out[k] = d.get(k, 0)
    return out


def _builtin():
    """Вбудовані пресети: {name: {ключ: значення}} - лише відмінності від дефолтів, решта підставляється defaults()."""
    return {
        "00_static": {},                          # усе вимкнено = дефолти ноди
        "gentle_parallax": {"dm_par": 0.3},       # лише повільна 2.5D-камера
    }


LOCKED = frozenset(_builtin())


def clean_name(name):
    """Перевіряє імʼя: літери/цифри/_-. (пробіл -> _), без шляху, без крапки на початку. Кидає ValueError - нічого не «чистить мовчки»."""
    if not isinstance(name, str):
        raise ValueError("invalid preset name")
    n = name.strip().replace(" ", "_")
    if not n or len(n) > NAME_MAX or n.startswith(".") or ".." in n or not all(ch.isalnum() or ch in "_-." for ch in n):
        raise ValueError("invalid preset name %r (letters, digits, _ - . only; no paths)" % (name,))
    return n


def is_locked(name):
    return isinstance(name, str) and name.lower() in {x.lower() for x in LOCKED}


def _path(name):
    p = os.path.join(PRESETS_DIR, name + ".json")
    if os.path.dirname(os.path.abspath(p)) != os.path.abspath(PRESETS_DIR):
        raise ValueError("invalid preset name %r" % (name,))
    return p


def sanitize(values):
    """Лишає лише відомі ключі й приводить тип до типу дефолту (bool/int/float/str); NaN/inf і сміття відкидає."""
    if not isinstance(values, dict):
        raise ValueError("values must be an object")
    base = defaults()
    out = {}
    for k, v in values.items():
        if k not in base:
            continue
        d = base[k]
        if isinstance(d, bool):
            out[k] = bool(v)
        elif isinstance(d, int):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            out[k] = int(round(v))
        elif isinstance(d, float):
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
                continue
            out[k] = float(v)
        else:
            if isinstance(v, str) and len(v) <= 64:
                out[k] = v
    return out


def list_presets():
    """[{"name", "locked"}]: спершу вбудовані, потім користувацькі за алфавітом."""
    rows = [{"name": n, "locked": True} for n in sorted(LOCKED)]
    if os.path.isdir(PRESETS_DIR):
        for f in sorted(os.listdir(PRESETS_DIR)):
            n = f[:-5]
            if f.endswith(".json") and n and not is_locked(n):
                rows.append({"name": n, "locked": False})
    return rows


def load_preset(name):
    n = clean_name(name)
    if n in LOCKED:
        v = defaults()
        v.update(sanitize(_builtin()[n]))
        return v
    p = _path(n)
    if not os.path.isfile(p):
        raise ValueError("preset %s not found" % n)
    with open(p, encoding="utf-8") as f:
        v = defaults()
        v.update(sanitize(json.load(f)))
    return v


def save_preset(name, values):
    n = clean_name(name)
    if is_locked(n):
        raise ValueError("preset %s is built-in, save it under a different name" % n)
    vals = sanitize(values)
    if not vals:
        raise ValueError("nothing to save")
    os.makedirs(PRESETS_DIR, exist_ok=True)
    p = _path(n)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=1, ensure_ascii=False)
    os.replace(tmp, p)
    return n


def delete_preset(name):
    if not isinstance(name, str) or clean_name(name) != name:
        raise ValueError("invalid preset name %r" % (name,))
    if is_locked(name):
        raise ValueError("preset %s is built-in, it cannot be deleted" % name)
    p = _path(name)
    if not os.path.isfile(p):
        raise ValueError("preset %s not found" % name)
    os.remove(p)
    return name
