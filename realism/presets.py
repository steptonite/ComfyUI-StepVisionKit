"""Власні пресети Camera Realism: знімок усіх повзунків (+ сила) у json. Тека - user-тека Comfy (переживає оновлення паку)."""
import json
import math
import os

from . import params as P

NAME_MAX = 48


def _dir():
    try:
        import folder_paths  # type: ignore
        return os.path.join(folder_paths.get_user_directory(), "svk_realism_presets")
    except Exception:   # noqa: BLE001 - поза Comfy (мікротест) - тека поруч із модулем
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "user_presets")


def clean_name(name):
    """Літери/цифри/пробіл/_-. , без шляху; не збігається з вбудованим профілем. Кидає ValueError - мовчки не «чистить»."""
    if not isinstance(name, str):
        raise ValueError("invalid preset name")
    n = " ".join(name.split())
    if not n or len(n) > NAME_MAX or n.startswith(".") or ".." in n or not all(ch.isalnum() or ch in " _-." for ch in n):
        raise ValueError("invalid preset name %r (letters, digits, space _ - . only)" % (name,))
    if n in P.PROFILE_NAMES:
        raise ValueError("%s is a built-in profile, pick another name" % n)
    return n


def sanitize(values):
    """Лише відомі ключі, скінченні числа, в межах SPEC; frame_offset не зберігається."""
    out = {}
    lim = {s[0]: (s[1], s[2], len(s) > 7 and s[7] == "i") for s in P.SPEC}
    lim["strength"] = (0.0, 1.0, False)
    for k, v in (values or {}).items():
        if k not in lim or k in P.RUNTIME or isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            continue
        lo, hi, is_int = lim[k]
        v = min(max(v, lo), hi)
        out[k] = int(round(v)) if is_int else float(v)
    return out


def _path(n):
    return os.path.join(_dir(), n + ".json")


def list_presets():
    """{імʼя: значення} за алфавітом; зіпсований файл пропускається."""
    d, out = _dir(), {}
    if os.path.isdir(d):
        for f in sorted(os.listdir(d), key=str.lower):
            if not f.endswith(".json"):
                continue
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fh:
                    out[f[:-5]] = sanitize(json.load(fh))
            except Exception as e:   # noqa: BLE001
                print("[CameraRealism] preset %s skipped: %s" % (f, e))
    return out


def save_preset(name, values):
    n = clean_name(name)
    vals = sanitize(values)
    if not vals:
        raise ValueError("nothing to save")
    os.makedirs(_dir(), exist_ok=True)
    p = _path(n)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(vals, f, indent=1, ensure_ascii=False)
    os.replace(p + ".tmp", p)
    return n


def delete_preset(name):
    n = clean_name(name)
    p = _path(n)
    if not os.path.isfile(p):
        raise ValueError("preset %s not found" % n)
    os.remove(p)
    return n
