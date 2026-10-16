"""Термо-сторож StepVisionKit: читач температури SoC, конфіг і файл стану паузи.

ЧОМУ: Льоша ганяє локально Depth Anything, YOLO, SAM, MatAnyone на MacBook Air M2
(пасивне охолодження). Рятує лише підставка, Катя генерує без неї. Коли SoC гарячий,
генерація ПАУЗИТЬСЯ до охолодження і сама продовжується. Задачу НІКОЛИ не вбиваємо.

ЧИТАЧ: чистий ctypes, без pyobjc (у venv ComfyUI його нема). Логіка та сама, що в
~/.claude/tools/thermal.py: IOHIDEventSystem, сенсори температури SoC, без sudo.
Один клієнт на процес: кожен новий клієнт — зʼєднання з WindowServer (див. issue 26.09.2026).

ПОРТАТИВНІСТЬ: не macOS, не Apple Silicon, помилка ctypes → read_max_c() повертає None.
Сторож тоді тихо вимкнений, ComfyUI працює як завжди.

ТЕСТ: SVK_THERMAL_FAKE=<шлях до файлу з числом °C> — температура береться з файлу,
нагрівати Мак не треба.
"""

import ctypes
import ctypes.util
import json
import math
import os
import sys
import threading
import time

LEX_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(LEX_DIR)
CONFIG_PATH = os.path.join(REPO_DIR, "thermal_config.json")
STATE_PATH = os.path.join(REPO_DIR, ".thermal_state.json")

PROFILES = {
    "stand": {"pause_at": 95.0, "resume_at": 85.0, "hard_at": 100.0},   # з підставкою
    "bare": {"pause_at": 90.0, "resume_at": 80.0, "hard_at": 97.0},     # без підставки
}
PROFILE_TITLES = {"stand": "з підставкою", "bare": "без підставки", "custom": "свій"}
TEMP_MIN, TEMP_MAX = 30.0, 100.0        # рішення Льоші 10.09.2026: вище 100 ніколи
DEFAULTS = {"enabled": True, "profile": "stand", "pause_at": 95.0, "resume_at": 85.0,
            "hard_at": 100.0, "poll_s": 2.0}


# ───────────────────────── конфіг ─────────────────────────

def _num(v, name):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError("%s: не число" % name)
    if not math.isfinite(f):
        raise ValueError("%s: не число" % name)
    return f


def validate(raw):
    """raw (dict з UI або файлу) → (cfg, None), або (None, помилка українською).

    Профіль «stand» і «bare» задає пороги сам; «custom» бере пороги з raw.
    Інваріант: resume_at < pause_at ≤ hard_at ≤ 100.
    """
    try:
        if not isinstance(raw, dict):
            raise ValueError("конфіг має бути обʼєктом")
        enabled = bool(raw.get("enabled", DEFAULTS["enabled"]))
        profile = str(raw.get("profile", DEFAULTS["profile"]))
        if profile not in PROFILE_TITLES:
            raise ValueError("невідомий профіль: %s" % profile)
        poll_s = _num(raw.get("poll_s", DEFAULTS["poll_s"]), "poll_s")
        if not (0.5 <= poll_s <= 30.0):
            raise ValueError("poll_s має бути від 0.5 до 30 с")
        if profile in PROFILES:
            th = dict(PROFILES[profile])
        else:
            th = {k: _num(raw.get(k, DEFAULTS[k]), k) for k in ("pause_at", "resume_at", "hard_at")}
        if not (TEMP_MIN <= th["resume_at"] < th["pause_at"] <= th["hard_at"] <= TEMP_MAX):
            raise ValueError("пороги: має бути продовження < пауза ≤ жорстка ≤ 100 °C")
    except ValueError as e:
        return None, str(e)
    cfg = {"enabled": enabled, "profile": profile, "poll_s": round(poll_s, 2)}
    cfg.update({k: round(v, 1) for k, v in th.items()})
    return cfg, None


def load_config(path=None):
    """Конфіг з файлу. Немає файлу, биті дані або хибні пороги → типові (з попередженням у лог)."""
    path = path or CONFIG_PATH
    raw = {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {}
    except Exception as e:
        print("[Термо-сторож] thermal_config.json не читається (%s) — беру типові" % e)
        raw = {}
    cfg, err = validate({**DEFAULTS, **raw} if isinstance(raw, dict) else DEFAULTS)
    if err:
        print("[Термо-сторож] конфіг відхилено: %s — беру типові" % err)
        cfg, _ = validate(DEFAULTS)
    return cfg


def save_config(cfg, path=None):
    """Валідує і пише атомарно. Повертає нормалізований cfg або піднімає ValueError."""
    norm, err = validate(cfg)
    if err:
        raise ValueError(err)
    path = path or CONFIG_PATH
    tmp = "%s.%d.tmp" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(norm, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)
    return norm


# ───────────────────────── файл стану (пише сторож, читає ComfyUI) ─────────────────────────

def read_state(path=None):
    try:
        with open(path or STATE_PATH, encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else None
    except Exception:
        return None


def write_state(data, path=None):
    path = path or STATE_PATH
    try:
        tmp = "%s.%d.tmp" % (path, os.getpid())
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        pass


# ───────────────────────── читач температури SoC ─────────────────────────

CACHE_S = 1.0
_TEMP_OK = (0.0, 150.0)                 # сенсори поза цим діапазоном — сміття
_lock = threading.Lock()
_cache_ts = None
_cache_val = None
_hid_read = None                        # callable -> float|None, або None
_hid_dead = False                       # читач не зібрався — більше не пробуємо


def _make_hid_reader():
    """Збирає ctypes-міст до IOKit один раз на процес. Повертає callable -> °C|None."""
    if sys.platform != "darwin":
        raise OSError("не macOS")
    iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
    cf = ctypes.CDLL(ctypes.util.find_library("CoreFoundation"))

    kCFStringEncodingUTF8 = 0x08000100
    kCFNumberSInt32Type = 3
    kIOHIDEventTypeTemperature = 15
    kHIDPage_AppleVendor = 0xFF00
    kHIDUsage_AppleVendor_TemperatureSensor = 0x0005

    cf.CFDictionaryCreateMutable.restype = ctypes.c_void_p
    cf.CFDictionarySetValue.argtypes = [ctypes.c_void_p] * 3
    cf.CFStringCreateWithCString.restype = ctypes.c_void_p
    cf.CFStringCreateWithCString.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_uint32]
    cf.CFNumberCreate.restype = ctypes.c_void_p
    cf.CFNumberCreate.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
    cf.CFArrayGetCount.restype = ctypes.c_long
    cf.CFArrayGetCount.argtypes = [ctypes.c_void_p]
    cf.CFArrayGetValueAtIndex.restype = ctypes.c_void_p
    cf.CFArrayGetValueAtIndex.argtypes = [ctypes.c_void_p, ctypes.c_long]
    cf.CFRelease.argtypes = [ctypes.c_void_p]

    iokit.IOHIDEventSystemClientCreate.restype = ctypes.c_void_p
    iokit.IOHIDEventSystemClientCreate.argtypes = [ctypes.c_void_p]
    iokit.IOHIDEventSystemClientSetMatching.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    iokit.IOHIDEventSystemClientCopyServices.restype = ctypes.c_void_p
    iokit.IOHIDEventSystemClientCopyServices.argtypes = [ctypes.c_void_p]
    iokit.IOHIDServiceClientCopyEvent.restype = ctypes.c_void_p
    iokit.IOHIDServiceClientCopyEvent.argtypes = [ctypes.c_void_p, ctypes.c_int64,
                                                  ctypes.c_int32, ctypes.c_int64]
    iokit.IOHIDEventGetFloatValue.restype = ctypes.c_double
    iokit.IOHIDEventGetFloatValue.argtypes = [ctypes.c_void_p, ctypes.c_int32]

    def cfstr(s):
        return cf.CFStringCreateWithCString(None, s.encode("utf-8"), kCFStringEncodingUTF8)

    def cfnum(n):
        v = ctypes.c_int32(n)
        return cf.CFNumberCreate(None, kCFNumberSInt32Type, ctypes.byref(v))

    key_cb = ctypes.byref(ctypes.c_void_p.in_dll(cf, "kCFTypeDictionaryKeyCallBacks"))
    val_cb = ctypes.byref(ctypes.c_void_p.in_dll(cf, "kCFTypeDictionaryValueCallBacks"))
    match = cf.CFDictionaryCreateMutable(None, 0, key_cb, val_cb)
    cf.CFDictionarySetValue(match, cfstr("PrimaryUsagePage"), cfnum(kHIDPage_AppleVendor))
    cf.CFDictionarySetValue(match, cfstr("PrimaryUsage"),
                            cfnum(kHIDUsage_AppleVendor_TemperatureSensor))
    client = iokit.IOHIDEventSystemClientCreate(None)
    if not client:
        raise RuntimeError("IOHIDEventSystemClientCreate не спрацював")
    iokit.IOHIDEventSystemClientSetMatching(client, match)

    def read():
        services = iokit.IOHIDEventSystemClientCopyServices(client)
        if not services:
            return None
        vals = []
        try:
            for i in range(cf.CFArrayGetCount(services)):
                sc = cf.CFArrayGetValueAtIndex(services, i)
                ev = iokit.IOHIDServiceClientCopyEvent(sc, kIOHIDEventTypeTemperature, 0, 0)
                if not ev:
                    continue
                try:
                    vals.append(iokit.IOHIDEventGetFloatValue(
                        ev, kIOHIDEventTypeTemperature << 16))
                finally:
                    cf.CFRelease(ev)
        finally:
            cf.CFRelease(services)
        good = [v for v in vals if math.isfinite(v) and _TEMP_OK[0] < v <= _TEMP_OK[1]]
        return round(max(good), 1) if good else None

    return read


def _read_hid():
    global _hid_read, _hid_dead
    if _hid_dead:
        return None
    if _hid_read is None:
        try:
            _hid_read = _make_hid_reader()
        except Exception:
            _hid_dead = True
            return None
    try:
        return _hid_read()
    except Exception:
        return None


def _read_fake(path):
    try:
        with open(path, encoding="utf-8") as f:
            v = float(f.read().strip())
    except Exception:
        return None
    return v if math.isfinite(v) and _TEMP_OK[0] < v <= _TEMP_OK[1] else None


def read_max_c():
    """Найгарячіший сенсор SoC, °C, кеш CACHE_S секунд. None — прочитати неможливо."""
    global _cache_ts, _cache_val
    with _lock:
        now = time.monotonic()
        if _cache_ts is not None and now - _cache_ts < CACHE_S:
            return _cache_val
        fake = os.environ.get("SVK_THERMAL_FAKE", "")
        val = _read_fake(fake) if fake else _read_hid()
        _cache_ts, _cache_val = now, val
        return val
