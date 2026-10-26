"""Термо-сторож StepVisionKit: читач температури SoC, конфіг і файл стану паузи.

ЧОМУ: Depth Anything, YOLO, SAM, MatAnyone ганяються локально на MacBook Air M2
(пасивне охолодження), рятує лише підставка. Коли SoC гарячий,
генерація ПАУЗИТЬСЯ до охолодження і сама продовжується. Задачу НІКОЛИ не вбиваємо.

ЧИТАЧ: чистий ctypes, без pyobjc (у venv ComfyUI його нема). Два джерела, поріг — по
НАЙГАРЯЧІШОМУ з них: IOHIDEventSystem (PMU tdie*, усереднене по кластеру) і AppleSMC
(ключі Tg* = GPU, Tp*/Te* = CPU — ті самі числа, що в меню Stats; issue 11.09.2026:
пороги мусять бути в тій самій шкалі, що бачить людина). Без sudo.
Один клієнт IOHID на процес: кожен новий клієнт — зʼєднання з WindowServer (див. issue 26.09.2026).

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
import struct
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
PROFILE_TITLES = {"stand": "with cooling stand", "bare": "no stand", "custom": "custom"}
TEMP_MIN, TEMP_MAX = 30.0, 100.0        # рішення автора 10.09.2026: вище 100 ніколи
DEFAULTS = {"enabled": True, "profile": "stand", "pause_at": 95.0, "resume_at": 85.0,
            "hard_at": 100.0, "poll_s": 2.0}


# ───────────────────────── конфіг ─────────────────────────

def _num(v, name):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ValueError("%s: not a number" % name)
    if not math.isfinite(f):
        raise ValueError("%s: not a number" % name)
    return f


def validate(raw):
    """raw (dict з UI або файлу) → (cfg, None), або (None, помилка українською).

    Профіль «stand» і «bare» задає пороги сам; «custom» бере пороги з raw.
    Інваріант: resume_at < pause_at ≤ hard_at ≤ 100.
    """
    try:
        if not isinstance(raw, dict):
            raise ValueError("config must be an object")
        enabled = bool(raw.get("enabled", DEFAULTS["enabled"]))
        profile = str(raw.get("profile", DEFAULTS["profile"]))
        if profile not in PROFILE_TITLES:
            raise ValueError("unknown profile: %s" % profile)
        poll_s = _num(raw.get("poll_s", DEFAULTS["poll_s"]), "poll_s")
        if not (0.5 <= poll_s <= 30.0):
            raise ValueError("poll_s must be between 0.5 and 30 s")
        if profile in PROFILES:
            th = dict(PROFILES[profile])
        else:
            th = {k: _num(raw.get(k, DEFAULTS[k]), k) for k in ("pause_at", "resume_at", "hard_at")}
        if not (TEMP_MIN <= th["resume_at"] < th["pause_at"] <= th["hard_at"] <= TEMP_MAX):
            raise ValueError("thresholds: must satisfy resume < pause ≤ hard ≤ 100 °C")
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
        print("[Thermal guard] thermal_config.json is unreadable (%s) — using defaults" % e)
        raw = {}
    cfg, err = validate({**DEFAULTS, **raw} if isinstance(raw, dict) else DEFAULTS)
    if err:
        print("[Thermal guard] config rejected: %s — using defaults" % err)
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
_SMC_OK = (10.0, 130.0)                 # SMC: 0 і -127 — непідключений сенсор, не температура
# Префікс ключа → джерело для позначки. Решта префіксів «Tf»/«Ts» — SoC.
_SMC_PREFIX = (("Tg", "GPU"), ("Tp", "CPU"), ("Te", "CPU"), ("Tf", "SoC"), ("Ts", "SoC"))
# Ключі, які показує Stats (ядра CPU і GPU) — по чипу. Інші T-ключі SMC мають невідомий зміст:
# на M2 Air частина Tp* (Tp0o, Tp0A, Tp0g, Tp02) тримає 96–100 °C, коли ядра Stats ~86 °C,
# і по них сторож заморожував би генерацію без причини. Тому беремо лише ключі Stats.
_STATS_KEYS = {
    "M1": {"CPU": "Tp09 Tp0T Tp01 Tp05 Tp0D Tp0H Tp0L Tp0P Tp0X Tp0b",
           "GPU": "Tg05 Tg0D Tg0L Tg0T"},
    "M2": {"CPU": "Tp1h Tp1t Tp1p Tp1l Tp01 Tp05 Tp09 Tp0D Tp0X Tp0b Tp0f Tp0j",
           "GPU": "Tg0f Tg0j"},
    "M3": {"CPU": "Te05 Te0L Te0P Te0S Tf04 Tf09 Tf0A Tf0B Tf0D Tf0E Tf44 Tf49 Tf4A Tf4B Tf4D Tf4E",
           "GPU": "Tf14 Tf18 Tf19 Tf1A Tf24 Tf28 Tf29 Tf2A"},
    "M4": {"CPU": "Te05 Te09 Te0H Te0S Tp01 Tp05 Tp09 Tp0D Tp0V Tp0Y Tp0b Tp0e",
           "GPU": "Tg0G Tg0H Tg1U Tg1k Tg0K Tg0L Tg0d Tg0e Tg0j Tg0k"},
}


def _stats_keys():
    """{ключ: 'CPU'|'GPU'} для цього чипа або None (невідомий чип — тоді лише Tp*/Tg* зі скану)."""
    try:
        import subprocess
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                               text=True, timeout=2).stdout
    except Exception:
        return None
    for chip, fam in _STATS_KEYS.items():
        if ("Apple " + chip) in brand:
            return {k: src for src, ks in fam.items() for k in ks.split()}
    return None


_smc = None                             # (call, [(name, src, size)]) після успішного скану
_smc_dead = False                       # AppleSMC не відкрився або ключів нема — більше не пробуємо


# Структури AppleSMC з smc.h. Розмір SMCKeyData має бути рівно 80 байт (перевірка в тестах).
class _SMCVersion(ctypes.Structure):
    _fields_ = [("major", ctypes.c_ubyte), ("minor", ctypes.c_ubyte), ("build", ctypes.c_ubyte),
                ("reserved", ctypes.c_ubyte), ("release", ctypes.c_ushort)]


class _SMCPLimitData(ctypes.Structure):
    _fields_ = [("version", ctypes.c_ushort), ("length", ctypes.c_ushort),
                ("cpuPLimit", ctypes.c_uint32), ("gpuPLimit", ctypes.c_uint32),
                ("memPLimit", ctypes.c_uint32)]


class _SMCKeyInfo(ctypes.Structure):
    _fields_ = [("dataSize", ctypes.c_uint32), ("dataType", ctypes.c_uint32),
                ("dataAttributes", ctypes.c_ubyte)]


class _SMCKeyData(ctypes.Structure):
    _fields_ = [("key", ctypes.c_uint32), ("vers", _SMCVersion), ("pLimitData", _SMCPLimitData),
                ("keyInfo", _SMCKeyInfo), ("result", ctypes.c_ubyte), ("status", ctypes.c_ubyte),
                ("data8", ctypes.c_ubyte), ("data32", ctypes.c_uint32),
                ("bytes", ctypes.c_ubyte * 32)]


_SMC_CMD_READ_BYTES = 5                 # data8 у KeyData
_SMC_CMD_READ_INDEX = 8
_SMC_CMD_READ_KEYINFO = 9


def _make_hid_reader():
    """Збирає ctypes-міст до IOKit один раз на процес. Повертає callable -> °C|None."""
    if sys.platform != "darwin":
        raise OSError("not macOS")
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
        raise RuntimeError("IOHIDEventSystemClientCreate failed")
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


def _smc_key(name):
    return int.from_bytes(name.encode("ascii"), "big")


def _smc_name(num):
    return num.to_bytes(4, "big").decode("ascii", errors="replace")


def _smc_open():
    """Відкриває AppleSMC один раз на процес. Повертає call(payload) -> _SMCKeyData | None."""
    if sys.platform != "darwin":
        raise OSError("not macOS")
    iokit = ctypes.CDLL(ctypes.util.find_library("IOKit"))
    libc = ctypes.CDLL(ctypes.util.find_library("c"))
    iokit.IOServiceMatching.restype = ctypes.c_void_p
    iokit.IOServiceMatching.argtypes = [ctypes.c_char_p]
    iokit.IOServiceGetMatchingService.restype = ctypes.c_uint
    iokit.IOServiceGetMatchingService.argtypes = [ctypes.c_uint, ctypes.c_void_p]
    iokit.IOServiceOpen.restype = ctypes.c_int
    iokit.IOServiceOpen.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                                    ctypes.POINTER(ctypes.c_uint)]
    iokit.IOConnectCallStructMethod.restype = ctypes.c_int
    iokit.IOConnectCallStructMethod.argtypes = [ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p,
                                                ctypes.c_size_t, ctypes.c_void_p,
                                                ctypes.POINTER(ctypes.c_size_t)]
    libc.mach_task_self.restype = ctypes.c_uint

    service = iokit.IOServiceGetMatchingService(0, iokit.IOServiceMatching(b"AppleSMC"))
    if not service:
        raise RuntimeError("AppleSMC not found")
    conn = ctypes.c_uint(0)
    if iokit.IOServiceOpen(service, libc.mach_task_self(), 0, ctypes.byref(conn)) != 0:
        raise RuntimeError("IOServiceOpen(AppleSMC) failed")

    def call(payload):
        out = _SMCKeyData()
        size = ctypes.c_size_t(ctypes.sizeof(_SMCKeyData))
        rc = iokit.IOConnectCallStructMethod(conn, 2,            # KERNEL_INDEX_SMC
                                             ctypes.byref(payload), ctypes.sizeof(_SMCKeyData),
                                             ctypes.byref(out), ctypes.byref(size))
        return out if rc == 0 and out.result == 0 else None

    return call


def _smc_keyinfo(call, name):
    """(dataSize, dataType) ключа або None."""
    p = _SMCKeyData()
    p.key, p.data8 = _smc_key(name), _SMC_CMD_READ_KEYINFO
    got = call(p)
    if got is None:
        return None
    return got.keyInfo.dataSize, _smc_name(got.keyInfo.dataType)


def _smc_scan(call):
    """Один раз на процес: перелік ключів Tg/Tp/Te/Tf/Ts з типом flt. [(name, src, size)]."""
    ki = _smc_keyinfo(call, "#KEY")
    if ki is None:
        raise RuntimeError("#KEY unavailable")
    p = _SMCKeyData()
    p.key, p.data8, p.keyInfo.dataSize = _smc_key("#KEY"), _SMC_CMD_READ_BYTES, ki[0]
    got = call(p)
    if got is None:
        raise RuntimeError("#KEY unreadable")
    total = int.from_bytes(bytes(got.bytes[:4]), "big")
    keys = []
    for i in range(total):
        p = _SMCKeyData()
        p.data8, p.data32 = _SMC_CMD_READ_INDEX, i
        got = call(p)
        if got is None:
            continue
        name = _smc_name(got.key)
        src = next((s for pre, s in _SMC_PREFIX if name.startswith(pre)), None)
        if src is None:
            continue
        info = _smc_keyinfo(call, name)
        if info is None or info[1] != "flt " or info[0] != 4:
            continue
        keys.append((name, src, info[0]))
    wl = _stats_keys()
    if wl:
        picked = [(n, wl[n], sz) for n, _s, sz in keys if n in wl]
        if picked:
            return picked
    # невідомий чип або жодного ключа Stats: лише ядра і GPU, без Tf/Ts невідомого змісту
    return [k for k in keys if k[0][:2] in ("Tp", "Te", "Tg")]


def _read_smc():
    """{'GPU'|'CPU'|'SoC': максимум °C} з SMC. None — SMC недоступний (тоді тихо лише IOHID)."""
    global _smc, _smc_dead
    if _smc_dead:
        return None
    if _smc is None:
        try:
            call = _smc_open()
            keys = _smc_scan(call)
        except Exception:
            _smc_dead = True
            return None
        if not keys:
            _smc_dead = True
            return None
        _smc = (call, keys)
    call, keys = _smc
    best = {}
    for name, src, size in keys:
        try:
            p = _SMCKeyData()
            p.key, p.data8, p.keyInfo.dataSize = _smc_key(name), _SMC_CMD_READ_BYTES, size
            got = call(p)
            v = struct.unpack("<f", bytes(got.bytes[:4]))[0] if got is not None else None
        except Exception:
            v = None
        if v is not None and math.isfinite(v) and _SMC_OK[0] <= v <= _SMC_OK[1]:
            best[src] = v if src not in best else max(best[src], v)
    return best


def _combine(hid, smc):
    """Найгарячіший з усіх сенсорів. Позначка джерела — звідки взято максимум."""
    smc = smc or {}
    cands = [(hid, "SoC"), (smc.get("GPU"), "GPU"), (smc.get("CPU"), "CPU"), (smc.get("SoC"), "SoC")]
    cands = [(v, s) for v, s in cands if v is not None]
    if not cands:
        return None
    t, src = max(cands, key=lambda c: c[0])
    return {"t": round(t, 1), "src": src, "hid": hid,
            "smc_gpu": None if smc.get("GPU") is None else round(smc["GPU"], 1),
            "smc_cpu": None if smc.get("CPU") is None else round(smc["CPU"], 1),
            "smc_soc": None if smc.get("SoC") is None else round(smc["SoC"], 1)}


def read_detail():
    """Найгарячіший сенсор з IOHID і SMC, кеш CACHE_S секунд. None — прочитати неможливо.

    {"t": °C, "src": "GPU"|"CPU"|"SoC", "hid": °C|None, "smc_gpu": °C|None,
     "smc_cpu": °C|None, "smc_soc": °C|None}
    """
    global _cache_ts, _cache_val
    with _lock:
        now = time.monotonic()
        if _cache_ts is not None and now - _cache_ts < CACHE_S:
            return _cache_val
        fake = os.environ.get("SVK_THERMAL_FAKE", "")
        if fake:
            v = _read_fake(fake)
            val = None if v is None else {"t": v, "src": "SoC", "hid": v, "smc_gpu": None,
                                          "smc_cpu": None, "smc_soc": None}
        else:
            val = _combine(_read_hid(), _read_smc())
        _cache_ts, _cache_val = now, val
        return val


def read_max_c():
    """Найгарячіший сенсор, °C (IOHID + SMC), кеш CACHE_S секунд. None — прочитати неможливо."""
    d = read_detail()
    return None if d is None else d["t"]
