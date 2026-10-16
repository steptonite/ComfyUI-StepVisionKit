"""М'яка пауза термо-сторожа всередині процесу ComfyUI + маршрути для UI.

Як це працює:
  · обгортаємо comfy.utils.ProgressBar.update_absolute і
    comfy.model_management.throw_exception_if_processing_interrupted — перед кожним кроком
    семплера/прогрес-бару дивимось на SoC;
  · якщо SoC ≥ pause_at — блокуємо потік (спимо) до resume_at. Пауза, не обрив: якщо Льоша
    натиснув «скасувати», пауза завершується, і оригінал кидає InterruptProcessingException;
  · жорсткий поріг (hard_at) і ноди без прогрес-бару тримає thermal_watch: окремий процес,
    що ставить ComfyUI на SIGSTOP. install() запускає і його.

Стан для UI: повідомлення svk.thermal через PromptServer (cooling / ok / was_frozen),
GET /svk/thermal/status, POST /svk/thermal/config (валідація, запис thermal_config.json).

install() викликається один раз з __init__.py. Усе обгорнуто в try: пакет не падає через сторожа.
Вимкнений гейт (enabled=false) → нічого не патчиться і не запускається; маршрути лишаються,
щоб UI міг увімкнути гейт.
"""

import os
import sys
import threading
import time

from . import thermal
from . import thermal_watch

COOL_STEP_S = 2.0           # крок перевірки під час паузи, с
_KEYS = ("enabled", "profile", "pause_at", "resume_at", "hard_at", "poll_s")

_cfg = thermal.validate(thermal.DEFAULTS)[0]
_lock = threading.Lock()
_active = False             # обгортки стоять і сторож запущено
_installed = False
_last_check = None          # monotonic: останній замір для тротла poll_s
_pausing = 0                # скільки потоків зараз у паузі


def _log(msg):
    print("[Термо-сторож] " + msg, file=sys.stderr, flush=True)


def _send(state, **data):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("svk.thermal", dict(state=state, **data))
    except Exception:
        pass


# ───────────────────────── м'яка пауза ─────────────────────────

def maybe_cool():
    """Викликається перед кожним кроком. Тротл poll_s; при гарячому SoC блокує потік до охолодження."""
    global _last_check
    cfg = _cfg
    if not _active or not cfg["enabled"]:
        return
    now = time.monotonic()
    with _lock:
        if _last_check is not None and now - _last_check < cfg["poll_s"]:
            return
        _last_check = now
    t = thermal.read_max_c()
    if t is None or t < cfg["pause_at"]:
        return
    _pause(t, cfg)


def _pause(t, cfg):
    global _pausing
    import comfy.model_management as mm

    resume_at = cfg["resume_at"]
    start = time.monotonic()
    _log("%.0f°C ≥ %.0f — пауза до %.0f°C" % (t, cfg["pause_at"], resume_at))
    interrupted = False
    last_t = now_t = t
    with _lock:
        _pausing += 1
    try:
        while True:
            _send("cooling", t=last_t, resume_at=resume_at,
                  since=round(time.monotonic() - start, 1))
            end = time.monotonic() + COOL_STEP_S
            while time.monotonic() < end:
                if getattr(mm, "interrupt_processing", False):
                    interrupted = True
                    break
                time.sleep(0.2)
            if interrupted:
                break
            now_t = thermal.read_max_c()
            if now_t is None:           # датчик зник посеред паузи — не блокуємо назавжди
                break
            last_t = now_t
            if now_t <= resume_at:
                break
    finally:
        with _lock:
            _pausing -= 1
    paused_s = round(time.monotonic() - start, 1)
    if interrupted:
        _log("пауза перервана скасуванням після %.0f с" % paused_s)
        _send("ok", t=last_t, paused_s=paused_s, interrupted=True)
    elif now_t is None:
        _log("температуру не прочитати під час паузи — продовжую (пауза %.0f с)" % paused_s)
        _send("ok", t=last_t, paused_s=paused_s)
    else:
        _log("%.0f°C — продовжую (пауза %.0f с)" % (last_t, paused_s))
        _send("ok", t=last_t, paused_s=paused_s)


def _patch():
    """Обгортки з guard від подвійного патчу. Оригінали лишаються живими всередині."""
    import comfy.utils as cu
    import comfy.model_management as mm

    PB = cu.ProgressBar
    orig_upd = PB.update_absolute
    if not getattr(orig_upd, "_svk_thermal", False):
        def update_absolute(self, *args, **kwargs):
            maybe_cool()
            return orig_upd(self, *args, **kwargs)
        update_absolute._svk_thermal = True
        PB.update_absolute = update_absolute

    orig_thr = mm.throw_exception_if_processing_interrupted
    if not getattr(orig_thr, "_svk_thermal", False):
        def throw_exception_if_processing_interrupted(*args, **kwargs):
            maybe_cool()
            return orig_thr(*args, **kwargs)
        throw_exception_if_processing_interrupted._svk_thermal = True
        mm.throw_exception_if_processing_interrupted = throw_exception_if_processing_interrupted


# ───────────────────────── фоновий звіт для UI ─────────────────────────

def _reporter_loop(seen):
    """Раз на 2 с читає файл стану сторожа і повідомляє UI про заморозку, що вже минула."""
    while True:
        time.sleep(2.0)
        try:
            st = thermal.read_state()
            if not st or st.get("stopped"):
                continue
            seq = int(st.get("seq", 0) or 0)
            if seq > seen and st.get("last_stop_s") is not None:
                seen = seq
                _send("was_frozen", seconds=float(st["last_stop_s"]), t=st.get("t"))
        except Exception:
            pass


def _watch_alive():
    p = thermal_watch._proc
    return p is not None and p.poll() is None


# ───────────────────────── активація ─────────────────────────

def _activate():
    """Ставить обгортки і сторожа. Повертає False, якщо датчик недоступний (тоді тихо вимкнено)."""
    global _active
    with _lock:
        if _active:
            return True
        if thermal.read_max_c() is None:
            _log("температуру не прочитати — вимкнено")
            return False
        _patch()
        _active = True
    prev = thermal.read_state() or {}
    seen = int(prev.get("seq", 0) or 0)
    thermal_watch.start(thermal.CONFIG_PATH, thermal.STATE_PATH)
    threading.Thread(target=_reporter_loop, args=(seen,), daemon=True,
                     name="svk-thermal-reporter").start()
    return True


def _apply(cfg):
    """Новий конфіг після POST: оновлюємо пам'ять; якщо гейт увімкнули вперше — активуємо."""
    global _cfg
    _cfg = cfg
    if cfg["enabled"] and not _active:
        _activate()


def status():
    cfg = _cfg
    t = thermal.read_max_c()
    st = thermal.read_state() or {}
    frozen = bool(st.get("stopped")) and _watch_alive()
    if not cfg["enabled"]:
        state = "off"
    elif _pausing > 0:
        state = "cooling"
    elif frozen:
        state = "frozen"
    else:
        state = "ok"
    return {"enabled": cfg["enabled"], "t": t, "state": state,
            "pause_at": cfg["pause_at"], "resume_at": cfg["resume_at"],
            "hard_at": cfg["hard_at"], "profile": cfg["profile"],
            "profile_title": thermal.PROFILE_TITLES.get(cfg["profile"], cfg["profile"]),
            "poll_s": cfg["poll_s"], "reader_ok": t is not None,
            "active": _active, "watch": _watch_alive()}


def _register_routes():
    from server import PromptServer
    from aiohttp import web as _web

    @PromptServer.instance.routes.get("/svk/thermal/status")
    async def _svk_thermal_status(request):
        return _web.json_response(status())

    @PromptServer.instance.routes.post("/svk/thermal/config")
    async def _svk_thermal_config(request):
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise ValueError("тіло запиту має бути обʼєктом")
        except Exception as e:
            return _web.json_response({"ok": False, "message": str(e)}, status=400)
        merged = dict(_cfg)
        for k in _KEYS:
            if k in body:
                merged[k] = body[k]
        try:
            cfg = thermal.save_config(merged)
        except ValueError as e:
            return _web.json_response({"ok": False, "message": str(e)}, status=400)
        _apply(cfg)
        return _web.json_response({"ok": True, **status()})


def install():
    """Викликається один раз з __init__.py. Ніколи не піднімає виняток назовні."""
    global _cfg, _installed
    if _installed:
        return
    _installed = True
    try:
        _cfg = thermal.load_config()
    except Exception as e:
        _log("конфіг не завантажено: %s" % e)
        return
    try:
        _register_routes()
    except Exception as e:
        _log("маршрути статусу не зареєстровано: %s" % e)
    if not _cfg["enabled"]:
        return
    try:
        _activate()
    except Exception as e:
        _log("не запущено: %s" % e)
