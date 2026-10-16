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

import logging
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
HEARTBEAT_S = 60.0          # рядок у консоль під час задачі: видно, що сторож живий

# Лічильники: за весь запуск ComfyUI і за поточну задачу (черга виконує по одній).
_total = {"jobs": 0, "pauses": 0, "paused_s": 0.0, "freezes": 0, "frozen_s": 0.0, "peak": None}
_job = None                 # None — задачі нема; інакше dict як _total + start


def _log(msg):
    # logging, а не print: так рядок потрапляє і в консоль ComfyUI (панель LOGS), і в comfyui.log
    logging.info("[Термо-сторож] " + msg)


def _fmt_s(s):
    s = int(round(s))
    return "%d с" % s if s < 60 else "%d хв %02d с" % (s // 60, s % 60)


def _peak(t):
    if t is None:
        return
    for d in (_total, _job):
        if d is not None and (d["peak"] is None or t > d["peak"]):
            d["peak"] = t


def _count(kind, seconds):
    """kind: pauses | freezes. Додає подію і час у лічильники запуску й поточної задачі."""
    key_s = "paused_s" if kind == "pauses" else "frozen_s"
    with _lock:
        for d in (_total, _job):
            if d is not None:
                d[kind] += 1
                d[key_s] += seconds


def _send(state, **data):
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("svk.thermal", dict(state=state, **data))
    except Exception:
        pass


# ───────────────────────── м'яка пауза ─────────────────────────

# ───────────────────────── API-ноди (партнерські, віддалені) ─────────────────────────
# API-нода чекає відповіді сервера (генерація оплачена). Пауза sleep-ом блокує той самий цикл,
# у якому крутяться її запити, а заморозка SIGSTOP рве відкриті з'єднання тайм-аутом.
# Локально вона Мак майже не гріє — тож на час її роботи сторож нічого не робить.

_prompt = None          # граф поточної задачі: id ноди → {"class_type": …}
_api_cache = {}         # class_type → чи це API-нода
_api_noted = None       # про яку ноду вже написали в консоль
HOLD_PATH = os.path.join(os.path.dirname(thermal.STATE_PATH), ".thermal_hold.json")


def _is_api_class(ct):
    if ct in _api_cache:
        return _api_cache[ct]
    api = False
    try:
        import nodes
        cls = nodes.NODE_CLASS_MAPPINGS.get(ct)
        if cls is not None:
            api = bool(getattr(cls, "API_NODE", False)) or "api_nodes" in (getattr(cls, "__module__", "") or "")
    except Exception:
        api = False
    _api_cache[ct] = api
    return api


def _current_api():
    """class_type API-ноди, що виконується зараз, або None."""
    pr = _prompt
    if not pr:
        return None
    try:
        from server import PromptServer
        nid = PromptServer.instance.last_node_id
    except Exception:
        return None
    if nid is None:
        return None
    nid = str(nid)
    for k in (nid, nid.split(".")[0], nid.split(":")[0]):
        node = pr.get(k)
        if isinstance(node, dict):
            ct = node.get("class_type")
            return ct if ct and _is_api_class(ct) else None
    return None


def _note_api(ct):
    global _api_noted
    if _api_noted != ct:
        _api_noted = ct
        _log("працює API-нода %s — паузу й заморозку пропускаю, щоб не обірвати запит" % ct)


def _hold(ct):
    """Сказати наглядачу (окремий процес): не заморожувати ще 10 с."""
    try:
        thermal.write_state({"until": time.time() + 10.0, "node": ct}, HOLD_PATH)
    except Exception:
        pass


def _unhold():
    global _api_noted
    _api_noted = None
    try:
        os.remove(HOLD_PATH)
    except OSError:
        pass


def maybe_cool():
    """Викликається перед кожним кроком. Тротл poll_s; при гарячому SoC блокує потік до охолодження."""
    global _last_check
    cfg = _cfg
    if not _active or not cfg["enabled"]:
        return
    api = _current_api()
    if api:
        _note_api(api)
        _hold(api)
        return
    now = time.monotonic()
    with _lock:
        if _last_check is not None and now - _last_check < cfg["poll_s"]:
            return
        _last_check = now
    d = thermal.read_detail()
    t = None if d is None else d["t"]
    _peak(t)
    if t is None or t < cfg["pause_at"]:
        return
    _pause(t, cfg, d["src"])


def _pause(t, cfg, src="SoC"):
    global _pausing
    import comfy.model_management as mm

    resume_at = cfg["resume_at"]
    start = time.monotonic()
    _log("%.0f°C (%s) ≥ %.0f — пауза до %.0f°C" % (t, src, cfg["pause_at"], resume_at))
    interrupted = False
    last_t = now_t = t
    with _lock:
        _pausing += 1
    try:
        while True:
            _send("cooling", t=last_t, src=src, resume_at=resume_at,
                  since=round(time.monotonic() - start, 1))
            end = time.monotonic() + COOL_STEP_S
            while time.monotonic() < end:
                if getattr(mm, "interrupt_processing", False):
                    interrupted = True
                    break
                time.sleep(0.2)
            if interrupted:
                break
            now_d = thermal.read_detail()
            now_t = None if now_d is None else now_d["t"]
            if now_t is None:           # датчик зник посеред паузи — не блокуємо назавжди
                break
            last_t, src = now_t, now_d["src"]
            if now_t <= resume_at:
                break
    finally:
        with _lock:
            _pausing -= 1
    paused_s = round(time.monotonic() - start, 1)
    _count("pauses", paused_s)
    if interrupted:
        _log("пауза перервана скасуванням після %.0f с" % paused_s)
        _send("ok", t=last_t, paused_s=paused_s, interrupted=True)
    elif now_t is None:
        _log("температуру не прочитати під час паузи — продовжую (пауза %.0f с)" % paused_s)
        _send("ok", t=last_t, paused_s=paused_s)
    else:
        _log("%.0f°C (%s) — продовжую (пауза %s)" % (last_t, src, _fmt_s(paused_s)))
        _send("ok", t=last_t, paused_s=paused_s)


def _job_line(d, head):
    pk = "—" if d["peak"] is None else "%.0f°C" % d["peak"]
    return "%s: пік %s · пауз %d (%s) · заморозок %d (%s)" % (
        head, pk, d["pauses"], _fmt_s(d["paused_s"]), d["freezes"], _fmt_s(d["frozen_s"]))


def _job_begin():
    global _job
    d = thermal.read_detail()
    t = None if d is None else d["t"]
    src = None if d is None else d["src"]
    with _lock:
        _job = {"pauses": 0, "paused_s": 0.0, "freezes": 0, "frozen_s": 0.0,
                "peak": t, "start": time.monotonic(), "beat": time.monotonic()}
        _total["jobs"] += 1
    cfg = _cfg
    _log("задача почалась · %s · пауза з %.0f°C, продовжити з %.0f°C, заморозка з %.0f°C" % (
        "—" if t is None else "%.0f°C (%s)" % (t, src), cfg["pause_at"], cfg["resume_at"], cfg["hard_at"]))


def _job_end():
    global _job
    with _lock:
        d, _job = _job, None
    if d is None:
        return
    _log(_job_line(d, "задача завершена за %s" % _fmt_s(time.monotonic() - d["start"])))


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

    # Початок/кінець задачі — для рядків у консоль і підсумку. Сам execute не змінюється.
    try:
        import execution
        PE = execution.PromptExecutor
        orig_exe = PE.execute
        if not getattr(orig_exe, "_svk_thermal", False):
            def execute(self, *args, **kwargs):
                global _prompt
                pr = args[0] if args else kwargs.get("prompt")
                _prompt = pr if isinstance(pr, dict) else None
                try:
                    _job_begin()
                except Exception:
                    pass
                try:
                    return orig_exe(self, *args, **kwargs)
                finally:
                    _prompt = None
                    _unhold()
                    try:
                        _job_end()
                    except Exception:
                        pass
            execute._svk_thermal = True
            PE.execute = execute
    except Exception as e:
        _log("підсумок задач вимкнено: %s" % e)


# ───────────────────────── фоновий звіт для UI ─────────────────────────

def _reporter_loop(seen):
    """Раз на 2 с читає файл стану сторожа і повідомляє UI про заморозку, що вже минула."""
    while True:
        time.sleep(2.0)
        try:
            # пік і пульс під час задачі: ноди без прогрес-бару сторож інакше не бачить
            d = _job
            if d is not None and _cfg["enabled"]:
                api = _current_api()
                if api:                 # API-нода без прогрес-бару теж має бути під захистом
                    _note_api(api)
                    _hold(api)
                t = thermal.read_max_c()
                _peak(t)
                now = time.monotonic()
                if now - d["beat"] >= HEARTBEAT_S and _pausing == 0:
                    d["beat"] = now
                    _log("%s · пік %s · задача йде %s · пауз %d, заморозок %d" % (
                        "—" if t is None else "%.0f°C" % t,
                        "—" if d["peak"] is None else "%.0f°C" % d["peak"],
                        _fmt_s(now - d["start"]), d["pauses"], d["freezes"]))
            st = thermal.read_state()
            if not st or st.get("stopped"):
                continue
            seq = int(st.get("seq", 0) or 0)
            if seq > seen and st.get("last_stop_s") is not None:
                seen = seq
                secs = float(st["last_stop_s"])
                _count("freezes", secs)
                _log("ComfyUI був заморожений на %s (%s ≥ межі %.0f°C), продовжено" % (
                    _fmt_s(secs), "—" if st.get("t") is None else "%.0f°C" % st["t"], _cfg["hard_at"]))
                _send("was_frozen", seconds=secs, t=st.get("t"))
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
    cfg, det = _cfg, thermal.read_detail()
    t = None if det is None else det["t"]
    _log("увімкнено · " + ("—" if det is None else det["src"]) + " %s · профіль «%s»: пауза з %.0f°C, продовжити з %.0f°C, заморозка з %.0f°C · наглядач %s" % (
        "—" if t is None else "%.0f°C" % t,
        thermal.PROFILE_TITLES.get(cfg["profile"], cfg["profile"]),
        cfg["pause_at"], cfg["resume_at"], cfg["hard_at"],
        "працює" if _watch_alive() else "НЕ запущено"))
    return True


def _apply(cfg):
    """Новий конфіг після POST: оновлюємо пам'ять; якщо гейт увімкнули вперше — активуємо."""
    global _cfg
    _cfg = cfg
    if cfg["enabled"] and not _active:
        _activate()


def status():
    cfg = _cfg
    d = thermal.read_detail() or {}
    t = d.get("t")
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
    return {"enabled": cfg["enabled"], "t": t, "src": d.get("src"), "state": state,
            "gpu": d.get("smc_gpu"), "cpu": d.get("smc_cpu"),
            "pause_at": cfg["pause_at"], "resume_at": cfg["resume_at"],
            "hard_at": cfg["hard_at"], "profile": cfg["profile"],
            "profile_title": thermal.PROFILE_TITLES.get(cfg["profile"], cfg["profile"]),
            "poll_s": cfg["poll_s"], "reader_ok": t is not None,
            "active": _active, "watch": _watch_alive(),
            "total": {k: _total[k] for k in ("jobs", "pauses", "paused_s", "freezes", "frozen_s", "peak")},
            "job": None if _job is None else {k: _job[k] for k in ("pauses", "paused_s", "freezes", "frozen_s", "peak")}}


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
