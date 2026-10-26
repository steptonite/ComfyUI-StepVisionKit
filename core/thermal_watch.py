"""Жорсткий запобіжник термо-сторожа: окремий процес-наглядач за ComfyUI.

ЧОМУ: м'яка пауза (thermal_guard) спрацьовує лише там, де є прогрес-бар. Ноди без нього
(YOLO тощо) м'якої паузи не мають. Тому наглядач стежить за температурою збоку, і при
жорсткому порозі ставить ПРОЦЕС ComfyUI НА ПАУЗУ (SIGSTOP), а після охолодження знімає
її (SIGCONT). Задачу не вбиває, лише зупиняє.

Запуск: thermal_guard.install() → start() тут: Popen([python, thermal_watch.py, --pid,
<ComfyUI>, --config, ..., --state, ...]). Один наглядач на процес ComfyUI.

🔴 Заморожений ComfyUI не має лишитися замороженим:
  - SIGCONT у finally і в обробниках SIGTERM/SIGINT/SIGHUP;
  - якщо читач температури повернув None під час заморозки, одразу SIGCONT;
  - стеля заморозки MAX_FREEZE_S → SIGCONT + попередження. Далі заморожувати не будемо,
    доки SoC не охолоне до resume_at (краще гаряче, ніж мовчки мертве);
  - батько (ComfyUI) зник або pid більше не живий → наглядач виходить.

Модуль не імпортує пакет ComfyUI: читач і конфіг беруться з thermal.py за шляхом.
"""

import argparse
import importlib.util
import os
import signal
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
MAX_FREEZE_S = 900.0        # стеля однієї заморозки, с
CFG_RELOAD_S = 10.0         # як часто перечитувати thermal_config.json (зміни з UI)


def _log(msg):
    try:
        print("[Thermal guard] " + msg, file=sys.stderr, flush=True)
    except Exception:
        pass


def _load_thermal():
    spec = importlib.util.spec_from_file_location("svk_thermal_core", os.path.join(_HERE, "thermal.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Watch:
    def __init__(self, th, target, config_path, state_path):
        self.th = th
        self.target = target
        self.config_path = config_path
        self.state_path = state_path
        self.parent0 = os.getppid()
        self.cfg = th.load_config(config_path)
        self.cfg_t = time.monotonic()
        self.stopped = False
        self.freeze_t0 = None
        self.cap_latch = False
        self.t_freeze = None
        self.seq = int((th.read_state(state_path) or {}).get("seq", 0) or 0)
        self.hold_path = os.path.join(os.path.dirname(state_path), ".thermal_hold.json")
        self.hold_noted = None

    # ── процес-ціль ──

    def _target_alive(self):
        if os.getppid() != self.parent0:
            return False                        # батько помер, нас перепідхопив launchd
        try:
            os.kill(self.target, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _signal(self, sig):
        try:
            os.kill(self.target, sig)
            return True
        except ProcessLookupError:
            return False
        except Exception as e:
            _log("signal %s not sent: %s" % (sig, e))
            return False

    # ── заморозка і розморожування ──

    def _freeze(self, t):
        if not self._signal(signal.SIGSTOP):
            return
        self.stopped = True
        self.freeze_t0 = time.time()
        self.t_freeze = t
        self.seq += 1
        self.th.write_state({"stopped": True, "since": self.freeze_t0, "t": t,
                             "seq": self.seq, "pid": self.target}, self.state_path)
        _log("%.0f°C ≥ %.0f — ComfyUI (pid %d) frozen, waiting to cool down to %.0f°C"
             % (t, self.cfg["hard_at"], self.target, self.cfg["resume_at"]))

    def _resume(self, reason):
        self._signal(signal.SIGCONT)
        if not self.stopped:
            return
        secs = round(time.time() - self.freeze_t0, 1)
        self.stopped = False
        self.th.write_state({"stopped": False, "last_stop_s": secs, "seq": self.seq, "t": self.t_freeze,
                             "ended": time.time(), "reason": reason, "pid": self.target},
                            self.state_path)
        _log("unfrozen (%s) after %.0f s" % (reason, secs))

    def _tick(self):
        cfg = self.cfg
        t = self.th.read_max_c()
        if self.stopped:
            if t is None:
                self._resume("temperature sensor unavailable")
            elif not cfg["enabled"]:
                self._resume("gate disabled")
            elif t <= cfg["resume_at"]:
                self._resume("cooled down to %.0f°C" % t)
            elif time.time() - self.freeze_t0 >= MAX_FREEZE_S:
                self.cap_latch = True
                _log("WARNING: already frozen for %.0f s, SoC %.0f°C — lifting the freeze. "
                     "The next freeze only after cooling down to %.0f°C"
                     % (MAX_FREEZE_S, t, cfg["resume_at"]))
                self._resume("freeze ceiling")
            else:
                self.th.write_state({"stopped": True, "since": self.freeze_t0, "t": t,
                                     "seq": self.seq, "pid": self.target}, self.state_path)
            return
        if t is None or not cfg["enabled"]:
            return
        if self.cap_latch:
            if t <= cfg["resume_at"]:
                self.cap_latch = False
            return
        if t >= cfg["hard_at"]:
            # API-нода чекає відповіді сервера: SIGSTOP обірвав би запит (і оплачену генерацію)
            hold = self.th.read_state(self.hold_path) or {}
            if float(hold.get("until", 0) or 0) > time.time():
                if self.hold_noted != hold.get("node"):
                    self.hold_noted = hold.get("node")
                    _log("%.0f°C ≥ %.0f, but API node %s is running — skipping freeze"
                         % (t, cfg["hard_at"], self.hold_noted))
                return
            self.hold_noted = None
            self._freeze(t)

    def run(self):
        while True:
            if not self._target_alive():
                _log("ComfyUI (pid %d) is no longer alive — watcher exiting" % self.target)
                return
            now = time.monotonic()
            if now - self.cfg_t >= CFG_RELOAD_S:
                self.cfg = self.th.load_config(self.config_path)
                self.cfg_t = now
            self._tick()
            time.sleep(self.cfg["poll_s"])

    def release(self):
        """Останній рубіж: ComfyUI не лишається замороженим, що б не сталося."""
        if self.stopped:
            self._resume("watcher exited")
        self._signal(signal.SIGCONT)


def main(argv=None):
    ap = argparse.ArgumentParser(description="StepVisionKit thermal guard hard failsafe")
    ap.add_argument("--pid", type=int, required=True, help="pid of the ComfyUI process")
    ap.add_argument("--config", default=None)
    ap.add_argument("--state", default=None)
    a = ap.parse_args(argv)

    th = _load_thermal()
    w = _Watch(th, a.pid, a.config or th.CONFIG_PATH, a.state or th.STATE_PATH)

    def _on_signal(_signum, _frame):
        raise SystemExit(0)

    for s in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(s, _on_signal)
    try:
        if th.read_max_c() is None:
            _log("cannot read temperature — watcher not started")
            return 0
        _log("watching pid %d: freeze at %.0f°C, unfreeze at %.0f°C"
             % (a.pid, w.cfg["hard_at"], w.cfg["resume_at"]))
        w.run()
    except SystemExit:
        pass
    except Exception as e:
        _log("watcher error: %s" % e)
    finally:
        w.release()
    return 0


def start(config_path, state_path):
    """Запускає наглядача за поточним процесом ComfyUI. Повертає Popen або None (вже працює)."""
    global _proc
    if _proc is not None and _proc.poll() is None:
        return None
    cmd = [sys.executable, os.path.abspath(__file__), "--pid", str(os.getpid()),
           "--config", config_path, "--state", state_path]
    _proc = subprocess.Popen(cmd, start_new_session=True, stdin=subprocess.DEVNULL, close_fds=True)
    return _proc


_proc = None

if __name__ == "__main__":
    sys.exit(main())
