"""Спільний запуск рендеру для всіх нод Depth Look (статична, проста, відео): тензори -> numpy -> воркер -> тензор.
Рендер іде в окремому процесі (worker.py): уся памʼять вертається ОС після кожного прогону.
Torch імпортується лише всередині функцій, що його потребують."""
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid

import numpy as np

from . import core

_PARENT = os.path.dirname(os.path.realpath(__file__))     # _cache/jobs поруч з рушієм
_HERE = os.path.dirname(os.path.realpath(__file__))


def _interrupted():
    """Чи натиснув користувач Cancel у Comfy (повертає модуль model_management або None)."""
    try:
        import comfy.model_management as mm
        return mm if mm.processing_interrupted() else None
    except Exception:
        return None


def _kill_group(proc):
    """Прибити воркер І його паралельних дітей (група процесу = pid воркера, див. start_new_session)."""
    try:
        if proc.poll() is None:
            os.killpg(proc.pid, 9)
        elif hasattr(os, "killpg"):
            os.killpg(proc.pid, 9)                         # воркер вийшов, а діти ще могли лишитись
    except Exception:
        pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def _render_subproc(frames, dep, kw, progress, out_shape=None):
    """Рендер у дочірньому процесі worker.py. -> (B,Ho,Wo,3) float32 ndarray; None, якщо воркер не стартував.
    out_shape за замовч. = форма входу. Завдання лежить у _cache/jobs/<id>/ (диск проєкту, не /tmp) і
    видаляється завжди, навіть при помилці."""
    import queue
    if out_shape is None:
        out_shape = tuple(frames.shape[:3]) + (3,)
    jobs = os.path.join(_PARENT, "_cache", "jobs")
    os.makedirs(jobs, exist_ok=True)
    for old in os.listdir(jobs):                           # залишки прогону, який убили посеред рендеру (finally не встиг)
        shutil.rmtree(os.path.join(jobs, old), ignore_errors=True)
    need = frames.nbytes + dep.nbytes + 4 * int(np.prod(out_shape))
    free = shutil.disk_usage(jobs).free
    if free < need + 2**30:
        raise RuntimeError("DepthLook: this clip needs ≈%.1f GB of free disk for rendering, only %.1f GB free. "
                           "Delete old outputs or render fewer frames." % (need / 1e9, free / 1e9))
    job = os.path.join(jobs, uuid.uuid4().hex[:12])
    proc = None
    try:
        if getattr(progress, "stage", None):
            progress.stage("start")
        os.makedirs(job)
        np.save(os.path.join(job, "img.npy"), frames)
        np.save(os.path.join(job, "depth.npy"), dep)
        json.dump(kw, open(os.path.join(job, "args.json"), "w", encoding="utf-8"), ensure_ascii=False)
        mm_out = np.lib.format.open_memmap(os.path.join(job, "out.npy"), mode="w+", dtype=np.float32, shape=tuple(out_shape))
        del mm_out
        errf = open(os.path.join(job, "err.txt"), "w")
        try:
            proc = subprocess.Popen([sys.executable, os.path.join(_HERE, "worker.py"), job],
                                    stdout=subprocess.PIPE, stderr=errf, text=True, bufsize=1,
                                    start_new_session=True)   # своя група: kill прибиває й паралельних дітей воркера
        except Exception as e:
            print("[DepthLook] worker failed to start (%s) - computing in the Comfy process" % e)
            return None
        q = queue.Queue()
        def _pump():                                       # stdout воркера -> черга (None = кінець)
            for l in proc.stdout:
                q.put(l)
            q.put(None)
        threading.Thread(target=_pump, daemon=True).start()
        done = False
        while not done:
            try:
                line = q.get(timeout=0.25)
            except queue.Empty:
                line = ""
            if line is None:
                done = True
            elif line.startswith("STAGE "):                # етап зсередини рушія (core.apply_numpy)
                if getattr(progress, "stage", None):
                    progress.stage(line.split()[1])
            elif line.startswith("PROGRESS "):
                try:
                    k, n = line.split()[1:3]
                    if progress:
                        progress(int(k), int(n))
                except Exception:
                    pass
            elif line.strip():
                print(line.rstrip())
            mm = _interrupted()
            if mm is not None:
                _kill_group(proc)
                raise mm.InterruptProcessingException()
        rc = proc.wait()
        if rc != 0:
            errf.flush()
            tail = open(os.path.join(job, "err.txt"), errors="replace").read()[-1500:]
            raise RuntimeError("DepthLook worker exited with code %s:\n%s" % (rc, tail))
        # результат лишається на диску (mmap), не копія в RAM: кліп 1080p не тримається в памʼяті вдруге.
        # rmtree нижче лише відвʼязує файл - дані живуть, доки масив/тензор не звільнять
        return np.load(os.path.join(job, "out.npy"), mmap_mode="r+")
    finally:
        if proc is not None:
            _kill_group(proc)
        shutil.rmtree(job, ignore_errors=True)


def tensors_to_numpy(image, depth):
    """IMAGE-тензори Comfy -> (frames (B,H,W,3) float32, dep (B|1,h,w) float32) без зайвих копій."""
    import torch
    img_t = image.detach()
    if img_t.dtype != torch.float32:
        img_t = img_t.float()
    dep_t = depth.detach()
    if dep_t.dtype != torch.float32:
        dep_t = dep_t.float()
    frames = img_t.cpu().numpy()                           # вʼю без копії для CPU-тензора
    dep = dep_t.cpu().numpy()
    dep = dep[..., 0] if dep.ndim == 4 else dep            # канал 0
    return frames, dep


LAST_STAGE = {}      # node id -> останній етап; вкладка після обриву звʼязку забирає його з /svk/stages


def stage(uid, name, **kw):
    """Етап виконання -> рядок стану на картці ноди (подія svk.stage; js: svk_ui.stageLine). Без Comfy або id - нічого."""
    if uid is None:
        return
    s = dict(node=str(uid), stage=name, **kw)
    LAST_STAGE[str(uid)] = dict(s, t=time.time())
    try:
        from server import PromptServer
        PromptServer.instance.send_sync("svk.stage", s)
    except Exception:
        pass


try:
    from server import PromptServer
    from aiohttp import web

    @PromptServer.instance.routes.get("/svk/stages")
    async def _svk_stages(request):
        now = time.time()
        return web.json_response({k: v for k, v in LAST_STAGE.items() if now - v["t"] < 6 * 3600})
except Exception:
    pass


def make_progress(n, uid=None):
    """Смужка прогресу по кадрах у Comfy + «кадр k / N · ~скільки лишилось» на картці; без Comfy -> лише рядок стану.
    progress.stage(name) - етап зсередини рендеру (запуск рушія, аналіз руху камери)."""
    try:
        import comfy.utils
        pbar = comfy.utils.ProgressBar(n)
    except Exception:
        pbar = None
    t0 = [None]
    last = [0.0]

    def progress(k, m):
        if pbar is not None:
            pbar.update_absolute(k, m)
        now = time.time()
        if t0[0] is None:                       # відлік ETA - від першого готового кадру, без старту воркера
            t0[0] = (now, k)
        if now - last[0] >= 0.4 or k >= m:
            last[0] = now
            t, k0 = t0[0]
            eta = (now - t) / (k - k0) * (m - k) if k > k0 else 0.0
            stage(uid, "frame", k=int(k), n=int(m), eta=round(eta, 1))
    progress.stage = lambda name, **kw: stage(uid, name, **kw)
    return progress


def render(frames, dep, kw, progress, out_shape):
    """Воркер (за замовч.) або, якщо DEPTHLOOK_INPROC=1 / воркер не стартував, - у цьому процесі. -> ndarray (out_shape)."""
    arr = None
    if os.environ.get("DEPTHLOOK_INPROC") != "1":
        arr = _render_subproc(frames, dep, kw, progress, out_shape)
    if arr is None:
        arr = np.empty(tuple(out_shape), np.float32)
        core.apply_numpy(frames, dep, out=arr, progress=progress, workers=1, **kw)   # у процесі Comfy fork небезпечний
    return arr


def to_result(arr, image):
    """ndarray -> тензор на пристрої входу."""
    import torch
    res = torch.from_numpy(arr)
    if image.device.type != "cpu":
        res = res.to(image.device)
    return res
