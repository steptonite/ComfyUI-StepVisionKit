"""Перевірка потокової глибини (SVKVideoDepthStream) проти старого ланцюга. НЕ нода.
ЗАПУСКАТИ ЛИШЕ НА ХМАРНІЙ БОКСІ (Vast) у python ComfyUI, НЕ на Маці: DA3 на всьому кліпі, старий ланцюг тримає кліп у RAM.
  python depth_look/tests/depth_stream_check.py --video clip.mp4 --frames 96 --work /workspace/dsc       # ref + new + cache + broken
  python depth_look/tests/depth_stream_check.py --video clip_2k.mp4 --frames 0 --work /workspace/dsc2k --only new,cache
Варіанти (кожен - ОКРЕМИЙ дочірній процес, щоб пік RAM не змішувався):
  ref    Load Video Pro → Image Scale By (area 0.5) → Run DA3 (518, mono) → Depth Receiver - як у робочих воркфлоу;
  new    SVKVideoDepthStream з тими ж налаштуваннями, порожній кеш;
  cache  те саме вдруге з тим самим кешем: DA3 не має рахуватись (час < --cache-s), вихід байт-у-байт той самий;
  broken навмисно хибна копія (scale_method=bilinear замість area): порівняння МУСИТЬ її відхилити, інакше тест сліпий.
Мак (MPS), ~30 кадрів,; друкує пік RAM і с/кадр (venv/bin/python з теки ComfyUI):
  venv/bin/python custom_nodes/ComfyUI-StepVisionKit/depth_look/tests/depth_stream_check.py --video clip.mp4 --frames 30 \
      --work /Volumes/…/dsc_mac --only ref,new --ram-gb 2
  (--dtype fp32, якщо fp16 на MPS дасть NaN - нода сама скаже)
Пройдено, якщо: max|ref - new| <= --tol (1e-6) · broken > --tol · пік RssAnon new - baseline <= --limit-gb ·
cache < --cache-s і sha256 як у new · (якщо є ref) час new <= ref × (1 + --slow). Код виходу 0 = усе ок, 1 = ні."""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PACK = os.path.dirname(os.path.dirname(HERE))
COMFY = os.path.dirname(os.path.dirname(PACK))
MODEL = "depth_anything_3_mono_large.safetensors"


def _rss_anon():
    """(anon, rss) у байтах. Linux - RssAnon з /proc; macOS - psutil (на Маці anon нема, беремо RSS)."""
    if not os.path.exists("/proc/self/status"):
        try:
            import psutil
            r = psutil.Process().memory_info().rss
        except ImportError:
            import resource
            r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r, r
    v = {}
    with open("/proc/self/status") as f:
        for l in f:
            if l.startswith(("RssAnon:", "VmRSS:")):
                k, x = l.split(":")
                v[k] = int(x.split()[0]) * 1024
    return v.get("RssAnon", v.get("VmRSS", 0)), v.get("VmRSS", 0)


class Sampler:
    def __init__(self):
        self.peak_anon = self.peak_rss = 0
        self._stop = threading.Event()
        self._t = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            a, r = _rss_anon()
            self.peak_anon, self.peak_rss = max(self.peak_anon, a), max(self.peak_rss, r)
            time.sleep(0.02)

    def __enter__(self):
        self._t.start()
        return self

    def __exit__(self, *a):
        self._stop.set()
        self._t.join()


def _boot():
    """ComfyUI як бібліотека: аргументи за замовчуванням, пак - пакетом без його __init__ (без сервера й решти нод)."""
    import importlib
    import types
    sys.argv = sys.argv[:1]
    sys.path.insert(0, COMFY)
    os.chdir(COMFY)
    pkg = types.ModuleType("svk")
    pkg.__path__ = [PACK]
    sys.modules["svk"] = pkg
    return importlib


def child(a):
    imp = _boot()
    import numpy as np
    import torch
    from comfy_extras.nodes_depth_anything_3 import LoadDA3Model, DA3Inference
    a0, _ = _rss_anon()
    t0 = time.time()
    model = LoadDA3Model.execute(MODEL, a.dtype).result[0]
    t_model = time.time() - t0
    base, _ = _rss_anon()                       # після імпортів і моделі в RAM: далі - лише те, що росте з кліпом
    t1 = time.time()
    with Sampler() as smp:
        if a.child == "ref":
            import nodes
            V = imp.import_module("svk.media.video")
            R = imp.import_module("svk.core.depth_raw")
            img = V.SVKLoadVideoPro().load("", 0, a.frames, "as encoded", path=a.video)[0]
            half = nodes.ImageScaleBy().upscale(img, "area", 0.5)[0]
            del img
            geo = DA3Inference._execute_mono(model, half, 518, "upper_bound_resize").result[0]
            del half
            dep = R.SVKDepthReceiver().receive(geo, True)[0]
            del geo
        else:
            S = imp.import_module("svk.depth_look.depth_stream")
            sm = "bilinear" if a.child == "broken" else "area"
            b = int(a.child[5:]) if a.child.startswith("batch") else 1
            dep, n, info = S.SVKVideoDepthStream().run(model, "", 0, a.frames, 0.5, 518, True, True, batch=b, ram_budget_gb=a.ram_gb,
                                                       path=a.video, scale_method=sm, cache_folder=a.cache)
            print(info)
        secs = time.time() - t1
        out = np.lib.format.open_memmap(a.out, mode="w+", dtype=np.float32, shape=tuple(dep.shape[:3]))
        for i in range(dep.shape[0]):
            out[i] = dep[i, ..., 0].numpy()
        out.flush()
    r = {"variant": a.child, "frames": int(dep.shape[0]), "shape": list(dep.shape[1:3]), "secs": round(secs, 2),
         "model_load_s": round(t_model, 2), "anon_start_gb": round(a0 / 2**30, 2), "anon_base_gb": round(base / 2**30, 2),
         "anon_peak_gb": round(smp.peak_anon / 2**30, 2), "rss_peak_gb": round(smp.peak_rss / 2**30, 2),
         "anon_growth_gb": round((smp.peak_anon - base) / 2**30, 2),
         "s_per_frame": round(secs / max(int(dep.shape[0]), 1), 3), "dtype": a.dtype,
         "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else
         (round(torch.mps.driver_allocated_memory() / 2**30, 2) if torch.backends.mps.is_available() else None)}
    print("RESULT " + json.dumps(r))


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def _maxdiff(p, q):
    import numpy as np
    x, y = np.load(p, mmap_mode="r"), np.load(q, mmap_mode="r")
    if x.shape != y.shape:
        return float("inf")
    return max(float(np.abs(x[i] - y[i]).max()) for i in range(x.shape[0]))


def run(variant, a):
    out = os.path.join(a.work, f"{variant}.npy")
    cache = os.path.join(a.work, "cache" if variant in ("new", "cache") else "cache_" + variant)
    cmd = [sys.executable, os.path.abspath(__file__), "--child", variant, "--video", a.video, "--frames", str(a.frames),
           "--out", out, "--cache", cache, "--work", a.work, "--dtype", a.dtype, "--ram-gb", str(a.ram_gb)]
    t = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(p.stdout[-3000:])
    if p.returncode:
        sys.stdout.write(p.stderr[-3000:])
        if variant.startswith("batch"):                       # пачка може не влізти у VRAM - це результат заміру, не збій тесту
            return {"variant": variant, "failed": "rc=%d %s" % (p.returncode, p.stderr.strip().splitlines()[-1][:200] if p.stderr.strip() else "")}
        raise SystemExit(f"{variant}: child failed rc={p.returncode}")
    r = json.loads([l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1][7:])
    r["wall_s"] = round(time.time() - t, 1)
    r["out"] = out
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--work", required=True)
    ap.add_argument("--only", default="ref,new,cache,broken")
    ap.add_argument("--tol", type=float, default=1e-6)
    ap.add_argument("--limit-gb", type=float, default=3.0)
    ap.add_argument("--cache-s", type=float, default=5.0)
    ap.add_argument("--slow", type=float, default=0.10)
    ap.add_argument("--dtype", default="fp16", help="weight_dtype для Load DA3 (fp16 / bf16 / fp32)")
    ap.add_argument("--ram-gb", type=float, default=0.0, help="ram_budget_gb ноди: 0 = вихід на диск (замір RAM), >0 = у RAM, якщо влазить")
    ap.add_argument("--batches", default="", help="пачки DA3 для заміру, напр. 4,8 (кожна - окремий процес)")
    ap.add_argument("--batch-tol", type=float, default=1e-3, help="допуск для пачки >1: різниця лише від округлення GPU")
    ap.add_argument("--child")
    ap.add_argument("--out")
    ap.add_argument("--cache")
    a = ap.parse_args()
    a.video = os.path.abspath(a.video)
    if a.child:
        return child(a)
    os.makedirs(a.work, exist_ok=True)
    want = a.only.split(",") + ["batch%d" % int(b) for b in a.batches.split(",") if b.strip()]
    for d in ["cache"] + ["cache_" + v for v in want]:
        shutil.rmtree(os.path.join(a.work, d), ignore_errors=True)
    res, checks = {}, []
    for v in [v for v in ("ref", "new", "cache", "broken") if v in want] + [v for v in want if v.startswith("batch")]:
        if True:
            res[v] = run(v, a)
            print(f"[{v}] " + json.dumps(res[v]))
    if "new" in res:
        g = res["new"]["anon_growth_gb"]
        lim = a.limit_gb + (a.ram_gb if a.ram_gb else 0)        # вихід у RAM у межах стелі - законний приріст
        checks.append(("RAM growth new <= %.1f GB" % lim, g <= lim, g))
    if "ref" in res and "new" in res:
        d = _maxdiff(res["ref"]["out"], res["new"]["out"])
        checks.append(("max|ref-new| <= %g" % a.tol, d <= a.tol, d))
        lim = res["ref"]["secs"] * (1 + a.slow)
        checks.append(("time new <= ref +%d%%" % (a.slow * 100), res["new"]["secs"] <= lim, (res["new"]["secs"], res["ref"]["secs"])))
    if "ref" in res and "broken" in res:
        d = _maxdiff(res["ref"]["out"], res["broken"]["out"])
        checks.append(("broken copy is caught (diff > tol)", d > a.tol, d))
    for v in [v for v in res if v.startswith("batch")]:          # пачка: швидкість і відхилення від ref (лише округлення GPU)
        if "failed" in res[v]:
            checks.append(("%s: did not run (info)" % v, True, res[v]["failed"]))
            continue
        base = res.get("new") or res.get("ref")
        d = _maxdiff(res["ref"]["out"], res[v]["out"]) if "ref" in res else float("nan")
        checks.append(("%s: max|ref-new| <= %g" % (v, a.batch_tol), d <= a.batch_tol, d))
        if base:
            checks.append(("%s: speed vs %s (info)" % (v, "new" if "new" in res else "ref"), True,
                           "%.2f s vs %.2f s = x%.2f" % (res[v]["secs"], base["secs"], base["secs"] / max(res[v]["secs"], 1e-9))))
    if "cache" in res and "new" in res:
        checks.append(("cache rerun < %.0f s" % a.cache_s, res["cache"]["secs"] < a.cache_s, res["cache"]["secs"]))
        same = _sha(res["cache"]["out"]) == _sha(res["new"]["out"])
        checks.append(("cache output sha256 == new", same, same))
    ok = all(c[1] for c in checks)
    print("\n=== depth_stream_check: %s frames=%d" % (os.path.basename(a.video), a.frames))
    for name, good, val in checks:
        print("  %-4s %-40s %s" % ("OK" if good else "FAIL", name, val))
    with open(os.path.join(a.work, "summary.json"), "w") as f:
        json.dump({"results": res, "checks": [[c[0], c[1], str(c[2])] for c in checks], "ok": ok}, f, indent=1)
    print("ALL OK" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
