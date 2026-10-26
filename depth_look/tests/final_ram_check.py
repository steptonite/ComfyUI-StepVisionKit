"""Пік RAM фінального (статичного) рендеру Depth Look на синтетичному 12-Мп кадрі + паритет зі старим шляхом. НЕ нода.
ЗАПУСКАТИ ЛИШЕ НА ХМАРНІЙ БОКСІ (Vast/Колаб), НЕ на Маці: рахує повний фінал look_v1 на ~12 Мп (>= 1-2 хв CPU).
  python depth_look/tests/final_ram_check.py                          # новий код: пік RAM, час
  python depth_look/tests/final_ram_check.py --ref-rev 5e62432        # + старий код з git-ревізії: пік RAM обох і max|diff| (/255)
  python depth_look/tests/final_ram_check.py --ref-rev 5e62432 --device cuda   # + torch/CUDA порт проти нового numpy
Опції: --w 4288 --h 2848 (12.2 Мп) · --preset look_v1 · --limit-gb 3.0 · --tracemalloc · --keep (лишити тимчасову теку).
Як міряє: вхід (img.npy / depth.npy) і вихід (out.npy) лежать у тимчасовій теці й відкриваються mmap-ом - так само, як у
render_proc/worker.py. Кожен варіант рахується в ОКРЕМОМУ дочірньому процесі (ru_maxrss монотонний, тож старий і новий
в одному процесі змішались би). Пік = max(RSS-зразків потоку-семплера кожні 20 мс, ru_maxrss); на Linux окремо RssAnon
(анонімна памʼять = реальний тиск на RAM, без файлових mmap-сторінок, які ОС може скинути). baseline = RSS після
відкриття масивів, ДО рендеру; «приріст» = пік - baseline. Тест проходить, якщо приріст <= --limit-gb
(дефолт 3.0 ГБ) і max|diff| (8 біт, /255) <= 1.0. Код виходу 0 = усе ок, 1 = пік вищий за ліміт або diff > 1/255.
Без приватних медіа: кадр і глибина процедурні, зерно фіксоване."""
import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)                       # depth_look/ (поточне дерево)
REPO = os.path.dirname(ENGINE)


def _rss():
    """(rss, rss_anon, rss_file) у байтах; поза Linux anon/file = None."""
    try:
        v = {}
        with open("/proc/self/status") as f:
            for l in f:
                if l.startswith(("RssAnon:", "RssFile:", "VmRSS:")):
                    k, x = l.split(":")
                    v[k] = int(x.split()[0]) * 1024
        return v.get("VmRSS", 0), v.get("RssAnon"), v.get("RssFile")
    except Exception:
        pass
    try:
        import psutil
        return psutil.Process().memory_info().rss, None, None
    except Exception:
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024), None, None


def _maxrss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024


# ---------------------------------------------------------------------------------------------------- синтетика
def make_inputs(job, W, H):
    """img.npy (1,H,W,3) і depth.npy (1,H,W) float32 смугами по рядках (як synth() у parity_torch.py, але без пікової памʼяті)."""
    import numpy as np
    img = np.lib.format.open_memmap(os.path.join(job, "img.npy"), mode="w+", dtype=np.float32, shape=(1, H, W, 3))
    dep = np.lib.format.open_memmap(os.path.join(job, "depth.npy"), mode="w+", dtype=np.float32, shape=(1, H, W))
    rng = np.random.default_rng(12345)
    xx = np.arange(W, dtype=np.float32)[None, :]
    u = xx / W
    for y0 in range(0, H, 256):
        y1 = min(H, y0 + 256)
        yy = np.arange(y0, y1, dtype=np.float32)[:, None]
        v = yy / H
        r = 0.5 + 0.4 * np.sin(6.0 * u + 2 * v)
        g = 0.45 + 0.4 * np.sin(9.0 * v - 3 * u + 1.0)
        b = 0.5 + 0.4 * np.cos(5.0 * u + 7 * v)
        band = np.stack(np.broadcast_arrays(r, g, b), -1).astype(np.float32)
        for cx, cy, rad, col in ((0.62, 0.55, 0.16, (0.95, 0.8, 0.7)), (0.25, 0.3, 0.07, (0.2, 0.3, 0.9))):
            m = np.clip((rad - np.hypot((u - cx) * W / H, v - cy)) / 0.01, 0, 1)[..., None]
            band = band * (1 - m) + np.array(col, np.float32) * m
        band += (np.sin(xx * 0.9) > 0.6).astype(np.float32)[..., None] * 0.1
        band += rng.normal(0.0, 0.015, band.shape).astype(np.float32)        # дрібна текстура, щоб не було ідеально гладко
        img[0, y0:y1] = np.clip(band, 0, 1)
        d = 0.12 + 0.45 * v + 0 * u
        subj = np.clip((0.2 - np.hypot((u - 0.62) * W / H, (v - 0.55) * 1.2)) / 0.06, 0, 1)
        box = ((u > 0.1) & (u < 0.3) & (v > 0.5) & (v < 0.85)).astype(np.float32) * 0.65
        d = np.maximum(np.maximum(d, subj * 0.95), box * np.clip(v, 0.5, 1))
        dep[0, y0:y1] = np.clip(d + 0.02 * np.sin(xx * 0.05), 0, 1)
    img.flush()
    dep.flush()
    del img, dep


# ---------------------------------------------------------------------------------------------------- дочірній процес
def child(a):
    """Рахує ОДИН варіант і друкує JSON-рядок (остання лінія stdout, префікс RESULT )."""
    if a.engine:
        sys.path.insert(0, a.engine)                 # старий рушій з git-ревізії (або поточний)
    os.environ.pop("SVK_TORCH", None)                # device задається лише прапорцем --device, а не змінною середовища
    import gc
    import numpy as np
    import core
    job = a.job
    img = np.load(os.path.join(job, "img.npy"), mmap_mode="r")
    dep = np.load(os.path.join(job, "depth.npy"), mmap_mode="r")
    out = np.load(os.path.join(job, "out_%s.npy" % a.tag), mmap_mode="r+")
    gc.collect()
    base, base_anon, _ = _rss()
    peak = [base, base_anon or 0]
    stop = threading.Event()

    def sampler():
        while not stop.is_set():
            r, an, _f = _rss()
            if r > peak[0]:
                peak[0] = r
            if an and an > peak[1]:
                peak[1] = an
            time.sleep(0.02)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()
    if a.tracemalloc:
        import tracemalloc
        tracemalloc.start()
    kw = dict(preset_name=a.preset, static=True, clean_depth=True, workers=1, out=out)
    if a.device:
        kw["device"] = a.device
    t0 = time.time()
    core.apply_numpy(img, dep, **kw)
    sec = time.time() - t0
    out.flush()
    stop.set()
    th.join()
    res = dict(tag=a.tag, engine=os.path.realpath(core.__file__), seconds=round(sec, 1),
               base_mb=round(base / 2**20), peak_mb=round(max(peak[0], _maxrss()) / 2**20),
               ru_maxrss_mb=round(_maxrss() / 2**20), base_anon_mb=round((base_anon or 0) / 2**20),
               peak_anon_mb=round(peak[1] / 2**20) if base_anon is not None else None)
    if a.tracemalloc:
        cur, trpeak = tracemalloc.get_traced_memory()
        res["tracemalloc_peak_mb"] = round(trpeak / 2**20)
    if a.device and a.device.startswith("cuda"):
        try:
            import torch
            res["cuda_peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        except Exception:
            pass
    print("RESULT " + json.dumps(res), flush=True)


# ---------------------------------------------------------------------------------------------------- батько
def run_variant(a, job, tag, engine, device=None):
    """Окремий процес для одного варіанта. -> dict результату."""
    import numpy as np
    H, W = a.h, a.w
    np.lib.format.open_memmap(os.path.join(job, "out_%s.npy" % tag), mode="w+", dtype=np.float32, shape=(1, H, W, 3)).flush()
    cmd = [sys.executable, os.path.abspath(__file__), "--child", "--job", job, "--tag", tag, "--engine", engine,
           "--preset", a.preset]
    if device:
        cmd += ["--device", device]
    if a.tracemalloc:
        cmd.append("--tracemalloc")
    print("[%s] %s" % (tag, " ".join(cmd[2:])), flush=True)
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        print(p.stdout[-1500:])
        print(p.stderr[-3000:])
        raise SystemExit("[%s] child failed rc=%s" % (tag, p.returncode))
    line = [l for l in p.stdout.splitlines() if l.startswith("RESULT ")][-1]
    res = json.loads(line[len("RESULT "):])
    grow = res["peak_mb"] - res["base_mb"]
    print("[%s] %.1f s | RSS baseline %d MB -> peak %d MB (приріст %d MB%s)%s" % (
        tag, res["seconds"], res["base_mb"], res["peak_mb"], grow,
        (", RssAnon peak %d MB" % res["peak_anon_mb"]) if res.get("peak_anon_mb") is not None else "",
        (" | tracemalloc peak %d MB" % res["tracemalloc_peak_mb"]) if "tracemalloc_peak_mb" in res else ""), flush=True)
    res["grow_mb"] = grow
    return res


def max_diff_255(job, ta, tb, H, W):
    """max і mean |a-b| у /255 після 8-бітного квантування (round(x*255)), смугами по рядках."""
    import numpy as np
    A = np.load(os.path.join(job, "out_%s.npy" % ta), mmap_mode="r")
    B = np.load(os.path.join(job, "out_%s.npy" % tb), mmap_mode="r")
    mx = 0.0
    sm = 0.0
    n = 0
    for y0 in range(0, H, 256):
        qa = np.rint(np.clip(A[0, y0:y0 + 256], 0, 1) * 255.0)
        qb = np.rint(np.clip(B[0, y0:y0 + 256], 0, 1) * 255.0)
        d = np.abs(qa - qb)
        mx = max(mx, float(d.max()))
        sm += float(d.sum())
        n += d.size
    return mx, sm / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w", type=int, default=4288)
    ap.add_argument("--h", type=int, default=2848)
    ap.add_argument("--preset", default="look_v1")
    ap.add_argument("--ref-rev", default="", help="git-ревізія СТАРОГО рушія (напр. 5e62432 = до економного фіналу): рахується окремо, для паритету й порівняння піку")
    ap.add_argument("--device", default="", help="додатково: look у look_torch на пристрої (cuda|cpu|mps) проти нового numpy")
    ap.add_argument("--limit-gb", type=float, default=3.0, help="ліміт приросту піку RSS нового коду, ГБ")
    ap.add_argument("--tracemalloc", action="store_true", help="додатково tracemalloc-пік (повільніше)")
    ap.add_argument("--keep", action="store_true")
    # службові (дочірній процес)
    ap.add_argument("--child", action="store_true")
    ap.add_argument("--job", default="")
    ap.add_argument("--tag", default="new")
    ap.add_argument("--engine", default="")
    a = ap.parse_args()
    if a.child:
        return child(a)

    import numpy as np  # noqa: F401
    job = tempfile.mkdtemp(prefix="final_ram_check_")
    old_dir = None
    ok = True
    try:
        print("кадр %dx%d = %.1f Мп, preset=%s, тека %s" % (a.w, a.h, a.w * a.h / 1e6, a.preset, job), flush=True)
        make_inputs(job, a.w, a.h)
        new = run_variant(a, job, "new", ENGINE)
        lim = a.limit_gb * 1024
        verdict = "OK" if new["grow_mb"] <= lim else "ПЕРЕВИЩЕНО"
        if new["grow_mb"] > lim:
            ok = False
        print("НОВИЙ: приріст піку %d MB (ліміт %d MB) -> %s" % (new["grow_mb"], lim, verdict))
        if a.ref_rev:
            old_dir = tempfile.mkdtemp(prefix="final_ram_check_old_")
            tar = subprocess.run(["git", "-C", REPO, "archive", a.ref_rev, os.path.relpath(ENGINE, REPO)], stdout=subprocess.PIPE, check=True)
            subprocess.run(["tar", "-x", "-C", old_dir], input=tar.stdout, check=True)
            old = run_variant(a, job, "old", os.path.join(old_dir, os.path.relpath(ENGINE, REPO)))
            mx, mean = max_diff_255(job, "new", "old", a.h, a.w)
            print("СТАРИЙ (%s): приріст піку %d MB; НОВИЙ: %d MB (x%.2f менше)" % (
                a.ref_rev, old["grow_mb"], new["grow_mb"], old["grow_mb"] / max(new["grow_mb"], 1)))
            print("паритет нове vs старе (8 біт): max |diff| = %.0f/255, mean = %.5f/255 -> %s" % (mx, mean, "OK" if mx <= 1.0 else "ЗАНАДТО ВЕЛИКИЙ"))
            if mx > 1.0:
                ok = False
        if a.device:
            tor = run_variant(a, job, "torch", ENGINE, device=a.device)
            mx, mean = max_diff_255(job, "torch", "new", a.h, a.w)
            print("паритет torch(%s) vs нове numpy (8 біт): max |diff| = %.0f/255, mean = %.5f/255 -> %s" % (
                a.device, mx, mean, "OK" if mx <= 1.0 else "ПОДИВИТИСЬ"))
            if mx > 1.0:
                ok = False
        print("ПІДСУМОК:", "ОК" if ok else "ПРОВАЛ")
    finally:
        if not a.keep:
            shutil.rmtree(job, ignore_errors=True)
            if old_dir:
                shutil.rmtree(old_dir, ignore_errors=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
