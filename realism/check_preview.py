"""Мікротест превʼю, лупи, власного зерна й пресетів Camera Realism (≤256 px, без Comfy). Запуск: python -m realism.check_preview"""
import os, shutil, sys, tempfile
import numpy as np

from . import params as P, preview, presets
from .engine import apply_realism

fails = []


def check(name, ok, extra=""):
    print(("ok   " if ok else "FAIL ") + name + ("  " + extra if extra else ""))
    if not ok:
        fails.append(name)


rng = np.random.default_rng(3)
base = np.clip(np.linspace(0, 1, 192)[None, :, None] + 0.1 * rng.standard_normal((128, 192, 3)), 0, 1).astype(np.float32)
frames = np.stack([base] * 30)
depth = np.linspace(1, 0, 128)[None, :, None].repeat(192, 2).astype(np.float32)

# власне зерно
og = dict(og_amount=0.8)
pr = P.resolve("ARRI Alexa 35", 0.5, og)
check("власне зерно йде поверх профілю, сила профілю його не множить", pr["og_amount"] == 0.8 and pr["noise_luma"] > 0)
check("вибір профілю не містить ключів власного зерна", not any(k in P.profile_values("iPhone") for k in P.OWN))
rp = P.resolve("ARRI Alexa 35", 1.0, dict(og_amount=0.5, og_replace=1))
check("og_replace вимикає шум профілю", rp["noise_luma"] == 0 and rp["film_grain"] == 0 and rp["noise_chroma"] == 0)
cu = P.resolve("Custom", 1.0, {k: 0.0 for k in P.STRENGTH_KEYS} | {"og_amount": 1.0})
o = apply_realism(frames[:3], None, cu, seed=1)
d01 = float(np.abs(o[0] - o[1]).mean())
check("лише власне зерно: змінює кадр, без NaN, 0..1", float(np.abs(o - frames[:3]).mean()) > 1e-3 and np.isfinite(o).all() and o.min() >= 0 and o.max() <= 1)
st = apply_realism(frames[:3], None, dict(cu, og_animate=0.0), seed=1)
check("og_animate=1: зерно різне між кадрами; 0: однакове", d01 > 1e-3 and float(np.abs(st[0] - st[1]).max()) < 1e-6, "Δ=%.4f" % d01)
y = base @ np.array([0.2126, 0.7152, 0.0722], np.float32)
hi = apply_realism(frames[:1], None, dict(cu, og_shadows=0.0, og_mids=0.0, og_highlights=2.0), seed=1)[0] - frames[0]
check("відгук по тонах: лише світлі -> у тінях зерна нема", float(np.abs(hi[y < 0.15]).mean()) < 0.2 * float(np.abs(hi[y > 0.85]).mean()))
for k, v in [("og_size", 4.0), ("og_clump", 1.0), ("og_softness", 1.0), ("og_colour", 1.0), ("og_blue", 2.5)]:
    a = apply_realism(frames[:1], None, dict(cu, **{k: v}), seed=1)
    check("повзунок %s змінює зерно" % k, float(np.abs(a - o[:1]).mean()) > 1e-4)

# превʼю
meta = preview.store("7", frames, depth, 0)
check("store: шматок ≤24 кадри з середини", meta["n"] == 24 and meta["total"] == 30 and preview._CACHE["7"]["start"] == 3)
body = {"profile": "16mm film", "strength": 0.7, "seed": 2, "values": {}, "size": 160, "orig": True}
r = preview.render("7", body)
check("render: усі кадри + оригінали, ≤size", len(r["frames"]) == 24 and len(r["orig"]) == 24 and max(r["w"], r["h"]) <= 160, "%d ms" % r["ms"])
lp = preview.loupe("7", dict(body, x=0.9, y=0.1, crop=64, frame=20, count=8))
check("лупа: 4 кадри до кінця шматка, кроп у межах", len(lp["proc"]) == 4 and lp["w"] == 64 and lp["x0"] + 64 <= 192 and lp["y0"] == 0)
lp2 = preview.loupe("7", dict(body, x=0.2, y=0.6, crop=64, frame=20, count=8))
check("лупа: інша точка з кешу (ті ж параметри)", preview._LOUPE["7"][0] and lp2["x0"] != lp["x0"])
one = preview.loupe("7", dict(body, x=0.5, y=0.5, crop=64, frame=0, count=1))
full = apply_realism(preview._CACHE["7"]["frames"][:1].astype(np.float32) / 255, depth, P.resolve("16mm film", 0.7), seed=2, frame_offset=3)
check("лупа = той самий рушій, що й Run (frame_offset шматка)", np.array_equal(preview._LOUPE["7"][1][0], np.clip(full[0] * 255 + 0.5, 0, 255).astype(np.uint8)))
try:
    preview.render("99", body); check("нема кешу -> NeedRun", False)
except preview.NeedRun:
    check("нема кешу -> NeedRun", True)

# пресети
tmp = tempfile.mkdtemp()
presets._dir = lambda: tmp
n = presets.save_preset("  Моє   зерно ", {"og_amount": 3.0, "noise_luma": 0.2, "frame_offset": 9, "bogus": 1, "strength": 0.6, "og_replace": 0.7})
lst = presets.list_presets()
check("пресет: імʼя чиститься, межі SPEC, int, без frame_offset/сміття",
      n == "Моє зерно" and lst[n] == {"og_amount": 1.0, "noise_luma": 0.2, "strength": 0.6, "og_replace": 1})
for bad in ["../x", "iPhone", "", "a/b"]:
    try:
        presets.save_preset(bad, {"og_amount": 1}); check("погане імʼя %r відкинуто" % bad, False)
    except ValueError:
        check("погане імʼя %r відкинуто" % bad, True)
presets.delete_preset(n)
check("пресет видалено", presets.list_presets() == {})
shutil.rmtree(tmp)

print("\nРЕЗУЛЬТАТ:", "усе ок" if not fails else "ПРОВАЛИ: " + "; ".join(fails))
sys.exit(1 if fails else 0)
