"""Мікротест Camera Realism (numpy, ≤512 px, 2-4 кадри; ні Comfy, ні torch). Запуск:
   ~/.claude/tools/.venv/bin/python realism/check_micro.py
(а) усі сили 0 -> вихід == вхід побітово · (б) кожна група окремо міняє картинку, без NaN, у 0..1
(в) фіксований патерн однаковий між кадрами, шум різний · (г) depth=None не падає · профілі та майстер-сила."""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from realism import params as P
from realism.engine import apply_realism

H, W, B = 192, 288, 3
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = np.stack([xx / W, yy / H, 0.5 + 0.3 * np.sin(xx / 9) * np.sin(yy / 7)], -1)
base += 0.25 * ((xx.astype(int) // 6 + yy.astype(int) // 6) % 2)[..., None] * (xx > W / 2)[..., None]   # шахівниця: дрібні краї для байєра/блоків
frames = np.clip(np.stack([base] * B), 0, 1).astype(np.float32)
depth = np.clip(np.stack([(yy / H)] * B), 0, 1).astype(np.float32)
fails = []


def check(name, ok, info=""):
    print(("  ok  " if ok else "  FAIL") + " " + name + (" " + info if info else ""))
    if not ok:
        fails.append(name)


def run(**kw):
    p = {k: 0.0 for k in P.NAMES}
    p.update(P.NEUTRAL); p.update(kw)
    for k in P.STRENGTH_KEYS:
        p.setdefault(k, 0.0)
    return p


# (а) нуль
zero = {**P.NEUTRAL}
out = apply_realism(frames, depth, zero, seed=5)
check("(а) усі сили 0 -> вихід == вхід побітово", np.array_equal(out, frames) and out.dtype == np.float32)
out = apply_realism(frames, depth, P.resolve("iPhone", 0.0), seed=5)
check("(а) профіль iPhone при майстер-силі 0 -> без змін", np.array_equal(out, frames))

# (б) кожна група окремо
groups = {
    "noise_luma": dict(noise_luma=0.8), "noise_chroma": dict(noise_chroma=0.8), "fixed_pattern": dict(fixed_pattern=1.0),
    "bayer_amount": dict(bayer_amount=1.0), "isp_denoise": dict(isp_denoise=0.8), "isp_sharpen": dict(isp_sharpen=0.8),
    "isp_blocks": dict(isp_blocks=0.8), "vignette": dict(vignette=0.8), "distortion+": dict(distortion=0.8),
    "distortion-": dict(distortion=-0.8), "chroma_ab": dict(chroma_ab=1.0), "corner_softness": dict(corner_softness=0.8),
    "veil": dict(veil=0.8), "depth_haze": dict(depth_haze=0.8), "depth_detail": dict(depth_detail=0.8),
    "shake": dict(shake=0.8), "flicker": dict(flicker=1.0), "rolling_shutter(+shake)": dict(shake=0.5, rolling_shutter=1.0),
}
for name, kw in groups.items():
    t = time.time()
    o = apply_realism(frames, depth, run(**kw), seed=3)
    d = float(np.abs(o - frames).mean())
    check("(б) " + name + " змінює картинку", d > 1e-4, "mean|Δ|=%.5f" % d)
    check("(б) " + name + " без NaN і в 0..1", bool(np.isfinite(o).all()) and o.min() >= 0.0 and o.max() <= 1.0 and o.shape == frames.shape)
# rolling shutter відрізняється від чистого тремтіння
a = apply_realism(frames, depth, run(shake=0.5), seed=3)
b = apply_realism(frames, depth, run(shake=0.5, rolling_shutter=1.0), seed=3)
check("(б) rolling_shutter додає перекіс понад тремтіння", float(np.abs(a - b).mean()) > 1e-5)
# вхід не мутовано
f2 = frames.copy(); apply_realism(f2, depth, run(noise_luma=1, isp_blocks=1, bayer_amount=1, vignette=1), seed=1)
check("(б) вхідний масив не змінюється", np.array_equal(f2, frames))

# (в) фіксований патерн і шум
flat = np.full((B, H, W, 3), 0.4, np.float32)
fp = apply_realism(flat, None, run(fixed_pattern=1.0), seed=9)
check("(в) фікс. патерн: кадри ідентичні", np.array_equal(fp[0], fp[1]) and np.array_equal(fp[1], fp[2]) and not np.array_equal(fp[0], flat[0]))
fp2 = apply_realism(flat, None, run(fixed_pattern=1.0), seed=10)
check("(в) інший seed -> інший патерн", not np.array_equal(fp[0], fp2[0]))
nz = apply_realism(flat, None, run(noise_luma=0.8, noise_chroma=0.5), seed=9)
check("(в) шум: кадри різні", not np.array_equal(nz[0], nz[1]) and not np.array_equal(nz[1], nz[2]))
again = apply_realism(flat, None, run(noise_luma=0.8, noise_chroma=0.5), seed=9)
check("(в) шум детермінований від seed", np.array_equal(nz, again))
chunk = apply_realism(flat[1:3], None, run(noise_luma=0.8), seed=9, frame_offset=1)
check("(в) frame_offset: чанк дає ті ж кадри, що й повний батч", np.array_equal(chunk, apply_realism(flat, None, run(noise_luma=0.8), seed=9)[1:3]))
# тіні шумлять сильніше
dark = apply_realism(np.full((1, H, W, 3), 0.08, np.float32), None, run(noise_luma=1.0), seed=2)
mid = apply_realism(np.full((1, H, W, 3), 0.6, np.float32), None, run(noise_luma=1.0), seed=2)
check("(в) тіні шумлять сильніше за середні тони", float(dark.std()) > float(mid.std()), "std %.4f > %.4f" % (dark.std(), mid.std()))
# тремтіння: щокадру різне, між прогонами однакове
sh = apply_realism(frames, None, run(shake=0.8), seed=4)
check("(в) тремтіння: кадри відрізняються один від одного", not np.array_equal(sh[0], sh[1]))

# (г) depth=None
o = apply_realism(frames, None, run(depth_haze=1.0, depth_detail=1.0), seed=1)
check("(г) depth=None: група глибини пропускається без помилки", np.array_equal(o, frames))
o = apply_realism(frames, None, run(depth_haze=1.0, noise_luma=0.5), seed=1)
check("(г) depth=None + шум: працює", not np.array_equal(o, frames))
d1 = apply_realism(frames, depth[:1], run(depth_haze=0.8), seed=1)
check("(г) глибина 1 на батч (B=1 -> B кадрів) працює", d1.shape == frames.shape)
dh = np.stack([yy[::2, ::2] / (H / 2)] * B)
check("(г) глибина іншого розміру ресайзиться", apply_realism(frames, dh.astype(np.float32), run(depth_haze=0.8), seed=1).shape == frames.shape)
# одиночний кадр: відео-група пропускається
one = apply_realism(frames[:1], None, run(shake=1.0, flicker=1.0), seed=1)
check("одиночний кадр: тремтіння/мерехтіння пропущені", np.array_equal(one, frames[:1]))

# v2: підписи пристроїв і ядро - кожен окремо
for k, v in [("highlight_rolloff", 1.0), ("highlight_clip", 1.0), ("blacks", 1.0), ("blacks", -1.0), ("warmth", 1.0),
             ("saturation", -1.0), ("skin_hue", 1.0), ("film_grain", 1.0), ("halation", 1.0), ("gate_weave", 1.0),
             ("dust", 1.0), ("smart_hdr", 1.0), ("deep_fusion", 1.0), ("shadow_smear", 1.0), ("portrait_bokeh", 1.0),
             ("ghost_flare", 1.0), ("line_skip", 1.0), ("chroma_blotch", 1.0), ("banding", 1.0), ("motion_blur", 1.0)]:
    fr = frames
    if k == "motion_blur":                         # потрібен рух між кадрами
        fr = np.stack([np.roll(frames[0], 4 * j, axis=1) for j in range(B)])
    if k == "ghost_flare":                         # потрібні яскраві точки
        fr = fr.copy(); fr[:, 20:26, 30:36] = 1.0
    o = apply_realism(fr, depth, run(**{k: v}), seed=3)
    check("(v2) %s=%+.0f: без NaN, в 0..1, змінює" % (k, v),
          bool(np.isfinite(o).all()) and o.min() >= 0 and o.max() <= 1 and float(np.abs(o - fr).mean()) > 1e-4,
          "mean|Δ|=%.4f" % float(np.abs(o - fr).mean()))
o = apply_realism(frames[:1], depth[:1], run(motion_blur=1.0, gate_weave=1.0), seed=3)
check("(v2) один кадр: моушн-блюр і gate weave пропущені", np.array_equal(o, frames[:1]))
o = apply_realism(frames, None, run(portrait_bokeh=1.0), seed=3)
check("(v2) портретне боке без глибини пропускається", np.array_equal(o, frames))

# профілі
for name in P.PROFILES:
    t = time.time()
    o = apply_realism(frames, depth, P.resolve(name, 1.0), seed=7)
    check("профіль %s: без NaN, в 0..1, змінює" % name,
          bool(np.isfinite(o).all()) and o.min() >= 0 and o.max() <= 1 and float(np.abs(o - frames).mean()) > 1e-4,
          "mean|Δ|=%.4f %.2fs" % (float(np.abs(o - frames).mean()), time.time() - t))
c = P.resolve("Custom", 1.0, {"noise_luma": 0.4, "iso": 3200})
check("Custom бере значення повзунків", c["noise_luma"] == 0.4 and c["iso"] == 3200 and c["vignette"] == P.WIDGET_DEFAULTS["vignette"])
h = P.resolve("iPhone", 0.5)
check("майстер-сила множить сили, не iso/радіус", abs(h["isp_sharpen"] - 0.2) < 1e-9 and h["iso"] == P.PROFILES["iPhone"]["iso"] and h["isp_sharpen_radius"] == 1.4 and abs(h["blacks"] + 0.125) < 1e-9)

print("\nРЕЗУЛЬТАТ:", "усе ок" if not fails else "ПРОВАЛИ: " + "; ".join(fails))
sys.exit(1 if fails else 0)
