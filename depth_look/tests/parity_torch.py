"""Паритет look_torch.depth_look_lens з look.depth_look_lens (numpy/cv2) на синтетичному кадрі + синтетичній глибині.
Запуск (НЕ нода; Comfy цю теку не імпортує):
  python depth_look/tests/parity_torch.py [--device cpu|cuda|mps] [--size 512] [--frames 2] [--e2e] [--sets look_v1,00_original,...]
Для кожного набору параметрів друкує max/mean |diff| у одиницях /255 і с/кадр (numpy vs torch). Без приватних медіа."""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ENGINE = os.path.dirname(HERE)
sys.path.insert(0, ENGINE)
import numpy as np  # noqa: E402
import look  # noqa: E402
import look_params  # noqa: E402
import core  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cpu")
ap.add_argument("--size", type=int, default=512, help="довга сторона, 16:9")
ap.add_argument("--frames", type=int, default=2)
ap.add_argument("--e2e", action="store_true", help="додатково через core.apply_numpy(device=...)")
ap.add_argument("--sets", default="")
ap.add_argument("--skip-cpu-ref", action="store_true")
args = ap.parse_args()
import look_torch as T  # noqa: E402
import torch  # noqa: E402


def synth(W, H, i=0):
    """Процедурний кадр і глибина (світле = близьке): гладкий фон зі смугами/колами + «предмет» і ящик."""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = xx / W, yy / H
    sh = 0.02 * i
    r = 0.5 + 0.4 * np.sin(6.0 * (u + sh) + 2 * v)
    g = 0.45 + 0.4 * np.sin(9.0 * v - 3 * u + 1.0)
    b = 0.5 + 0.4 * np.cos(5.0 * (u - sh) + 7 * v)
    img = np.stack([r, g, b], -1)
    for cx, cy, rad, col in ((0.62 + sh, 0.55, 0.16, (0.95, 0.8, 0.7)), (0.25, 0.3, 0.07, (0.2, 0.3, 0.9))):
        m = np.clip((rad - np.hypot((u - cx) * W / H, v - cy)) / 0.01, 0, 1)[..., None]
        img = img * (1 - m) + np.array(col, np.float32) * m
    stripes = (np.sin(xx * 0.9) > 0.6).astype(np.float32)[..., None] * 0.1
    img = np.clip(img + stripes, 0, 1).astype(np.float32)
    d = 0.12 + 0.45 * v                                                    # земля ближче внизу
    subj = np.clip((0.2 - np.hypot((u - 0.62 - sh) * W / H, (v - 0.55) * 1.2)) / 0.06, 0, 1)
    box = ((u > 0.1) & (u < 0.3) & (v > 0.5) & (v < 0.85)).astype(np.float32) * 0.65
    d = np.maximum(np.maximum(d, subj * 0.95), box * np.clip(v, 0.5, 1))
    d = np.clip(d + 0.02 * np.sin(xx * 0.05), 0, 1).astype(np.float32)
    return img, d


def preset_kw(name):
    return look_params.kwargs(core.build_preset(name))


def sets():
    s = {"look_v1": preset_kw("look_v1"), "00_original": preset_kw("00_original")}
    eff = preset_kw("look_v1")
    eff.update(grain_amount=45.0, grain_size=30.0, grain_roughness=40.0, glow=0.5, glow_radius=0.03, glow_threshold=0.5,
               glow_warmth=0.6, haze=0.45, haze_start=0.2, haze_curve=1.3, haze_t=0.6, haze_scatter=0.6, haze_drift=0.3,
               halation_strength=0.8, halation_radius=0.012, chroma_ab=6.0, grain_animate=1, glow_pulse=0.2, warp_speed=0.5)
    s["effects(grain+glow+haze)"] = eff
    base = preset_kw("00_original")
    gm = dict(base)
    gm.update(far_gradient=0.8, near_tint=0.5, bg_bands=0.9, subject_map=0.5, r_min=0.002, r_max=0.04, blur_amount=1.0,
              grad_mode=4, grad_amount=1.0, grad_speed=0.3, grain_amount=0, halation_strength=0.4)
    s["grad_mode=4 + bands"] = gm
    g5 = dict(gm)
    g5.update(grad_mode=5, grad_amp=0.2)
    s["grad_mode=5"] = g5
    g1 = dict(gm)
    g1.update(grad_mode=1, grad_offset_y=0.05)
    s["grad_mode=1"] = g1
    sil = preset_kw("look_v1")
    sil.update(silhouette=0.7, sil_bg=1.0, sil_rim=0.6, stretch_lighten=0.4, blur_target=0.4, glow=0.3, smear_radial=0.0, grain_amount=20.0)
    s["silhouette+blur_target"] = sil
    fb = preset_kw("look_v1")
    fb.update(trail=0.6, trail_source=0.5, spark=0.5, spark_rays=4, grain_amount=15.0)
    s["trail+spark (numpy fallback)"] = fb
    return s


def run_numpy(img, d, kw):
    return look.depth_look_lens(img, d, **kw)


def main():
    W = args.size
    H = int(round(W * 9 / 16))
    want = [x for x in args.sets.split(",") if x]
    S = sets()
    print("device=%s size=%dx%d frames=%d torch=%s" % (args.device, W, H, args.frames, torch.__version__))
    print("%-34s %9s %9s %10s %10s" % ("set", "max/255", "mean/255", "numpy s/f", "torch s/f"))
    worst = 0.0
    for name, kw0 in S.items():
        if want and name not in want:
            continue
        mx = mean = 0.0
        tn = tt = 0.0
        for i in range(args.frames):
            img, d = synth(W, H, i)
            kw = dict(kw0)
            kw["time"] = i / 24.0
            kw["seed"] = int(kw0.get("seed", 7)) + i
            t0 = time.time()
            ref = run_numpy(img, d, kw)
            tn += time.time() - t0
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            t0 = time.time()
            got = T.depth_look_lens(img, d, device=args.device, **kw)
            tt += time.time() - t0
            df = np.abs(ref.astype(np.float64) - got.astype(np.float64))
            mx = max(mx, df.max() * 255)
            mean += df.mean() * 255 / args.frames
        worst = max(worst, mx)
        print("%-34s %9.4f %9.5f %10.2f %10.2f%s" % (name, mx, mean, tn / args.frames, tt / args.frames, "" if mx <= 1.0 else "  <-- > 1/255"))
    print("fallbacks used:", sorted(T.FALLBACKS_USED) or "none")
    if args.e2e:
        e2e(W, H)
    print("worst max diff: %.4f /255" % worst)


def e2e(W, H):
    """core.apply_numpy: CPU (дефолт) проти device=...; глибина йде через prepare_frame_depth (clean_depth), 2 кадри."""
    B = max(2, args.frames)
    fr, dp = zip(*[synth(W, H, i) for i in range(B)])
    frames, depths = np.stack(fr), np.stack(dp)
    for preset in ("look_v1", "00_original"):
        t0 = time.time()
        a = core.apply_numpy(frames, depths, preset_name=preset, workers=1)
        ta = time.time() - t0
        t0 = time.time()
        b = core.apply_numpy(frames, depths, preset_name=preset, device=args.device)
        tb = time.time() - t0
        df = np.abs(a.astype(np.float64) - b.astype(np.float64))
        print("e2e apply_numpy %-12s max %.4f /255  mean %.5f  cpu %.2fs/f  %s %.2fs/f" % (preset, df.max() * 255, df.mean() * 255, ta / B, args.device, tb / B))


main()
