"""Швидкість torch-шляху: B кадрів 1080p через core.apply_numpy(device=...), прогрів 2 кадри окремо. НЕ нода.
  python depth_look/tests/bench_torch.py --device cuda --frames 60 --size 1920 --preset look_v1 [--lens-only] [--cpu-ref]"""
import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402
import core  # noqa: E402
import look_torch as T  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="cuda")
ap.add_argument("--frames", type=int, default=60)
ap.add_argument("--size", type=int, default=1920)
ap.add_argument("--preset", default="look_v1")
ap.add_argument("--cpu-ref", action="store_true", help="для порівняння: 1 кадр numpy, workers=1")
ap.add_argument("--profile", action="store_true", help="torch profiler по одному кадру (топ операцій)")
args = ap.parse_args()

# synth() з parity_torch без запуску його main
src = open(os.path.join(HERE, "parity_torch.py")).read()
ns = {}
exec(src[src.index("def synth"):src.index("def preset_kw")], {"np": np}, ns)
synth = ns["synth"]
W = args.size
H = int(round(W * 9 / 16))
base = [synth(W, H, i) for i in range(8)]
frames = np.stack([base[i % 8][0] for i in range(args.frames)])
depths = np.stack([base[i % 8][1] for i in range(args.frames)])
print("device=%s %dx%d frames=%d preset=%s" % (args.device, W, H, args.frames, args.preset))
if args.cpu_ref:
    t0 = time.time()
    core.apply_numpy(frames[:1], depths[:1], preset_name=args.preset, workers=1)
    print("numpy CPU workers=1: %.2f s/frame" % (time.time() - t0))
core.apply_numpy(frames[:2], depths[:2], preset_name=args.preset, device=args.device)      # прогрів
if args.device.startswith("cuda"):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
if args.profile:
    from torch.profiler import profile, ProfilerActivity
    kw = core.look_params.kwargs(core.build_preset(args.preset))
    img, d = base[0]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as pr:
        T.depth_look_lens(img, d, device=args.device, **kw)
        torch.cuda.synchronize()
    print(pr.key_averages().table(sort_by="cuda_time_total", row_limit=18, max_name_column_width=40))
t0 = time.time()
out = core.apply_numpy(frames, depths, preset_name=args.preset, device=args.device)
if args.device.startswith("cuda"):
    torch.cuda.synchronize()
dt = time.time() - t0
print("TOTAL %d frames: %.2f s  = %.3f s/frame   peak VRAM %.2f GB" % (
    args.frames, dt, dt / args.frames, (torch.cuda.max_memory_allocated() / 1e9) if args.device.startswith("cuda") else 0))
