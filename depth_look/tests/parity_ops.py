"""Паритет окремих cv2-операцій з їх torch-відповідниками (look_torch). Запуск:
  python depth_look/tests/parity_ops.py [cpu|cuda|mps]
Друкує max|diff| (в одиницях /255) для кожної операції. Це НЕ нода - Comfy цю теку не імпортує."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import numpy as np  # noqa: E402
import cv2  # noqa: E402
import torch  # noqa: E402
import look  # noqa: E402
import look_torch as T  # noqa: E402

dev = torch.device(sys.argv[1] if len(sys.argv) > 1 else "cpu")
rng = np.random.default_rng(1)
H, W = 180, 260
yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
base = (0.5 + 0.3 * np.sin(xx / 17.0) * np.cos(yy / 11.0)).astype(np.float32)
img = np.clip(np.stack([base, base[::-1], base[:, ::-1]], -1) + 0.15 * rng.random((H, W, 3), dtype=np.float32), 0, 1).astype(np.float32)
rows = []


def rec(name, ref, got):
    got = got.cpu().numpy() if hasattr(got, "cpu") else got
    d = np.abs(ref.astype(np.float64) - got.astype(np.float64))
    rows.append((name, d.max() * 255, d.mean() * 255))


def chw(a):
    return torch.from_numpy(np.ascontiguousarray(a.transpose(2, 0, 1) if a.ndim == 3 else a[None])).to(dev)


def hwc(t):
    t = t.cpu().numpy()
    return t.transpose(1, 2, 0) if t.shape[0] > 1 else t[0]


x = chw(img)
for s in (0.5, 1.0, 2.5, 5.0, 12.0, 23.0, 77.0):
    ref = cv2.GaussianBlur(img, (0, 0), s)
    rec("GaussianBlur sigma=%g" % s, ref, hwc(T.gaussian(x, s)))
for r in (1.0, 3.7, 9.0, 18.0):
    k = look._disk_kernel(r)
    ref = cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)
    rec("filter2D disk r=%g (REFLECT)" % r, ref, hwc(T.filter2d(x, k, "reflect")))
for r in (0.4, 2.2, 11.0, 18.0, 40.0, 96.0):
    ref = np.stack([look._disk_blur(img[..., c].copy(), r) for c in range(3)], -1)
    rec("_disk_blur r=%g" % r, ref, hwc(T._disk_blur(x, r)))
for (Wd, Hd) in ((130, 90), (87, 61), (52, 36), (260, 180), (400, 270), (600, 100)):
    for nm, fl in (("area", cv2.INTER_AREA), ("linear", cv2.INTER_LINEAR), ("cubic", cv2.INTER_CUBIC)):
        if nm == "area" and (Wd > W or Hd > H):
            continue        # INTER_AREA вгору в пайплайні не використовується (лише зменшення) - не відтворено
        ref = cv2.resize(img, (Wd, Hd), interpolation=fl)
        rec("resize %s %dx%d->%dx%d" % (nm, W, H, Wd, Hd), ref, hwc(T.resize(x, (Wd, Hd), nm)))
for sg in (11.0, 30.0, 64.0):
    ref = look._gblur(img, sg)
    rec("_gblur sigma=%g" % sg, ref, hwc(T._gblur(x, sg)))
rgba = np.dstack([img, base]).astype(np.float32)
x4 = chw(rgba)
for i, M in enumerate([np.float32([[0.8, 0, 30.3], [0, 0.8, 12.7]]), np.float32([[1, 0, 7.31], [0, 1, -3.9]]),
                       np.float32([[1.07, 0, -9.5], [0, 1.07, 4.2]]), np.float32([[0.55, 0, 100.123], [0, 0.55, 60.45]]),
                       np.float32([[1, 0, 0], [0, 1, 0]])]):
    ref = cv2.warpAffine(rgba, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    rec("warpAffine #%d (CONSTANT)" % i, ref, hwc(T.warp_affine(x4, M)))
gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
mx = (gx + 6.0 * np.sin(gy / 13.0)).astype(np.float32)
my = (gy + 5.0 * np.cos(gx / 9.0)).astype(np.float32)
ref = cv2.remap(img, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
rec("remap float maps (REFLECT)", ref, hwc(T.remap(x, torch.from_numpy(mx).to(dev), torch.from_numpy(my).to(dev), "reflect")))
rec("_chroma_ab 9px", look._chroma_ab(img, 9.0), hwc(T._chroma_ab(x, 9.0)))
rec("cvtColor RGB2GRAY", cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), T.gray(x))
st = [(0.0, (0.1, 0.0, 0.2)), (0.3, (0.9, 0.3, 0.1)), (0.7, (0.6, 0.6, 0.9)), (1.0, (0.4, 0.8, 1.0))]
rec("_gradient_map_pos", look._gradient_map_pos(base, st), hwc(T._gmap_pos(torch.from_numpy(base).to(dev), st, dev)))
w = np.clip(base - 0.3, 0, 1)
a16 = w.astype(np.float16).astype(np.float32)
rec("fp16 round trip", a16, torch.from_numpy(w).to(dev).half().float())
# grain: те саме зерно (CPU rng) - порівняння film_grain
for rough in (0.0, 35.0):
    ref = look.film_grain(img, 30.0, 40.0, rough, 11)
    nz = T.make_grain_noise(H, W, 30.0, 40.0, rough, 11)
    rec("film_grain rough=%g" % rough, ref, hwc(T._film_grain(x, 30.0, 40.0, rough, nz)))
print("%-44s %10s %10s" % ("op", "max/255", "mean/255"))
bad = 0
for n, mx_, me in rows:
    flag = "" if mx_ <= 1.0 else "   <-- > 1/255"
    bad += mx_ > 1.0
    print("%-44s %10.5f %10.6f%s" % (n, mx_, me, flag))
print("ops > 1/255: %d of %d" % (bad, len(rows)))
