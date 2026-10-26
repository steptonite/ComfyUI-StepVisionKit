"""depth_look.look_torch - torch (CUDA / CPU / MPS) порт look.depth_look_lens. Етап 1 GPU-порту (25.10.2026).

Ті самі параметри й семантика, що в look.depth_look_lens (сигнатура береться з нього - списки не розходяться).
Мета: піксель у піксель з numpy/cv2-шляхом (|різниця| <= 1/255, на практиці ~1e-6), але на GPU.
Тому cv2-операції відтворені ТОЧНО, а не "приблизно":
  * GaussianBlur   - те саме ядро (ksize = round(8*sigma+1)|1, нормування в double), межа REFLECT_101;
  * filter2D       - кореляція, межа REFLECT (cv2 "fedcba|abcdefgh|hgfedcb" = torch "symmetric", не torch "reflect");
  * resize         - матриці INTER_AREA / LINEAR / CUBIC за кодом OpenCV (дробові ваги area, A=-0.75 cubic, replicate-межа);
  * warpAffine     - фіксована крапка OpenCV: обернена матриця в double, AB_BITS=10, координати квантуються до 1/32 px;
  * remap (float)  - теж квантування до 1/32 px + таблична білінійна вага, межі CONSTANT / REFLECT;
  * шари fp16      - округлення в half, як `.astype(np.float16)` у prepare_lens.
Зерно: випадкові числа беруться з numpy (default_rng(seed).standard_normal) на CPU - щоб піксель збігався; далі все на пристрої.
(GPU-генератор дав би інше зерно з тією ж статистикою - це окремий крок, див. docs/gpu_port_progress.md.)

Лишилось numpy-фолбеком (цілий підкрок, round-trip через CPU): _light_trail (trail>0), _sparks (spark>0).
Перелік реально викликаних фолбеків: look_torch.FALLBACKS_USED.

Тензори всередині: (C,H,W) float32. Публічно: depth_look_lens(img HxWx3 np, depth HxW np, device=..., **kw) -> HxWx3 np."""
import inspect
import math
import threading

import numpy as np
import torch
import torch.nn.functional as F

import look as _L

FALLBACKS_USED = set()          # які підкроки йшли через numpy (для звіту)
_CACHE = {}
_LOCK = threading.Lock()
_DEFAULTS = None


def _cached(key, build):
    v = _CACHE.get(key)
    if v is None:
        v = build()
        with _LOCK:
            if len(_CACHE) > 300:
                _CACHE.clear()
            _CACHE[key] = v
    return v


def _tt(a, dev, dtype=torch.float32):
    return torch.as_tensor(np.ascontiguousarray(a), dtype=dtype).to(dev)


# =============================================================================== межі (border) та індекси
def _border_index(n, j, mode):
    """numpy: індекси j (можуть виходити за [0,n)) -> у [0,n) за правилом межі OpenCV."""
    if mode == "reflect101":
        if n == 1:
            return np.zeros_like(j)
        per = 2 * (n - 1)
        j = np.mod(j, per)
        return np.where(j > n - 1, per - j, j)
    if mode == "reflect":                      # cv2.BORDER_REFLECT: fedcba|abcdefgh|hgfedcb
        per = 2 * n
        j = np.mod(j, per)
        return np.where(j >= n, per - 1 - j, j)
    return np.clip(j, 0, n - 1)                # replicate


def _border_index_t(n, j, mode):
    """те саме для int64-тензора j."""
    if mode == "reflect101":
        if n == 1:
            return torch.zeros_like(j)
        per = 2 * (n - 1)
        j = torch.remainder(j, per)
        return torch.where(j > n - 1, per - j, j)
    if mode == "reflect":
        per = 2 * n
        j = torch.remainder(j, per)
        return torch.where(j >= n, per - 1 - j, j)
    return j.clamp(0, n - 1)


def _pad_idx(n, p, mode, dev):
    return _cached(("pad", n, p, mode, str(dev)),
                   lambda: torch.as_tensor(_border_index(n, np.arange(-p, n + p), mode), dtype=torch.long).to(dev))


# =============================================================================== Gaussian / filter2D
def gauss_kernel(sigma):
    """cv2.getGaussianKernel для CV_32F зі sigma > 0: ksize = round(sigma*4*2+1)|1."""
    n = int(round(sigma * 4 * 2 + 1)) | 1
    xs = np.arange(n, dtype=np.float64) - (n - 1) * 0.5
    k = np.exp((-0.5 / (sigma * sigma)) * xs * xs)
    k = k * (1.0 / k.sum())
    return k.astype(np.float32)


def _blur_mat(n, k, mode, dev):
    """Щільна (n,n) матриця одновимірного фільтра з межею: out = M @ x. Для великих ядер матмул швидший за згортку."""
    key = ("bmat", n, k.tobytes(), mode, str(dev))

    def build():
        K = len(k)
        r = K // 2
        idx = _border_index(n, np.arange(-r, n + r), mode)
        rows = np.repeat(np.arange(n), K)
        cols = idx[np.arange(n)[:, None] + np.arange(K)[None, :]].ravel()
        w = np.tile(k.astype(np.float64), n)
        m = np.bincount(rows * n + cols, weights=w, minlength=n * n).reshape(n, n)
        return _tt(m, dev)
    return _cached(key, build)


_DENSE_K = 24       # ядро довше за це -> матмул


def _axis_filter(x, axis, k, mode):
    """Одновимірна кореляція ядром k уздовж axis (1 = y, 2 = x) тензора (C,H,W)."""
    C = x.shape[0]
    n = x.shape[axis]
    K = len(k)
    if K == 1:
        return x * float(k[0])
    if K > _DENSE_K:
        M = _blur_mat(n, k, mode, x.device)
        return torch.matmul(x, M.t()) if axis == 2 else torch.matmul(M, x)
    r = K // 2
    xp = x.index_select(axis, _pad_idx(n, r, mode, x.device))
    kt = _cached(("k1", k.tobytes(), str(x.device)), lambda: _tt(k, x.device))
    if axis == 2:
        return F.conv2d(xp.unsqueeze(1), kt.view(1, 1, 1, K)).squeeze(1)
    return F.conv2d(xp.unsqueeze(1), kt.view(1, 1, K, 1)).squeeze(1)


def sep_filter(x, kx, ky, mode):
    return _axis_filter(_axis_filter(x, 2, kx, mode), 1, ky, mode)


def gaussian(x, sigma, mode="reflect101"):
    """cv2.GaussianBlur(x, (0,0), sigma) для (C,H,W)."""
    k = gauss_kernel(sigma)
    return sep_filter(x, k, k, mode)


def filter2d(x, k2, mode="reflect"):
    """cv2.filter2D(x, -1, k2, borderType=...) (кореляція, anchor = центр, непарне ядро) для (C,H,W)."""
    C, H, W = x.shape
    kh, kw = k2.shape
    xp = x.index_select(1, _pad_idx(H, kh // 2, mode, x.device)).index_select(2, _pad_idx(W, kw // 2, mode, x.device))
    kt = _cached(("k2", k2.tobytes(), k2.shape, str(x.device)), lambda: _tt(k2, x.device))
    return F.conv2d(xp.unsqueeze(1), kt.view(1, 1, kh, kw)).squeeze(1)


# =============================================================================== resize
def _axis_resize_mat(src, dst, interp):
    scale = src / dst
    M = np.zeros((dst, src), np.float64)
    if interp == "area":
        for dx in range(dst):
            fsx1 = dx * scale
            fsx2 = fsx1 + scale
            cell = min(scale, src - fsx1)
            sx1 = math.ceil(fsx1)
            sx2 = math.floor(fsx2)
            sx2 = min(sx2, src - 1)
            sx1 = min(sx1, sx2)
            if sx1 - fsx1 > 1e-3:
                M[dx, sx1 - 1] += (sx1 - fsx1) / cell
            for sx in range(sx1, sx2):
                M[dx, sx] += 1.0 / cell
            if fsx2 - sx2 > 1e-3:
                M[dx, sx2] += min(min(fsx2 - sx2, 1.0), cell) / cell
        return M
    fx = (np.arange(dst) + 0.5) * scale - 0.5
    sx = np.floor(fx).astype(np.int64)
    f = fx - sx
    d = np.arange(dst)
    if interp == "linear":
        np.add.at(M, (d, np.clip(sx, 0, src - 1)), 1.0 - f)
        np.add.at(M, (d, np.clip(sx + 1, 0, src - 1)), f)
        return M
    A = -0.75                                       # cubic
    c0 = ((A * (f + 1) - 5 * A) * (f + 1) + 8 * A) * (f + 1) - 4 * A
    c1 = ((A + 2) * f - (A + 3)) * f * f + 1
    c2 = ((A + 2) * (1 - f) - (A + 3)) * (1 - f) * (1 - f) + 1
    c3 = 1 - c0 - c1 - c2
    for t, c in enumerate((c0, c1, c2, c3)):
        np.add.at(M, (d, np.clip(sx - 1 + t, 0, src - 1)), c)
    return M


def _resize_mats(H, W, Hd, Wd, interp, dev):
    def build():
        return (_tt(_axis_resize_mat(H, Hd, interp), dev), _tt(_axis_resize_mat(W, Wd, interp), dev).t().contiguous())
    return _cached(("rs", H, W, Hd, Wd, interp, str(dev)), build)


def resize(x, size, interp):
    """cv2.resize(x, (Wd,Hd), interpolation=...) для (C,H,W); interp: 'area' | 'linear' | 'cubic'."""
    C, H, W = x.shape
    Wd, Hd = size
    if (W, H) == (Wd, Hd):
        return x
    sx, sy = W / Wd, H / Hd
    if interp == "linear" and abs(sx - round(sx)) < 2.3e-16 and abs(sy - round(sy)) < 2.3e-16 and round(sx) == 2 and round(sy) == 2:
        interp = "area"                             # cv2: LINEAR на рівно 2x вниз = AREA
    if interp == "area" and not (sx >= 1 and sy >= 1):
        interp = "linear"                           # апскейл через AREA в пайплайні не зустрічається (лише вниз); не відтворено
    My, MxT = _resize_mats(H, W, Hd, Wd, interp, x.device)
    return torch.matmul(torch.matmul(My, x), MxT)


# =============================================================================== bilinear fixed-point (warpAffine / remap)
def _sample_fixed(x, X, Y, border):
    """x (C,H,W); X,Y - int-тензори (Hd,Wd) у 1/32 px (INTER_BITS=5). Білінійна вага з таблиці OpenCV (k/32)."""
    C, H, W = x.shape
    Hd, Wd = X.shape
    X = X.long()
    Y = Y.long()
    ix = torch.div(X, 32, rounding_mode="floor")
    iy = torch.div(Y, 32, rounding_mode="floor")
    fx = (X - ix * 32).to(torch.float32) / 32.0
    fy = (Y - iy * 32).to(torch.float32) / 32.0
    xf = x.reshape(C, H * W)
    acc = None
    for dy, wy in ((0, 1.0 - fy), (1, fy)):
        for dx, wx in ((0, 1.0 - fx), (1, fx)):
            sx = ix + dx
            sy = iy + dy
            if border == "const":
                valid = ((sx >= 0) & (sx < W) & (sy >= 0) & (sy < H)).to(torch.float32)
                idx = sy.clamp(0, H - 1) * W + sx.clamp(0, W - 1)
                v = xf.index_select(1, idx.reshape(-1)).reshape(C, Hd, Wd) * valid
            else:
                idx = _border_index_t(H, sy, border) * W + _border_index_t(W, sx, border)
                v = xf.index_select(1, idx.reshape(-1)).reshape(C, Hd, Wd)
            t = v * (wy * wx)
            acc = t if acc is None else acc + t
    return acc


def warp_affine(x, M, border="const"):
    """cv2.warpAffine(x, M, (W,H)) з INTER_LINEAR (M - пряма матриця 2x3, як у cv2; float32 -> double). x (C,H,W)."""
    C, H, W = x.shape
    m = np.asarray(M, np.float64).reshape(2, 3)
    D = m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]
    D = 1.0 / D if D != 0 else 0.0
    A11, A22, A12, A21 = m[1, 1] * D, m[0, 0] * D, -m[0, 1] * D, -m[1, 0] * D
    b1 = -A11 * m[0, 2] - A12 * m[1, 2]
    b2 = -A21 * m[0, 2] - A22 * m[1, 2]
    AB = 1024.0
    xs = np.arange(W, dtype=np.float64)
    ys = np.arange(H, dtype=np.float64)
    ad = np.rint(A11 * xs * AB).astype(np.int64)
    bd = np.rint(A21 * xs * AB).astype(np.int64)
    X0 = np.rint((A12 * ys + b1) * AB).astype(np.int64) + 16
    Y0 = np.rint((A22 * ys + b2) * AB).astype(np.int64) + 16
    dev = x.device
    X = torch.div(_tt(X0, dev, torch.long)[:, None] + _tt(ad, dev, torch.long)[None, :], 32, rounding_mode="floor")
    Y = torch.div(_tt(Y0, dev, torch.long)[:, None] + _tt(bd, dev, torch.long)[None, :], 32, rounding_mode="floor")
    return _sample_fixed(x, X, Y, border)


def remap(x, mapx, mapy, border="reflect"):
    """cv2.remap(x, mapx, mapy, INTER_LINEAR, borderMode=...) з float-картами (квантування до 1/32 px, як в OpenCV)."""
    X = torch.round(mapx * 32.0).to(torch.long)
    Y = torch.round(mapy * 32.0).to(torch.long)
    return _sample_fixed(x, X, Y, border)


# =============================================================================== дрібні поточкові хелпери
def gray(x):
    """cv2.COLOR_RGB2GRAY для (3,H,W) -> (H,W)."""
    return x[0] * 0.299 + x[1] * 0.587 + x[2] * 0.114


def _sm_s(x, a, b):
    t = float(np.clip((x - a) / max(b - a, 1e-6), 0, 1))
    return t * t * (3 - 2 * t)


def _sm(x, a, b):
    t = ((x - a) / max(b - a, 1e-6)).clamp(0, 1)
    return t * t * (3 - 2 * t)


def _screen(a, b):
    return 1 - (1 - a) * (1 - b.clamp(0, 1))


def _hex(c):
    return _L._hex(c)


def _interp1(x, xp, fp, dev):
    """np.interp(x, xp, fp) для тензора x; xp (n,) зростаючий, fp (n,) або (n,3) -> (3,)+x.shape / x.shape."""
    xp = np.asarray(xp, np.float32)
    fp = np.asarray(fp, np.float32)
    n = len(xp)
    three = fp.ndim == 2
    if n == 1:
        if three:
            return torch.stack([torch.full_like(x, float(fp[0, c])) for c in range(3)])
        return torch.full_like(x, float(fp[0]))

    def build():
        dx = np.maximum(np.diff(xp.astype(np.float64)), 1e-30)
        sl = np.diff(fp.astype(np.float64), axis=0) / (dx[:, None] if three else dx)
        return (_tt(xp, dev), _tt(fp, dev), _tt(sl, dev))
    xt, ft, sl = _cached(("interp", xp.tobytes(), fp.tobytes(), str(dev)), build)
    x = x.contiguous()
    idx = (torch.searchsorted(xt, x, right=True) - 1).clamp(0, n - 2)
    base = xt[idx]
    lo = x <= xt[0]
    hi = x >= xt[-1]
    if three:
        outs = []
        for c in range(3):
            v = sl[:, c][idx] * (x - base) + ft[:, c][idx]
            v = torch.where(lo, ft[0, c], torch.where(hi, ft[-1, c], v))
            outs.append(v)
        return torch.stack(outs)
    v = sl[idx] * (x - base) + ft[idx]
    return torch.where(lo, ft[0], torch.where(hi, ft[-1], v))


def _gmap_pos(t, stops, dev):
    """look._gradient_map_pos для тензора t (будь-яка форма) -> (3,)+t.shape."""
    xs = [s[0] for s in stops]
    S = [_hex(s[1]) for s in stops]
    return _interp1(t.clamp(0, 1), xs, S, dev)


def _gmap_scalar(t, stops):
    """Кольір палітри в одній точці -> numpy (3,) float32 (дрібна CPU-річ: ідентично look)."""
    return _L._gradient_map_pos(np.full((1, 1), t, np.float32), stops)[0, 0]


def _percentile(t, qs):
    t = t.reshape(-1)
    if t.numel() <= 16_000_000 and t.device.type != "mps":
        q = torch.tensor([v / 100.0 for v in qs], dtype=torch.float32, device=t.device)
        return [float(v) for v in torch.quantile(t, q)]
    a = t.detach().cpu().numpy()
    return [float(np.percentile(a, v)) for v in qs]


# =============================================================================== розмиття
def _gblur(x, sigma):
    """look._gblur: великі sigma - на зменшеній копії."""
    if sigma < 0.5:
        return x
    if sigma > 10:
        f = sigma / 5.0
        C, H, W = x.shape
        sm = resize(x, (max(2, int(round(W / f))), max(2, int(round(H / f)))), "area")
        sm = gaussian(sm, 5.0)
        return resize(sm, (W, H), "linear")
    return gaussian(x, sigma)


def _disk_k(r):
    return _cached(("disk", r), lambda: _L._disk_kernel(r))


def _disk_blur(x, r):
    """look._disk_blur для (C,H,W) (усі канали з однаковим r)."""
    if r < 0.6:
        return x
    C, H, W = x.shape
    if r > 18:
        f = r / 18.0
        sm = resize(x, (max(2, int(W / f)), max(2, int(H / f))), "area")
        sm = filter2d(sm, _disk_k(18.0), "reflect")
        return resize(sm, (W, H), "linear")
    return filter2d(x, _disk_k(r), "reflect")


# =============================================================================== зерно
def _grain_seed(kw):
    seed = int(kw.get("seed", 7))
    if kw.get("grain_animate"):
        seed = seed + int(np.floor(float(kw.get("time", 0.0)) * kw.get("grain_fps", 24.0)))
    return seed


def make_grain_noise(H, W, amount, size, roughness, seed, color=0.25, ref_width=2848):
    """CPU-частина film_grain: випадкові масиви в ТОМУ Ж порядку звернень до rng, що й look.film_grain.
    Безпечно викликати з потоку (numpy RNG відпускає GIL). Повертає dict або None (amount <= 0)."""
    if amount <= 0:
        return None
    rng = np.random.default_rng(seed)
    cell = (0.8 + size / 100.0 * 2.4) * (min(H, W) / ref_width)
    cell = max(cell, 0.6)
    vis_w = float(np.clip((2144.0 - max(H, W)) / (2144.0 - 720.0), 0.0, 1.0))
    if vis_w > 0:
        cell = max(cell, vis_w * (1.0 + size / 100.0 * 1.4))

    def shape(c):
        return max(2, int(round(H / c))), max(2, int(round(W / c)))

    r = np.clip(roughness / 100.0, 0, 1)
    out = {"base": rng.standard_normal(shape(cell)).astype(np.float32), "fine": None, "chroma": None}
    if r > 0:
        out["fine"] = rng.standard_normal(shape(cell * 0.5)).astype(np.float32)
    else:
        rng.standard_normal(shape(cell * 0.5))
    if color > 0:
        h, w = shape(cell)
        out["chroma"] = rng.standard_normal((h, w, 3)).astype(np.float32)
    return out


def make_noise_for(H, W, kw):
    """Шум для кадру за kwargs depth_look_lens (з урахуванням grain_animate). None, якщо зерна нема."""
    ga = kw.get("grain_amount", 27.0)
    if not (ga > 0):
        return None
    return make_grain_noise(H, W, ga, kw.get("grain_size", 54.0), kw.get("grain_roughness", 35.0), _grain_seed(kw))


def _film_grain(img, amount, size, roughness, noise, color=0.25):
    """look.film_grain на пристрої; img (3,H,W), noise = make_grain_noise(...)."""
    img = img.clamp(0, 1)
    if amount <= 0:
        return img
    _, H, W = img.shape
    dev = img.device
    r = float(np.clip(roughness / 100.0, 0, 1))
    base = resize(_tt(noise["base"], dev).unsqueeze(0), (W, H), "cubic")[0]
    if r > 0:
        fine = resize(_tt(noise["fine"], dev).unsqueeze(0), (W, H), "cubic")[0]
        n = (1 - r) * base + r * (0.55 * base + 0.85 * fine)
    else:
        n = (1 - r) * base
    n = n / (n.std(unbiased=False) + 1e-6)
    if r > 0:
        n = torch.sign(n) * n.abs() ** (1.0 - 0.25 * r)
        n = n / (n.std(unbiased=False) + 1e-6)
    y = gray(img)
    wgt = 0.22 + 0.78 * (4 * y * (1 - y)).clamp(0, 1) ** 0.8
    k = amount / 100.0 * 0.22
    if color > 0:
        chroma = resize(_tt(noise["chroma"], dev).permute(2, 0, 1), (W, H), "cubic")
        chroma = chroma / (chroma.std(unbiased=False) + 1e-6)
        nz = (n[None] * (1 - color) + chroma * color * 0.8) * k
    else:
        nz = n[None] * k
    return (img + nz * wgt[None]).clamp(0, 1)


# =============================================================================== окремі ефекти
def _chroma_ab(img, px):
    if abs(px) < 1e-3:
        return img
    _, H, W = img.shape
    dev = img.device
    cx, cy = (W - 1) / 2, (H - 1) / 2
    k = px / np.hypot(cx, cy)
    yy = torch.arange(H, device=dev, dtype=torch.float32)[:, None].expand(H, W)
    xx = torch.arange(W, device=dev, dtype=torch.float32)[None, :].expand(H, W)
    out = img.clone()
    for ch, sgn in ((0, 1.0), (2, -1.0)):
        s = 1.0 + sgn * k
        mx = cx + (xx - cx) / s
        my = cy + (yy - cy) / s
        out[ch] = remap(img[ch:ch + 1], mx, my, "reflect")[0]
    return out


def _warp_fields_t(seed, gs, ww, hh, phase, dev):
    """look._warp_fields: випадкові (gs,gs) на CPU, збільшення cubic - на пристрої."""
    def one(sd):
        rs = np.random.default_rng(int(sd))
        a = rs.standard_normal((gs, gs)).astype(np.float32)
        b = rs.standard_normal((gs, gs)).astype(np.float32)
        t = resize(_tt(np.stack([a, b]), dev), (ww, hh), "cubic")
        return t[0], t[1]
    k = int(np.floor(phase))
    fr = phase - k
    ax, ay = one(seed + k)
    if fr <= 0:
        return ax, ay
    f = fr * fr * (3 - 2 * fr)
    bx, by = one(seed + k + 1)
    nrm = 1.0 / np.sqrt((1 - f) ** 2 + f ** 2)
    return ((1 - f) * ax + f * bx) * nrm, ((1 - f) * ay + f * by) * nrm


def _near_map(d, layers, rmaxpx, onset, dev):
    cs, nd = [], []
    for L in layers:
        if L is None:
            continue
        v = float(1 - _sm_s(L["r"] / rmaxpx, onset, 0.55))
        cs.append(L["c"])
        nd.append(v if v > 0.05 else 0.0)
    if not cs or max(nd) <= 0.0:
        return None
    return _interp1(d, np.asarray(cs, np.float32), np.asarray(nd, np.float32), dev)


def _warp_map(d, nmap, warp, warp_scale, seed, phase):
    H, W = d.shape
    gs = max(2, int(round(1 / max(warp_scale, 0.02))))
    amp = warp * nmap * (max(H, W) * 0.04)
    fx, fy = _warp_fields_t(int(seed), gs, W, H, phase, d.device)
    gy = torch.arange(H, device=d.device, dtype=torch.float32)[:, None].expand(H, W)
    gx = torch.arange(W, device=d.device, dtype=torch.float32)[None, :].expand(H, W)
    return gx + fx * amp, gy + fy * amp


def _remap_planes(x, mp):
    return remap(x, mp[0], mp[1], "reflect")


def _smear_planes(planes, ln, ang, radial, cx, cy):
    """12 зсувів/зумів, вагові sum (спільна частина layer-smear і _smear_depth). planes (C,H,W)."""
    C, hh, ww = planes.shape
    acc = torch.zeros_like(planes)
    wsum = 0.0
    for k in range(12):
        t = k / 11.0
        if radial:
            sc = 1 + (ln / hh) * 2.0 * t
            M = np.float32([[sc, 0, cx * (1 - sc)], [0, sc, cy * (1 - sc)]])
        else:
            M = np.float32([[1, 0, np.cos(ang) * ln * t], [0, 1, np.sin(ang) * ln * t]])
        wk = (1 - t) ** 1.5 + 0.05
        acc = acc + warp_affine(planes, M) * wk
        wsum += wk
    return acc, wsum


def _centroid(mask, ww, hh):
    ys, xs = torch.nonzero(mask, as_tuple=True)
    n = int(xs.numel())
    if n == 0:
        return ww / 2, hh / 2
    return int(xs.sum().item()) / n, int(ys.sum().item()) / n


def _smear_depth(d, nmap, smear, smear_angle, smear_radial):
    H, W = d.shape
    ln = smear * max(H, W) * 0.15
    ang = np.deg2rad(smear_angle)
    AD = torch.stack([nmap, d * nmap])
    cx = cy = 0.0
    if smear_radial > 0.5:
        cx, cy = _centroid(nmap > 0.5, W, H)
    acc, wsum = _smear_planes(AD, ln, ang, smear_radial > 0.5, cx, cy)
    A2 = torch.maximum(AD[0], acc[0] / wsum * 1.6).clamp(0, 1)
    D2 = torch.minimum(torch.maximum(AD[1], acc[1] / wsum * 1.6), A2)
    return (D2 + d * (1 - A2)).clamp(0, 1)


def _stretch_iter(img, m, N, length, vp, shift, blur, opacity, fade, shift_col, stops, focus_t0=0.0):
    """Копії предмета від найдальшої до найближчої: premult RGBA (4,H,W)."""
    _, H, W = img.shape
    rgba = torch.cat([img * m[None], m[None]], 0)
    vx, vy = vp[0] * W, vp[1] * H
    for k in range(N, 0, -1):
        u = k / N
        s = 1.0 - length * u
        M = np.array([[s, 0, vx * (1 - s) + shift[0] * W * u], [0, s, vy * (1 - s) + shift[1] * H * u]], np.float32)
        c = warp_affine(rgba, M)
        sg = blur * max(H, W) * (0.25 + u)
        if sg > 0.4:
            c = _gblur(c, sg)
        a = c[3].clamp(1e-4, 1)
        col = c[:3] / a[None]
        g0 = torch.as_tensor(_gmap_scalar(focus_t0 + u * (1 - focus_t0), stops), device=img.device)
        lm = gray(col.clamp(0, 1))
        g = g0[:, None, None] * (0.55 + 0.9 * (lm * 2.0).clamp(0, 1))[None]
        sc = float(np.clip(shift_col * (0.25 + 0.75 * u), 0, 1))
        col = col * (1 - sc) + g * sc
        f = opacity * (1 - u) ** fade
        a2 = a * f
        yield torch.cat([col * a2[None], a2[None]], 0)


def _drift_noise_t(shape, time, dev):
    H, W = shape
    HL = max(H, W)
    f = 64.0 / HL
    h, w = max(2, int(round(H * f))), max(2, int(round(W * f)))
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    x /= 64.0
    y /= 64.0
    tau = 2 * np.pi
    n = (np.sin(tau * (1.3 * x + 0.7 * y + 0.08 * time))
         + np.sin(tau * (-0.6 * x + 1.9 * y + 0.05 * time + 0.3))
         + np.sin(tau * (0.9 * x - 1.1 * y + 0.11 * time + 0.6))) / 3.0
    return resize(_tt(n.astype(np.float32), dev).unsqueeze(0), (W, H), "cubic")[0]


def _haze(out, d, stops, amt, start, curve, t_col, scatter, drift=0.0, time=0.0):
    dev = out.device
    HL = max(d.shape)
    dens = amt * _sm(1 - d, start, 1.0) ** curve
    if drift > 0:
        dens = dens * (1.0 + drift * _drift_noise_t(d.shape, time, dev)).clamp(0, 3)
    col = torch.as_tensor(_gmap_scalar(t_col, stops), device=dev)[:, None, None]
    if scatter > 0:
        sc = gaussian(out, max(0.06 * HL, 1.0))
        col = col * (1 - scatter) + _screen(sc, col * 0.6) * scatter
    w = dens.clamp(0, 1)[None]
    return (out * (1 - w) + col * w).clamp(0, 1)


def _silhouette(out, d, stops, amt, lo, hi, bg_amt, bg_lo, detail, rim, rim_r):
    dev = out.device
    H = max(d.shape)
    ms = gaussian(_sm(d, lo, hi).unsqueeze(0), max(0.002 * H, 0.5))[0]
    bright = None
    if bg_amt > 0 or rim > 0:
        y = gray(out)
        yb = gaussian(y.unsqueeze(0), max(0.01 * H, 0.5))[0]
        bgm = ms < 0.5
        if bool(bgm.any()):
            p0, p1 = _percentile(yb[bgm], (2, 98))
        else:
            p0, p1 = 0.0, 1.0
        tb = ((yb - p0) / max(p1 - p0, 1e-4)).clamp(0, 1)
        tb = bg_lo + (1 - bg_lo) * (0.7 * tb + 0.3 * (1 - d))
        bright = _gmap_pos(tb, stops, dev)
    if bg_amt > 0:
        wb = (amt * bg_amt * (1 - ms))[None]
        out = out * (1 - wb) + bright * wb
    dark = torch.as_tensor(_gmap_scalar(0.0, stops) * 0.35, device=dev)[:, None, None]
    subj = dark + out * detail
    ws = (amt * ms)[None]
    out = out * (1 - ws) + subj * ws
    if rim > 0:
        src = gaussian(bright * (1 - ms)[None], max(rim_r * H, 0.5))
        edge = (ms * (1 - gaussian(ms.unsqueeze(0), max(rim_r * H, 0.5))[0]) * 4).clamp(0, 1)
        out = _screen(out, src * (edge * rim * amt)[None] * 1.6)
    return out.clamp(0, 1)


def _glow(out, amt, radius_px, thr, warmth, stops):
    dev = out.device
    y = gray(out)
    m = _sm(y, thr, min(thr + 0.25, 0.999))
    src = out * m[None]
    bl = (_gblur(src, radius_px) + _gblur(src, 2 * radius_px) + _gblur(src, 4 * radius_px)) * (1.0 / 3.0)
    if warmth > 0:
        pal = _gmap_scalar(0.5 + 0.5 * warmth, stops)
        by = gray(bl)[None]
        pn = torch.as_tensor(np.clip(pal / max(float(pal.max()), 1e-3), 0, 1).astype(np.float32), device=dev)[:, None, None]
        bl = bl * (1 - warmth) + by * pn * warmth
    return _screen(out, (bl * amt * 1.0).clamp(0, 1))


def _numpy_fallback(name, fn, out, *args):
    """Підкрок, не портований на torch: round-trip через CPU (look.* на numpy)."""
    FALLBACKS_USED.add(name)
    arr = np.ascontiguousarray(out.permute(1, 2, 0).detach().cpu().numpy())
    res = fn(arr, *args)
    return torch.from_numpy(np.ascontiguousarray(res)).to(out.device).permute(2, 0, 1).contiguous()


def _pp(x):
    return 1.0 - (1.0 - torch.remainder(x, 2.0)).abs()


def _lut_apply(t, lut):
    n = lut.shape[0]
    idx = (t * (n - 1) + 0.5).to(torch.int32).clamp(0, n - 1).long()
    return lut[idx].permute(2, 0, 1)


# =============================================================================== головна функція
def _defaults():
    global _DEFAULTS
    if _DEFAULTS is None:
        _DEFAULTS = {k: v.default for k, v in inspect.signature(_L.depth_look_lens).parameters.items()
                     if v.default is not inspect.Parameter.empty}
    return _DEFAULTS


def depth_look_lens_t(img, depth, noise=None, **kw):
    """img (3,H,W) float32 0..1, depth (H,W) float32 0..1 - тензори на одному пристрої. -> (3,H,W) float32.
    noise = make_noise_for(...) (CPU-зерно, можна підготувати наперед у потоці); None - згенерується тут."""
    p = dict(_defaults())
    bad = [k for k in kw if k not in p]
    if bad:
        raise TypeError("unknown depth_look_lens args: %s" % bad)
    p.update(kw)
    if p["cache"] is not None:
        raise NotImplementedError("look_torch: cache не підтримується (кеш шарів лише у numpy-шляху)")
    g = type("P", (), p)           # доступ як g.name
    dev = img.device
    img0 = img.clamp(0, 1)
    d0 = depth.clamp(0, 1)
    _, H, W = img0.shape
    HL = max(H, W)
    time = float(g.time)
    stops = g.gradient_stops or _L.LENS_STOPS
    stretch_steps, stretch_opacity = g.stretch_steps, g.stretch_opacity

    # ---------- prepare_lens (шари глибини)
    focus = g.focus
    if g.blur_target is not None and g.blur_target >= 0:
        p5, p95 = _percentile(d0, (5, 95))
        focus = float(p5 + float(np.clip(g.blur_target, 0, 1)) * (p95 - p5))
    r_min = g.r_min * g.blur_amount
    r_max = g.r_max * g.blur_amount
    n_layers = g.n_layers
    cs = np.linspace(0, 1, n_layers, dtype=np.float32)
    step = float(cs[1] - cs[0])
    cs_t = _tt(cs, dev)
    Wall = (1.0 - (d0[None] - cs_t[:, None, None]).abs() / step).clamp(0, 1)          # (n,H,W)
    wmax = Wall.reshape(n_layers, -1).amax(1).cpu().numpy()
    layers = []
    for i, c in enumerate(cs):
        if wmax[i] < 1e-3:
            layers.append(None)
            continue
        r = (r_min + r_max * abs(float(c) - focus) ** g.lens_gamma) * HL
        layers.append(dict(i=i, c=float(c), r=float(r)))
    rmaxpx = max((r_min + r_max) * HL, 1e-6)

    def layer_planes(L):
        """premult RGB + A шару: диск-блюр з дисперсією каналів, у fp16-округленні, як prepare_lens."""
        w = Wall[L["i"]]
        r = L["r"]
        disp = g.dispersion
        radii = [r, r * (1 + disp), r * 1.0, r * (1 - disp)]            # a, R, G, B
        src = torch.cat([w[None], img0 * w[None]], 0)                     # (4,H,W)
        res = [None] * 4
        for rv in sorted(set(radii)):
            ch = [j for j in range(4) if radii[j] == rv]
            blurred = _disk_blur(src[ch], rv)
            for q, j in enumerate(ch):
                res[j] = blurred[q]
        pl = torch.stack(res)                                             # a,R,G,B
        pl = pl.half().float()
        return pl[1:], pl[0:1]

    comp = torch.zeros(4, H, W, device=dev)

    def over(rgb, a):
        nonlocal comp
        ia = 1 - a
        comp = torch.cat([rgb + comp[:3] * ia, a + comp[3:4] * ia], 0)

    # ---------- деформація
    d = d0
    img_w = img0
    warp, smear = g.warp, g.smear
    nmap = _near_map(d, layers, rmaxpx, g.gradient_onset, dev) if (warp > 0 or smear > 0) else None
    wmap = None
    if warp > 0 and nmap is not None:
        wmap = _warp_map(d, nmap, warp, g.warp_scale, g.warp_seed, g.warp_speed * time if g.warp_speed > 0 else 0.0)
        d = _remap_planes(d[None], wmap)[0].clamp(0, 1)
        img_w = _remap_planes(img0, wmap)
        nmap = _near_map(d, layers, rmaxpx, g.gradient_onset, dev)

    stretch_done = stretch_steps <= 0 or stretch_opacity <= 0
    m = None
    if not stretch_done:
        m = _sm(d, g.subject_lo, g.subject_hi)
        m = gaussian(m.unsqueeze(0), 0.003 * HL)[0]
    use_grad = g.far_gradient > 0 or g.near_tint > 0 or g.subject_map > 0 or g.bg_bands > 0

    # ---------- анімація градієнта
    gmode = int(g.grad_mode) if (use_grad and int(g.grad_mode) in (1, 2, 3, 4, 5) and float(g.grad_amount) > 0) else 0
    gamt = float(np.clip(g.grad_amount, 0, 1))
    g_lut = g_field4 = g_wave = None
    g_cyc = 0.0
    g_ang = np.deg2rad(float(g.grad_angle))
    g_dyn = float(g.grad_offset_y) * HL / H
    yy_ = torch.arange(H, device=dev, dtype=torch.float32)[:, None]
    xx_ = torch.arange(W, device=dev, dtype=torch.float32)[None, :]
    if gmode:
        g_cyc = _L._grad_cycles(g.grad_speed, g.grad_loop, g.grad_phase, time)
        g_lut = _tt(_L._gradient_map_pos(np.linspace(0, 1, 2048, dtype=np.float32), stops), dev)
        g_wl = max(float(g.grad_wavelength), 1e-3)
        if gmode == 4:
            tp = ((focus - d).abs() / max(focus, 1e-3)).clamp(0, 1)
            t4 = _pp(2.0 * (tp / g_wl - g_cyc))
            t4 = tp * (1 - gamt) + t4 * gamt
            g_field4 = _lut_apply(t4, g_lut)
            g_d4 = t4 - tp
        if gmode == 5:
            a5 = g_ang - np.pi / 2
            g_wave = torch.sin(2 * np.pi * (((xx_ * float(np.cos(a5)) + yy_ * float(np.sin(a5))) / HL) / g_wl - g_cyc)) * (float(g.grad_amp) * gamt)

    def grad_t0(t0):
        if gmode == 1:
            return float(_L._pingpong(t0 - g_dyn * gamt))
        if gmode == 2:
            return float(_L._pingpong(t0 - 2.0 * g_cyc * gamt))
        if gmode == 3:
            return float(t0 * (1 - gamt) + _L._pingpong(t0 + 2.0 * g_cyc) * gamt)
        return t0

    def grad_band_coord():
        yn = yy_ / max(H - 1, 1)
        xn = xx_ / max(H - 1, 1)
        flip = g.band_flip > 0.5
        if gmode == 2:
            c = xn * float(np.cos(g_ang)) + yn * float(np.sin(g_ang))
        else:
            c = yn + 0.0 * xn
        if flip:
            c = 1.0 - c
        if gmode == 1:
            c = _pp(c - g_dyn * gamt)
        elif gmode == 2:
            c = _pp(c - 2.0 * g_cyc * gamt)
        elif gmode == 3:
            c = c * (1 - gamt) + _pp(c + 2.0 * g_cyc) * gamt
        elif gmode == 4:
            c = _pp(c + g_d4)
        elif gmode == 5:
            c = _pp(c + g_wave)
        return c.expand(H, W)

    band_c = None
    stretch_lighten = g.stretch_lighten
    for L in layers:
        if L is None:
            continue
        if not stretch_done and L["c"] >= g.subject_lo:
            for c4 in _stretch_iter(img_w, m, int(stretch_steps), g.stretch_length, g.vanishing_point,
                                    g.stretch_shift, g.stretch_blur, stretch_opacity, g.stretch_fade,
                                    g.stretch_color_shift, stops):
                prev = comp[:3] if stretch_lighten > 0 else None
                over(c4[:3], c4[3:4])
                if prev is not None:
                    comp = torch.cat([comp[:3] * (1 - stretch_lighten) + torch.maximum(comp[:3], prev) * stretch_lighten,
                                      comp[3:4]], 0)
            stretch_done = True
        rgb, a = layer_planes(L)                      # (3,H,W), (1,H,W)
        r_rel = L["r"] / rmaxpx
        near_d = float(1 - _sm_s(r_rel, g.gradient_onset, 0.55))
        if wmap is not None:
            pl = _remap_planes(torch.cat([rgb, a], 0), wmap)
            rgb, a = pl[:3], pl[3:4]
        if smear > 0 and near_d > 0.05:
            hh, ww = a.shape[1:]
            ln = smear * near_d * max(hh, ww) * 0.15
            ang = np.deg2rad(g.smear_angle)
            radial = g.smear_radial > 0.5
            cx = cy = 0.0
            if radial:
                cx, cy = _centroid(a[0] > 0.5, ww, hh)
            acc, wsum = _smear_planes(torch.cat([rgb, a], 0), ln, ang, radial, cx, cy)
            rgb = torch.maximum(rgb, acc[:3] / wsum * 1.6)
            a = torch.maximum(a, acc[3:4] / wsum * 1.6).clamp(0, 1)
            rgb = torch.minimum(rgb, a)
        mix = g.near_tint * (1 - _sm_s(r_rel, 0, 0.1)) + g.far_gradient * _sm_s(r_rel, g.gradient_onset, 0.55)
        mix = float(np.clip(mix, 0, 1))
        un = rgb / a.clamp_min(1e-4)
        lm = None
        if mix > 0:
            t0 = float(np.clip(abs(focus - L["c"]) / max(focus, 1e-3), 0, 1))
            if gmode in (1, 2, 3):
                t0 = grad_t0(t0)
            gg = torch.as_tensor(_gmap_scalar(t0, stops), device=dev)[:, None, None]
            if gmode == 4:
                gg = g_field4
            elif gmode == 5:
                gg = _lut_apply(_pp(t0 + g_wave), g_lut)
            lm = gray(un.clamp(0, 1))
            mod = (1 - g.luma_keep) + g.luma_keep * (lm * 2.5).clamp(0, 1.3)
            col = un * (1 - mix) + (gg * mod[None] * g.gradient_gain).clamp(0, 1) * mix
        else:
            col = un
        if g.subject_map > 0:
            if lm is None:
                lm = gray(un.clamp(0, 1))
            near_w = float(g.subject_map * (1 - _sm_s(r_rel, g.gradient_onset, 0.55)))
            lc = ((lm - 0.5) * g.subject_contrast + 0.5).clamp(0, 1)
            sm_ = _gmap_pos(lc, stops, dev)
            col = col * (1 - near_w) + (sm_ * g.gradient_gain).clamp(0, 1) * near_w
        if g.bg_bands > 0:
            far_w = float(g.bg_bands * _sm_s(r_rel, g.gradient_onset, 0.55))
            sig = max(1.0, max(H, W) * 0.04)
            if gmode:
                if band_c is None:
                    band_c = gaussian(_gmap_pos(grad_band_coord(), stops, dev), sig)
            elif band_c is None:                     # смуги сталі по x: рахуємо один стовпчик і розтягуємо
                yy = torch.linspace(0, 1, H, device=dev, dtype=torch.float32)
                if g.band_flip > 0.5:
                    yy = yy.flip(0)
                col1 = _gmap_pos(yy[:, None], stops, dev)                      # (3,H,1)
                k1 = gauss_kernel(sig)
                band_c = _axis_filter(col1, 1, k1, "reflect101").expand(3, H, W)
            col = col * (1 - far_w) + (band_c * g.gradient_gain).clamp(0, 1) * far_w
        over(col * a, a)
    out = comp[:3] / comp[3:4].clamp_min(1e-3)
    out = out.clamp(0, 1)
    if smear > 0 and nmap is not None:
        d = _smear_depth(d, nmap, smear, g.smear_angle, g.smear_radial)
    if g.silhouette > 0:
        out = _silhouette(out, d, stops, g.silhouette, g.sil_lo, g.sil_hi, g.sil_bg, g.sil_bg_lo, g.sil_detail,
                          g.sil_rim, g.sil_rim_radius)
    if g.haze > 0:
        out = _haze(out, d, stops, g.haze, g.haze_start, g.haze_curve, g.haze_t, g.haze_scatter, g.haze_drift, time)
    if g.trail > 0:
        trail_angle = g.trail_angle
        if g.trail_wobble:
            trail_angle = float(trail_angle + g.trail_wobble * np.sin(2 * np.pi * time * 0.3))
        dn = d.detach().cpu().numpy()
        out = _numpy_fallback("_light_trail", lambda o: _L._light_trail(
            o, dn, stops, g.trail, trail_angle, g.trail_length, g.trail_source, g.trail_threshold,
            g.trail_decay, g.trail_tint, g.subject_lo, g.subject_hi), out)
    out = _chroma_ab(out, g.chroma_ab * HL / 2144.0)
    if g.halation_strength > 0:
        y = gray(out)
        mk = _sm(y, g.halation_threshold, min(g.halation_threshold + 0.2, 0.999)) * y
        tint = g.halation_tint
        glow_ = torch.stack([gaussian(mk[None], max(g.halation_radius * HL * rm, 0.5))[0] * float(tint[c])
                             for c, rm in ((0, 1.0), (1, 0.55), (2, 0.3))])
        out = 1 - (1 - out) * (1 - (glow_ * g.halation_strength * 2.0).clamp(0, 1))
    pulse = 1.0
    if g.glow_pulse:
        pulse = max(0.0, 1.0 + g.glow_pulse * float(np.sin(2 * np.pi * time * g.pulse_hz)))
    if g.glow > 0:
        out = _glow(out, g.glow * pulse, g.glow_radius * HL, g.glow_threshold, g.glow_warmth, stops)
    if g.spark > 0:
        out = _numpy_fallback("_sparks", lambda o: _L._sparks(
            o, g.spark * pulse, g.spark_threshold, g.spark_size, g.spark_rays, g.spark_angle, g.spark_length), out)
    if g.grain_amount > 0:
        if noise is None:
            noise = make_noise_for(H, W, p)
        out = _film_grain(out, g.grain_amount, g.grain_size, g.grain_roughness, noise)
    return out.clamp(0, 1)


def resolve_device(device):
    """'cuda' / 'cpu' / 'mps' / torch.device / None, 'auto' -> torch.device (cuda, якщо є)."""
    if isinstance(device, torch.device):
        return device
    d = str(device or "auto").lower()
    if d in ("auto", "1", "true", "yes", "on"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(d)


def depth_look_lens(img, depth, device="auto", noise=None, **kw):
    """numpy-обгортка: img HxWx3 float32 0..1, depth HxW float32 0..1 -> HxWx3 float32 numpy."""
    dev = resolve_device(device)
    with torch.no_grad():
        it = _tt(np.asarray(img, np.float32).transpose(2, 0, 1), dev)
        dt = _tt(np.asarray(depth, np.float32), dev)
        out = depth_look_lens_t(it, dt, noise=noise, **kw)
        return np.ascontiguousarray(out.permute(1, 2, 0).cpu().numpy(), dtype=np.float32)
