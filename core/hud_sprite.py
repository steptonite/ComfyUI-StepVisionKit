"""LexHUDSprite — кастомна HUD-рамка (PNG з альфою) на трекнутому YOLO-боксі.

Нащо: LexHUDBoxes малює рамку кодом (cv2.line). Тут рамка — ЗОВНІШНЯ картинка,
яку ти генеруєш де завгодно (GPT/Nano Banana/Photoshop), а нода її:
  · чіпляє до бокса й веде за ним по кадрах (не статична),
  · масштабує під бокс із запасом (HUD завжди БІЛЬШИЙ за бокс — не зрізає його),
  · тримає задане співвідношення сторін і мінімальну роздільність,
  · тягне кути 9-slice, щоб орнамент по кутах не розмазувало розтягуванням,
  · згладжує тремтіння бокса (EMA), бо покадровий YOLO стрибає,
  · вирізає з відео сам бокс (дірка) і віддає окремо альфа-секвенцію HUD.
"""

import numpy as np
import torch

from .boxes import any_type, _to_xyxy_frames, lab_text


def _put_text_f32(dst_rgb, dst_a, txt, org, font, fs, color, thick):
    """Підпис на float32-полотні.

    🔴 cv2.putText з OpenCV 5 приймає ТІЛЬКИ CV_8U (`img.depth() == CV_8U`);
    4.x мовчки їв float32, тому на Маку працювало, а на орендованій машині
    з opencv 5.0 нода падала. Малюємо штамп у uint8-ROI і змішуємо як альфу —
    згладжування лишається, від версії OpenCV більше не залежимо.
    """
    import cv2

    (tw, th), bl = cv2.getTextSize(txt, font, fs, thick)
    pad = thick + 2
    x0 = max(0, org[0] - pad)
    y0 = max(0, org[1] - th - pad)
    x1 = min(dst_rgb.shape[1], org[0] + tw + pad)
    y1 = min(dst_rgb.shape[0], org[1] + bl + pad)
    if x1 <= x0 or y1 <= y0:
        return
    stamp = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.putText(stamp, txt, (org[0] - x0, org[1] - y0), font, fs, 255, thick, cv2.LINE_AA)
    a = (stamp.astype(np.float32) / 255.0)[:, :, None]
    roi = dst_rgb[y0:y1, x0:x1]
    roi *= 1.0 - a
    roi += a * np.asarray(color, np.float32)
    da = dst_a[y0:y1, x0:x1]
    np.maximum(da, a[:, :, 0], out=da)


def _resize_9slice(img, tw, th, corner):
    """Ресайз із збереженням кутів. corner=0 -> звичайний ресайз."""
    import cv2

    tw, th = max(1, int(tw)), max(1, int(th))
    h, w = img.shape[:2]
    c = int(corner)
    if c <= 0 or w < 2 * c + 2 or h < 2 * c + 2:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    # 🔴 corner задано в пікселях ВИХІДНОЇ картинки (1536-й рамці кут ~240 px).
    # У кадрі HUD зазвичай менший, тому кут масштабуємо разом зі спрайтом —
    # інакше на боксі 500 px кутовий кронштейн з'їв би всю рамку.
    k = min(tw / float(w), th / float(h))
    d = max(1, int(round(c * k)))
    d = min(d, tw // 2 - 1, th // 2 - 1)
    if d < 1:
        return cv2.resize(img, (tw, th), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((th, tw, img.shape[2]), img.dtype)
    src_r = ((0, c), (c, h - c), (h - c, h))
    src_c = ((0, c), (c, w - c), (w - c, w))
    dst_r = ((0, d), (d, th - d), (th - d, th))
    dst_c = ((0, d), (d, tw - d), (tw - d, tw))
    for sr, dr in zip(src_r, dst_r):
        for sc, dc in zip(src_c, dst_c):
            dh, dw = dr[1] - dr[0], dc[1] - dc[0]
            patch = img[sr[0]:sr[1], sc[0]:sc[1]]
            if dh <= 0 or dw <= 0 or patch.size == 0:
                continue
            out[dr[0]:dr[1], dc[0]:dc[1]] = cv2.resize(
                patch, (dw, dh), interpolation=cv2.INTER_LINEAR)
    return out


_FONT = 4  # cv2.FONT_HERSHEY_TRIPLEX — рівний технічний шрифт без засічок-кучерів


def _parse_hex(s):
    """'#RRGGBB' / 'RRGGBB' -> (r,g,b) у 0..1. Порожньо або сміття -> None."""
    s = (s or "").strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return None
    try:
        return tuple(int(s[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    except ValueError:
        return None


# 🔴 Був вільний рядок — руками легко написати «16/9» чи «3-2» і мовчки отримати іншу рамку.
# Список: нічого вводити не треба, помилитись нема де. 'png' рахується з самого спрайта.
ASPECTS = ["box", "png", "1:1", "4:5", "5:4", "2:3", "3:2", "3:4", "4:3", "9:16", "16:9"]


def _parse_aspect(s):
    s = (s or "").strip().lower().replace("х", "x")
    if not s or s in ("box", "auto", "-"):
        return 0.0
    for sep in (":", "x", "/"):
        if sep in s:
            a, _, b = s.partition(sep)
            try:
                a, b = float(a), float(b)
                return a / b if b else 0.0
            except ValueError:
                return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _paste(dst_rgb, dst_a, sprite_rgb, sprite_a, x, y):
    """Альфа-композ спрайта у (x,y). Усе float32 [0,1]. Обрізає по краях кадру."""
    H, W = dst_rgb.shape[:2]
    h, w = sprite_rgb.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(W, x + w), min(H, y + h)
    if x1 <= x0 or y1 <= y0:
        return
    sx0, sy0 = x0 - x, y0 - y
    s_rgb = sprite_rgb[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)]
    s_a = sprite_a[sy0:sy0 + (y1 - y0), sx0:sx0 + (x1 - x0)][:, :, None]
    dst_rgb[y0:y1, x0:x1] = dst_rgb[y0:y1, x0:x1] * (1 - s_a) + s_rgb * s_a
    dst_a[y0:y1, x0:x1] = np.maximum(dst_a[y0:y1, x0:x1], s_a[:, :, 0])


class LexHUDSprite:
    """Кастомна HUD-рамка на боксі: трекінг, аспект, 9-slice, дірка у відео."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "BOUNDING_BOX / BOXES / BBOX — output of LexYOLODetect"}),
                "mode": (["largest", "index", "all"], {"default": "largest", "tooltip": "largest = the biggest box only, index = the box number below, all = a frame on every box"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "box number for mode=index"}),
                "margin_pct": ("FLOAT", {"default": 25.0, "min": -50.0, "max": 400.0, "step": 1.0,
                               "tooltip": "make the HUD N% larger than the box so the frame does NOT clip the subject"}),
                "margin_px": ("INT", {"default": 0, "min": -512, "max": 512, "tooltip": "extra margin in pixels on top of margin_pct; negative shrinks"}),
                "aspect": (ASPECTS, {"default": "box",
                           "tooltip": "'box' = the frame takes the shape of the YOLO box itself: a vertical face gives a vertical frame. The other options force a ratio and the shorter side is extended (the subject is never cropped). 'png' = the aspect of the frame file itself: no deformation at all"}),
                "min_width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8,
                              "tooltip": "minimum HUD size in frame pixels"}),
                "min_height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 8, "tooltip": "minimum HUD height in frame pixels"}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "shift the HUD right (+) or left (-), px"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "shift the HUD down (+) or up (-), px"}),
                "smooth": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.98, "step": 0.02,
                           "tooltip": "0 = raw box (jumpy), 0.5–0.8 = smooth follow. Has no effect in 'all' mode"}),
                "nine_slice": ("INT", {"default": 0, "min": 0, "max": 512, "step": 4,
                               "tooltip": "px of the corners that must NOT be stretched. 0 = plain resize"}),
                "hole": (["none", "box", "hud_rect"], {"default": "box",
                         "tooltip": "what to cut out of the video: nothing / the YOLO box itself / the whole HUD rectangle"}),
                "hole_feather": ("INT", {"default": 0, "min": 0, "max": 128, "tooltip": "soft edge of the cut-out, px"}),
                "alpha_mode": (["loadimage", "mask_direct", "luma", "opaque"], {"default": "loadimage",
                               "tooltip": "loadimage = MASK from LoadImage (it is inverted!). mask_direct = the MASK is already the alpha"}),
                "label": ("STRING", {"default": "auto",
                          "tooltip": "label above the frame. 'auto' = class from the detector (face/hand/person). Empty = no label. Any other text is used verbatim"}),
                "label_scale": ("FLOAT", {"default": 1.0, "min": 0.2, "max": 4.0, "step": 0.1,
                                "tooltip": "label size relative to the frame height"}),
                "label_conf": ("BOOLEAN", {"default": False,
                               "tooltip": "append the detector confidence: 'face 0.92'"}),
                "tint": ("STRING", {"default": "",
                         "tooltip": "recolor the frame, hex #RRGGBB. Empty = keep the original color. The artwork's brightness is preserved; only the hue changes"}),
                "tint_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                                  "tooltip": "0 = original color, 1 = fully the tint color"}),
                "emit": (["composite", "hud_only", "both"], {"default": "composite",
                         "tooltip": "WHICH OUTPUTS TO COMPUTE. A disabled output returns a 1×1 placeholder — if hud_rgb/hud_alpha look empty, this is exactly why.\n· composite = video with the frame and cutout (hud_* are dead)\n· hud_only = just the frame on transparent, at frame size (composite is dead) — to grade/composite the HUD separately from the gray depth\n· both = both, twice the RAM.\nCost: every enabled output is one more full batch of frames, 200 FullHD frames ≈ 5 GB EACH"}),
                "emit_box_mask": ("BOOLEAN", {"default": False,
                                  "tooltip": "A white FILL of the YOLO rectangle (the box itself, not the HUD frame), black around it — exactly the area that hole=box cuts out.\nWhy: pass it on as a mask — cut the face out of another layer, put your own solid under it, feed SAM2 or Cut By Mask, or do a local correction on the face only.\nIf it is not connected anywhere, keep it False: another +1.7 GB per 200 FullHD frames"}),
            },
            "optional": {
                "images": ("IMAGE", {"tooltip": "background video. Without it only the HUD on transparent is returned"}),
                "hud_image": ("IMAGE", {"tooltip": "custom frame. Batch >1 = animated HUD (looped over frames)"}),
                "hud_mask": ("MASK", {"tooltip": "alpha of the custom frame (MASK from Load Image)"}),
            },
        }

    DESCRIPTION = "Puts your own PNG frame on detected boxes and can cut a hole in the video under it. Returns the composite, the HUD alone and masks."
    RETURN_TYPES = ("IMAGE", "MASK", "IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("composite", "composite_alpha", "hud_rgb", "hud_alpha", "box_mask", "info")
    FUNCTION = "run"
    CATEGORY = "StepVisionKit"

    def run(self, boxes, mode, index, margin_pct, margin_px, aspect, min_width, min_height,
            offset_x, offset_y, smooth, nine_slice, hole, hole_feather, alpha_mode,
            label="auto", label_scale=1.0, label_conf=False, tint="", tint_strength=1.0,
            emit="composite", emit_box_mask=False, images=None, hud_image=None, hud_mask=None):
        import cv2

        frames, labels = _to_xyxy_frames(boxes, "auto")
        n_frames = images.shape[0] if images is not None else len(frames)
        if images is not None:
            H, W = int(images.shape[1]), int(images.shape[2])
        elif hud_image is not None:
            H, W = int(hud_image.shape[1]), int(hud_image.shape[2])
        else:
            raise ValueError("LexHUDSprite: either images or hud_image is required")

        # --- спрайт: RGB + альфа
        sprite = None
        if hud_image is not None:
            sp_rgb = hud_image.cpu().numpy().astype(np.float32)
            if hud_mask is not None:
                m = hud_mask.cpu().numpy().astype(np.float32)
                if m.ndim == 2:
                    m = m[None]
                a = 1.0 - m if alpha_mode == "loadimage" else m
            elif alpha_mode == "luma":
                a = sp_rgb.mean(axis=3)
            else:
                a = np.ones(sp_rgb.shape[:3], np.float32)
            if alpha_mode == "opaque":
                a = np.ones(sp_rgb.shape[:3], np.float32)
            tc = _parse_hex(tint)
            if tc is not None and tint_strength > 0:
                # яскравість малюнка лишаємо, підмінюємо тільки відтінок
                luma = sp_rgb.mean(axis=3, keepdims=True)
                colored = luma * np.asarray(tc, np.float32)[None, None, None, :]
                sp_rgb = sp_rgb * (1.0 - tint_strength) + colored * tint_strength
            sprite = [np.concatenate([sp_rgb[i], a[min(i, a.shape[0] - 1)][:, :, None]], axis=2)
                      for i in range(sp_rgb.shape[0])]

        ar = _parse_aspect(aspect)
        if aspect == "png":
            # пропорція самого файлу рамки: спрайт масштабується рівномірно, кути не тягне
            ar = (sprite[0].shape[1] / sprite[0].shape[0]) if sprite else 0.0

        # колір підпису: tint, якщо заданий; інакше беремо середній колір самої рамки
        tcol = _parse_hex(tint)
        if tcol is None:
            if sprite is not None:
                s0 = sprite[0]
                op = s0[:, :, 3] > 0.5
                tcol = tuple(float(v) for v in (s0[:, :, :3][op].mean(0) if op.any()
                                                else np.ones(3, np.float32)))
            else:
                tcol = (1.0, 1.0, 1.0)
        want_comp = emit in ("composite", "both")
        want_hud = emit in ("hud_only", "both")
        # 🔴 преалокація замість list+torch.stack: stack тримає обидві копії одночасно
        dummy_i, dummy_m = torch.zeros(1, 1, 1, 3), torch.zeros(1, 1, 1)
        comp = torch.empty(n_frames, H, W, 3, dtype=torch.float32) if want_comp else dummy_i
        comp_a = torch.empty(n_frames, H, W, dtype=torch.float32) if want_comp else dummy_m
        hud_rgb_o = torch.empty(n_frames, H, W, 3, dtype=torch.float32) if want_hud else dummy_i
        hud_a_o = torch.empty(n_frames, H, W, dtype=torch.float32) if want_hud else dummy_m
        box_m_o = torch.empty(n_frames, H, W, dtype=torch.float32) if emit_box_mask else dummy_m
        ema, drawn = None, 0

        for i in range(n_frames):
            arr = frames[i] if i < len(frames) else (frames[-1] if frames else np.zeros((0, 4), np.float32))
            labs = list(labels[i]) if i < len(labels) else []
            if mode == "largest" and arr.shape[0] > 1:
                k = int(np.argmax((arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])))
                arr, labs = arr[k:k + 1], labs[k:k + 1]
            elif mode == "index" and arr.shape[0] > 1:
                k = min(max(index, 0), arr.shape[0] - 1)
                arr, labs = arr[k:k + 1], labs[k:k + 1]

            base = (images[i].cpu().numpy().astype(np.float32) if images is not None
                    else np.zeros((H, W, 3), np.float32))
            base_a = np.ones((H, W), np.float32) if images is not None else np.zeros((H, W), np.float32)
            hud_only = np.zeros((H, W, 3), np.float32) if want_hud else None
            hud_only_a = np.zeros((H, W), np.float32) if want_hud else None
            box_mask = np.zeros((H, W), np.float32) if emit_box_mask else None

            for bi, b in enumerate(arr):
                bx1, by1, bx2, by2 = [float(v) for v in b]
                bw, bh = max(1.0, bx2 - bx1), max(1.0, by2 - by1)
                cx, cy = bx1 + bw / 2 + offset_x, by1 + bh / 2 + offset_y
                # 🔴 зазор ОДНАКОВИЙ з усіх чотирьох боків, у пікселях.
                # Було: bw*(1+pct) і bh*(1+pct) окремо — на вертикальному боксі 300×420
                # це давало 45 px з боків і 63 px згори/знизу, рамка «не огортала рівномірно».
                m = (bw + bh) / 4.0 * (margin_pct / 100.0) + margin_px
                rw, rh = bw + 2 * m, bh + 2 * m
                if ar > 0:
                    if rw / max(rh, 1e-6) < ar:
                        rw = rh * ar
                    else:
                        rh = rw / ar
                rw, rh = max(rw, float(min_width)), max(rh, float(min_height))

                if mode != "all" and smooth > 0:
                    v = np.array([cx, cy, rw, rh], np.float32)
                    ema = v if ema is None else ema * smooth + v * (1 - smooth)
                    cx, cy, rw, rh = [float(t) for t in ema]

                rx, ry = int(round(cx - rw / 2)), int(round(cy - rh / 2))
                rw_i, rh_i = max(1, int(round(rw))), max(1, int(round(rh)))

                # маска бокса (для вирізання)
                mx1, my1 = max(0, int(bx1)), max(0, int(by1))
                mx2, my2 = min(W, int(bx2)), min(H, int(by2))
                if emit_box_mask and mx2 > mx1 and my2 > my1:
                    box_mask[my1:my2, mx1:mx2] = 1.0

                if hole != "none":
                    cut = np.zeros((H, W), np.float32)
                    if hole == "box":
                        if mx2 > mx1 and my2 > my1:
                            cut[my1:my2, mx1:mx2] = 1.0
                    else:
                        hx1, hy1 = max(0, rx), max(0, ry)
                        hx2, hy2 = min(W, rx + rw_i), min(H, ry + rh_i)
                        if hx2 > hx1 and hy2 > hy1:
                            cut[hy1:hy2, hx1:hx2] = 1.0
                    if hole_feather > 0:
                        k = hole_feather * 2 + 1
                        cut = cv2.GaussianBlur(cut, (k, k), 0)
                    base *= (1 - cut)[:, :, None]
                    base_a *= (1 - cut)

                if sprite is not None:
                    sp = sprite[i % len(sprite)]
                    sp_r = _resize_9slice(sp, rw_i, rh_i, nine_slice)
                    s_rgb, s_a = sp_r[:, :, :3], np.clip(sp_r[:, :, 3], 0.0, 1.0)
                    if want_comp:
                        _paste(base, base_a, s_rgb, s_a, rx, ry)
                    if want_hud:
                        _paste(hud_only, hud_only_a, s_rgb, s_a, rx, ry)
                    drawn += 1

                # --- підпис класу над рамкою (як у стандартного YOLO)
                txt = ""
                if label == "auto":
                    txt = lab_text(labs[bi], label_conf) if bi < len(labs) else ""
                elif label:
                    txt = label
                if txt:
                    fs = max(0.35, rh_i / 320.0) * label_scale
                    th_ = max(1, int(round(fs * 1.6)))
                    (_tw, _th), _ = cv2.getTextSize(txt, _FONT, fs, th_)
                    ty = ry - int(_th * 0.5)
                    if ty - _th < 0:                 # не влазить над рамкою — кладемо всередину
                        ty = ry + int(_th * 1.6)
                    org = (rx, ty)
                    targets = []
                    if want_comp:
                        targets.append((base, base_a))
                    if want_hud:
                        targets.append((hud_only, hud_only_a))
                    for dst_rgb, dst_a in targets:
                        _put_text_f32(dst_rgb, dst_a, txt, org, _FONT, fs, tcol, th_)

            if want_comp:
                comp[i] = torch.from_numpy(np.clip(base, 0, 1, out=base))
                comp_a[i] = torch.from_numpy(np.clip(base_a, 0, 1, out=base_a))
            if want_hud:
                hud_rgb_o[i] = torch.from_numpy(np.clip(hud_only, 0, 1, out=hud_only))
                hud_a_o[i] = torch.from_numpy(np.clip(hud_only_a, 0, 1, out=hud_only_a))
            if emit_box_mask:
                box_m_o[i] = torch.from_numpy(box_mask)

        gb = n_frames * H * W * 4 / 2**30
        info = (f"frames {n_frames} · HUD drawn {drawn} · aspect {aspect or 'box'} · "
                f"cutout '{hole}' · 9-slice {nine_slice}px · smooth {smooth} · "
                f"emit={emit}{'+boxmask' if emit_box_mask else ''} ≈ "
                f"{gb * (3 * want_comp + 3 * want_hud + want_comp + want_hud + emit_box_mask):.2f} GB of outputs")
        print(f"[LexHUDSprite] {info}")
        return (comp, comp_a, hud_rgb_o, hud_a_o, box_m_o, info)
