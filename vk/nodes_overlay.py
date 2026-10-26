"""Overlay On Boxes — stamp your own graphic onto every detected box, every frame.

This is the node ComfyUI does not have. ImageCompositeMasked, ImageComposite+ and
FL_PasteOnCanvas all take ONE fixed x/y. Here the position and size come from the
detector, so a HUD frame you generated yourself lands on the object and follows it.
"""

import numpy as np
import torch

from .boxes import ON_MISSING, FORMATS, any_type, frame_at, hex_to_rgb, to_frames

FIT = ["nine_slice", "stretch", "contain", "cover", "fixed_scale"]
ANCHOR = ["box", "center", "top_left", "top_center", "above_box", "below_box",
          "corner_top_left", "corner_top_right", "corner_bottom_left", "corner_bottom_right"]
ALPHA_FROM = ["auto", "mask_input", "luminance", "chroma_key", "opaque"]
SOURCE_MODE = ["single", "by_frame", "by_box_index", "by_track_id"]


def _resize(a, w, h):
    import cv2
    if w <= 0 or h <= 0:
        return np.zeros((max(h, 0), max(w, 0), a.shape[2]), np.float32)
    interp = cv2.INTER_AREA if (w < a.shape[1] or h < a.shape[0]) else cv2.INTER_LINEAR
    return cv2.resize(a, (w, h), interpolation=interp)


def _nine_slice(a, W, H, c):
    """Scale to WxH without stretching the corners — the whole point of a HUD frame."""
    h, w = a.shape[:2]
    c = int(max(1, min(c, (h - 1) // 2, (w - 1) // 2, (H - 1) // 2, (W - 1) // 2)))
    out = np.zeros((H, W, a.shape[2]), np.float32)
    mw, mh = W - 2 * c, H - 2 * c
    sw, sh = w - 2 * c, h - 2 * c

    out[0:c, 0:c] = _resize(a[0:c, 0:c], c, c)
    out[0:c, W - c:W] = _resize(a[0:c, w - c:w], c, c)
    out[H - c:H, 0:c] = _resize(a[h - c:h, 0:c], c, c)
    out[H - c:H, W - c:W] = _resize(a[h - c:h, w - c:w], c, c)
    if mw > 0 and sw > 0:
        out[0:c, c:W - c] = _resize(a[0:c, c:w - c], mw, c)
        out[H - c:H, c:W - c] = _resize(a[h - c:h, c:w - c], mw, c)
    if mh > 0 and sh > 0:
        out[c:H - c, 0:c] = _resize(a[c:h - c, 0:c], c, mh)
        out[c:H - c, W - c:W] = _resize(a[c:h - c, w - c:w], c, mh)
    if mw > 0 and mh > 0 and sw > 0 and sh > 0:
        out[c:H - c, c:W - c] = _resize(a[c:h - c, c:w - c], mw, mh)
    return out


def _fit(rgba, bw, bh, fit, corner, scale):
    h, w = rgba.shape[:2]
    if fit == "nine_slice":
        return _nine_slice(rgba, max(1, bw), max(1, bh), corner)
    if fit == "stretch":
        return _resize(rgba, max(1, bw), max(1, bh))
    if fit == "fixed_scale":
        return _resize(rgba, max(1, int(w * scale)), max(1, int(h * scale)))
    k = min(bw / w, bh / h) if fit == "contain" else max(bw / w, bh / h)
    return _resize(rgba, max(1, int(w * k)), max(1, int(h * k)))


def _place(box, pw, ph, anchor, off_x, off_y):
    x1, y1, x2, y2 = box["x1"], box["y1"], box["x2"], box["y2"]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    pos = {
        "box": (x1, y1),
        "center": (cx - pw / 2, cy - ph / 2),
        "top_left": (x1, y1),
        "top_center": (cx - pw / 2, y1),
        "above_box": (cx - pw / 2, y1 - ph),
        "below_box": (cx - pw / 2, y2),
        "corner_top_left": (x1 - pw / 2, y1 - ph / 2),
        "corner_top_right": (x2 - pw / 2, y1 - ph / 2),
        "corner_bottom_left": (x1 - pw / 2, y2 - ph / 2),
        "corner_bottom_right": (x2 - pw / 2, y2 - ph / 2),
    }[anchor]
    return int(round(pos[0])) + off_x, int(round(pos[1])) + off_y


class VKOverlayOnBoxes:
    """Stamp an image onto every detected box on every frame."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Video frames the graphic is stamped onto — from Load Video · Pro or any image node."}),
                "overlay": ("IMAGE", {"tooltip": "Your own graphic — the IMAGE output of Load Image: a generated HUD frame, a sticker, an icon. One image, or a batch (see source_mode)."}),
                "boxes": (any_type, {"tooltip": "Boxes — the boxes output of Detect Objects or Select Box (or bounding_box of YOLO Detect)."}),
                "fit": (FIT, {"default": "nine_slice", "tooltip": "How the graphic is resized to the box. nine_slice keeps the corners sharp and stretches only the edges — use it for frames and brackets. stretch distorts. contain/cover keep the aspect ratio. fixed_scale ignores the box size."}),
                "corner": ("INT", {"default": 24, "min": 1, "max": 512, "tooltip": "nine_slice only: how many pixels at each corner of your graphic must NOT stretch."}),
                "scale": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05, "tooltip": "Extra size multiplier applied after fit."}),
                "anchor": (ANCHOR, {"default": "box", "tooltip": "Where the graphic sits relative to the box. box = fills it. above_box/below_box = label plates. corner_* = a mark on one corner."}),
                "offset_x": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "shift right (+) or left (-), px"}),
                "offset_y": ("INT", {"default": 0, "min": -4096, "max": 4096, "tooltip": "shift down (+) or up (-), px"}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "0 = invisible, 1 = solid"}),
                "alpha_from": (ALPHA_FROM, {"default": "auto", "tooltip": "Where transparency comes from. auto = the mask input if connected, otherwise brightness. luminance = black areas of your graphic become transparent (right for a HUD drawn bright-on-black). chroma_key = a colour becomes transparent."}),
                "key_color": ("STRING", {"default": "#000000", "tooltip": "chroma_key only: the colour to knock out."}),
                "key_tolerance": ("FLOAT", {"default": 0.12, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "chroma_key only: how close a colour must be to key_color"}),
                "invert_alpha": ("BOOLEAN", {"default": False, "tooltip": "Turn on if the graphic appears inside-out. Load Image gives a mask where 1 = transparent."}),
                "source_mode": (SOURCE_MODE, {"default": "single", "tooltip": "Which image from the overlay batch to use. single = the first one everywhere. by_frame = an animated overlay, image N on frame N. by_box_index / by_track_id = a different graphic per object."}),
                "max_boxes": ("INT", {"default": 20, "min": 1, "max": 200, "tooltip": "stamp on at most this many boxes per frame"}),
                "format_in": (FORMATS, {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
                "on_missing": (ON_MISSING, {"default": "empty", "tooltip": "What to do on a frame with no detection."}),
            },
            "optional": {
                "overlay_mask": ("MASK", {"tooltip": "Transparency of the graphic. Connect the MASK output of Load Image here."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "coverage_mask", "info")
    FUNCTION = "overlay"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Stamp your own graphic onto every detected box on every frame — the box supplies the position and size."

    def overlay(self, images, overlay, boxes, fit, corner, scale, anchor, offset_x, offset_y,
                opacity, alpha_from, key_color, key_tolerance, invert_alpha, source_mode,
                max_boxes, format_in, on_missing, overlay_mask=None):
        n, H, W = images.shape[0], images.shape[1], images.shape[2]
        frames = to_frames(boxes, format_in)
        if on_missing != "empty":
            from .boxes import fill_missing
            frames = fill_missing(frames, on_missing, W, H)

        src = overlay.detach().cpu().float().numpy()          # (B,h,w,C)
        alphas = []
        for i in range(src.shape[0]):
            a = src[i]
            if a.shape[2] == 4:
                al = a[:, :, 3].copy()
                a = a[:, :, :3]
            elif overlay_mask is not None and alpha_from in ("auto", "mask_input"):
                m = overlay_mask.detach().cpu().float().numpy()
                al = m[min(i, m.shape[0] - 1)].copy()
            elif alpha_from == "chroma_key":
                kr, kg, kb = [v / 255.0 for v in hex_to_rgb(key_color)]
                d = np.sqrt((a[:, :, 0] - kr) ** 2 + (a[:, :, 1] - kg) ** 2 + (a[:, :, 2] - kb) ** 2)
                al = np.clip((d - key_tolerance) / max(key_tolerance, 1e-3), 0, 1)
            elif alpha_from == "opaque":
                al = np.ones(a.shape[:2], np.float32)
            else:                                             # auto without a mask, or luminance
                al = np.clip(a[:, :, 0] * 0.299 + a[:, :, 1] * 0.587 + a[:, :, 2] * 0.114, 0, 1)
            if invert_alpha:
                al = 1.0 - al
            alphas.append(np.dstack([a[:, :, :3], al * opacity]).astype(np.float32))

        out = images.detach().cpu().float().numpy().copy()
        cov = np.zeros((n, H, W), np.float32)
        stamped = 0

        for i in range(n):
            f = frame_at(frames, i)[:max_boxes]
            base = out[i]
            for j, b in enumerate(f):
                if source_mode == "by_frame":
                    k = i
                elif source_mode == "by_box_index":
                    k = j
                elif source_mode == "by_track_id":
                    k = b["id"] if b["id"] >= 0 else j
                else:
                    k = 0
                rgba = alphas[k % len(alphas)]

                bw = int(round((b["x2"] - b["x1"]) * scale))
                bh = int(round((b["y2"] - b["y1"]) * scale))
                patch = _fit(rgba, bw, bh, fit, corner, scale)
                ph, pw = patch.shape[:2]
                if ph < 1 or pw < 1:
                    continue
                x, y = _place(b, pw, ph, anchor, offset_x, offset_y)

                sx0, sy0 = max(0, -x), max(0, -y)
                dx0, dy0 = max(0, x), max(0, y)
                cw = min(pw - sx0, W - dx0)
                ch = min(ph - sy0, H - dy0)
                if cw <= 0 or ch <= 0:
                    continue

                p = patch[sy0:sy0 + ch, sx0:sx0 + cw]
                a = p[:, :, 3:4]
                dst = base[dy0:dy0 + ch, dx0:dx0 + cw]
                base[dy0:dy0 + ch, dx0:dx0 + cw] = dst * (1 - a) + p[:, :, :3] * a
                cv = cov[i, dy0:dy0 + ch, dx0:dx0 + cw]
                cov[i, dy0:dy0 + ch, dx0:dx0 + cw] = np.maximum(cv, a[:, :, 0])
                stamped += 1

        info = f"frames {n} · stamps {stamped} · overlay {src.shape[0]} image(s) · fit {fit} · anchor {anchor}"
        print(f"[VKOverlayOnBoxes] {info}")
        return (torch.from_numpy(np.clip(out, 0, 1)), torch.from_numpy(cov), info)


NODES = {"VKOverlayOnBoxes": VKOverlayOnBoxes}
DISPLAY = {"VKOverlayOnBoxes": "Overlay On Boxes"}
