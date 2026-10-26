"""Draw Boxes — the built-in HUD look, with every part switchable.

Use this when you want boxes drawn by code. When you want your OWN generated
graphic on the boxes instead, use Overlay On Boxes.
"""

import numpy as np
import torch

from .boxes import FORMATS, any_type, frame_at, hex_to_rgb, to_frames

STYLE = ["corner_brackets", "solid", "dashed", "crosshair", "dot", "none"]
LABEL_POS = ["above", "below", "inside_top", "inside_bottom"]


class VKDrawBoxes:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "Frames to draw on — the same video the boxes were found on (Load Video · Pro)"}),
                "boxes": (any_type, {"tooltip": "Boxes — the boxes output of Detect Objects or Select Box (or the bounding_box output of YOLO Detect)"}),
                "style": (STYLE, {"default": "corner_brackets", "tooltip": "Shape of the box. none = labels and trails only."}),
                "color": ("STRING", {"default": "#FFFFFF", "tooltip": "Line and text colour, hex."}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 16, "tooltip": "line width, px"}),
                "corner_length": ("INT", {"default": 24, "min": 4, "max": 400, "tooltip": "corner_brackets only: how long each bracket arm is."}),
                "fill_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Tint inside the box. 0 = no fill."}),
                "show_class": ("BOOLEAN", {"default": True, "tooltip": "class name on each box"}),
                "show_confidence": ("BOOLEAN", {"default": False, "tooltip": "detector confidence on each box"}),
                "show_index": ("BOOLEAN", {"default": True, "tooltip": "The #N you type into Select Box."}),
                "show_track_id": ("BOOLEAN", {"default": True, "tooltip": "The id:N that stays on the same object across the whole video."}),
                "label_text": ("STRING", {"default": "", "tooltip": "Fixed text on every box. Leave empty to use the detector's own labels. Massive Attack's screens used invented text, not real data."}),
                "label_size": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 4.0, "step": 0.05, "tooltip": "text size"}),
                "label_position": (LABEL_POS, {"default": "above", "tooltip": "where the text sits relative to the box"}),
                "label_background": ("BOOLEAN", {"default": False, "tooltip": "Solid plate behind the text."}),
                "trail_length": ("INT", {"default": 0, "min": 0, "max": 240, "tooltip": "Draw the path of each tracked object over the last N frames. Needs track ids from Detect Objects."}),
                "format_in": (FORMATS, {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "draw"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Draw detection boxes with full control over shape, colour, labels and motion trails."

    def draw(self, images, boxes, style, color, thickness, corner_length, fill_opacity,
             show_class, show_confidence, show_index, show_track_id, label_text, label_size,
             label_position, label_background, trail_length, format_in):
        import cv2

        r, g, b_ = hex_to_rgb(color)
        bgr = (b_, g, r)
        frames = to_frames(boxes, format_in)
        n = images.shape[0]
        out = []
        history = {}                                   # track id -> [(cx, cy) per frame]

        for i in range(n):
            img = (images[i].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
            f = frame_at(frames, i)

            for j, box in enumerate(f):
                x1, y1, x2, y2 = (int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"]))
                if fill_opacity > 0:
                    ov = img.copy()
                    cv2.rectangle(ov, (x1, y1), (x2, y2), bgr, -1)
                    cv2.addWeighted(ov, fill_opacity, img, 1 - fill_opacity, 0, img)

                if style == "solid":
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness, cv2.LINE_AA)
                elif style == "dashed":
                    step = max(6, thickness * 5)
                    for x in range(x1, x2, step * 2):
                        cv2.line(img, (x, y1), (min(x + step, x2), y1), bgr, thickness, cv2.LINE_AA)
                        cv2.line(img, (x, y2), (min(x + step, x2), y2), bgr, thickness, cv2.LINE_AA)
                    for y in range(y1, y2, step * 2):
                        cv2.line(img, (x1, y), (x1, min(y + step, y2)), bgr, thickness, cv2.LINE_AA)
                        cv2.line(img, (x2, y), (x2, min(y + step, y2)), bgr, thickness, cv2.LINE_AA)
                elif style == "crosshair":
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    L = max(6, corner_length // 2)
                    cv2.line(img, (cx - L, cy), (cx + L, cy), bgr, thickness, cv2.LINE_AA)
                    cv2.line(img, (cx, cy - L), (cx, cy + L), bgr, thickness, cv2.LINE_AA)
                elif style == "dot":
                    cv2.circle(img, ((x1 + x2) // 2, (y1 + y2) // 2),
                               max(2, thickness * 2), bgr, -1, cv2.LINE_AA)
                elif style == "corner_brackets":
                    L = min(corner_length, max(4, (x2 - x1) // 3), max(4, (y2 - y1) // 3))
                    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
                        cv2.line(img, (px, py), (px + dx * L, py), bgr, thickness, cv2.LINE_AA)
                        cv2.line(img, (px, py), (px, py + dy * L), bgr, thickness, cv2.LINE_AA)

                parts = []
                if show_index:
                    parts.append(f"#{j}")
                if show_track_id and box["id"] >= 0:
                    parts.append(f"id:{box['id']}")
                if label_text:
                    parts.append(label_text)
                elif show_class and box["label"]:
                    parts.append(box["label"])
                if show_confidence and box["score"]:
                    parts.append(f"{box['score']:.2f}")
                text = " ".join(parts)

                if text:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    (tw, th), bl = cv2.getTextSize(text, font, label_size, max(1, thickness - 1))
                    ty = {"above": y1 - 6, "below": y2 + th + 6,
                          "inside_top": y1 + th + 6, "inside_bottom": y2 - 6}[label_position]
                    ty = int(np.clip(ty, th + 2, img.shape[0] - 2))
                    if label_background:
                        cv2.rectangle(img, (x1, ty - th - 4), (x1 + tw + 6, ty + bl), bgr, -1)
                        cv2.putText(img, text, (x1 + 3, ty), font, label_size,
                                    (0, 0, 0), max(1, thickness - 1), cv2.LINE_AA)
                    else:
                        cv2.putText(img, text, (x1, ty), font, label_size, bgr,
                                    max(1, thickness - 1), cv2.LINE_AA)

                if trail_length > 0:
                    key = box["id"] if box["id"] >= 0 else j
                    history.setdefault(key, []).append(((x1 + x2) // 2, (y1 + y2) // 2))

            if trail_length > 0:
                for pts in history.values():
                    tail = pts[-trail_length:]
                    for k in range(1, len(tail)):
                        w = k / len(tail)
                        col = tuple(int(c * w) for c in bgr)
                        cv2.line(img, tail[k - 1], tail[k], col, max(1, thickness - 1), cv2.LINE_AA)

            out.append(torch.from_numpy(img[:, :, ::-1].copy().astype(np.float32) / 255.0))

        return (torch.stack(out),)


NODES = {"VKDrawBoxes": VKDrawBoxes}
DISPLAY = {"VKDrawBoxes": "Draw Boxes · HUD"}
