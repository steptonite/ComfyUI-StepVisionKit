"""Спільні хелпери lex-bridge: типи боксів і парсинг у xyxy."""

import numpy as np
import torch


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")


# ---------------------------------------------------------------- parsing


def lab_text(s, with_conf=False):
    """'face\\x00 0.92' -> 'face' або 'face 0.92'. Порожньо лишається порожнім."""
    name, _, conf = str(s or "").partition("\x00")
    if with_conf and conf:
        try:
            return f"{name} {float(conf):.2f}".strip()
        except ValueError:
            pass
    return name


def _to_xyxy_frames(boxes, fmt):
    """Будь-що -> список кадрів, кожен = список [x1,y1,x2,y2] + список міток."""
    frames, labels = [], []

    def conv(arr, f):
        arr = np.asarray(arr, dtype=np.float32).reshape(-1, 4)
        if f == "xyxy":
            return arr
        if f == "xywh_center":
            cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
            return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
        cx, cy, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        return np.stack([cx, cy, cx + w, cy + h], 1)

    def one_frame(item):
        # список словників — формат ядра ComfyUI
        if isinstance(item, dict):
            item = [item]
        if isinstance(item, (list, tuple)) and len(item) and isinstance(item[0], dict):
            f = fmt if fmt != "auto" else "xywh_topleft"
            raw = [[d.get("x", 0), d.get("y", 0), d.get("width", 0), d.get("height", 0)] for d in item]
            # мітка + впевненість пакуються в один рядок через \x00, розбирає lab_text()
            lab = [str(d.get("label", "")) + "\x00" + str(d.get("confidence", "") or "")
                   for d in item]
            return conv(raw, f), lab
        if isinstance(item, torch.Tensor):
            f = fmt if fmt != "auto" else "xywh_center"   # ultralytics .xywh
            a = item.detach().cpu().float().numpy()
            if a.size == 0:
                return np.zeros((0, 4), np.float32), []
            return conv(a, f), [""] * max(1, a.reshape(-1, 4).shape[0])
        if isinstance(item, (list, tuple)):
            if len(item) == 0:
                return np.zeros((0, 4), np.float32), []
            if isinstance(item[0], (int, float, np.floating)):   # один голий бокс
                f = fmt if fmt != "auto" else "xyxy"
                return conv([item], f), [""]
            f = fmt if fmt != "auto" else "xyxy"
            return conv(list(item), f), [""] * len(item)
        return np.zeros((0, 4), np.float32), []

    # чи це вже "список кадрів"?
    is_batch = isinstance(boxes, (list, tuple)) and len(boxes) and isinstance(
        boxes[0], (list, tuple, torch.Tensor)) and not isinstance(boxes[0], dict) and not (
        isinstance(boxes[0], (list, tuple)) and len(boxes[0]) == 4 and isinstance(
            boxes[0][0], (int, float, np.floating)))

    items = list(boxes) if is_batch else [boxes]
    for it in items:
        a, l = one_frame(it)
        frames.append(a)
        labels.append(l)
    return frames, labels


def _select(arr, labels, mode, index, pad, w, h):
    if arr.shape[0] == 0:
        return arr, labels
    if mode == "largest":
        area = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
        k = int(np.argmax(area))
        arr, labels = arr[k:k + 1], labels[k:k + 1] if labels else [""]
    elif mode == "index":
        k = min(max(index, 0), arr.shape[0] - 1)
        arr, labels = arr[k:k + 1], labels[k:k + 1] if labels else [""]
    if pad:
        arr = arr + np.array([-pad, -pad, pad, pad], np.float32)
    if w and h:
        arr[:, 0::2] = np.clip(arr[:, 0::2], 0, w - 1)
        arr[:, 1::2] = np.clip(arr[:, 1::2], 0, h - 1)
    return arr, labels
