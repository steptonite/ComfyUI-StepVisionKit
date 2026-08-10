"""Box plumbing shared by all VisionKit nodes.

ComfyUI has three mutually incompatible box types:
  BOUNDING_BOX  core / RT-DETR / MediaPipe -> per-frame list of {x,y,width,height,label,score}
  BOXES         ComfyUI-YOLO              -> ultralytics tensor, xywh measured from the CENTRE
  BBOX          kijai SAM2 / KJNodes      -> per-frame list of [x1,y1,x2,y2]

Everything here normalises to one internal shape:
  frames = [ [ {x1,y1,x2,y2,label,score,id}, ... ],  ... ]   # one list per frame
"""

import numpy as np
import torch


class AnyType(str):
    def __ne__(self, other):
        return False


any_type = AnyType("*")

FORMATS = ["auto", "xyxy", "xywh_topleft", "xywh_center"]
ON_MISSING = ["previous", "full_frame", "empty"]
SORT_BY = ["area", "confidence", "left_to_right", "top_to_bottom", "track_id", "detector"]


# --------------------------------------------------------------------- parse


def _conv(arr, fmt):
    arr = np.asarray(arr, dtype=np.float32).reshape(-1, 4)
    if fmt == "xyxy":
        return arr
    a, b, w, h = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if fmt == "xywh_center":
        return np.stack([a - w / 2, b - h / 2, a + w / 2, b + h / 2], 1)
    return np.stack([a, b, a + w, b + h], 1)


def _one_frame(item, fmt):
    """-> list of box dicts for a single frame."""
    if isinstance(item, dict):
        item = [item]

    if isinstance(item, (list, tuple)) and len(item) and isinstance(item[0], dict):
        f = fmt if fmt != "auto" else "xywh_topleft"
        raw = [[d.get("x", 0), d.get("y", 0), d.get("width", 0), d.get("height", 0)] for d in item]
        xy = _conv(raw, f)
        return [
            {"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
             "label": str(d.get("label", "")), "score": float(d.get("score", 0.0)),
             "id": int(d["id"]) if d.get("id") is not None else -1}
            for b, d in zip(xy, item)
        ]

    if isinstance(item, torch.Tensor):
        f = fmt if fmt != "auto" else "xywh_center"          # ultralytics .xywh
        a = item.detach().cpu().float().numpy()
        if a.size == 0:
            return []
        xy = _conv(a, f)
        return [{"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                 "label": "", "score": 0.0, "id": -1} for b in xy]

    if isinstance(item, (list, tuple)):
        if len(item) == 0:
            return []
        f = fmt if fmt != "auto" else "xyxy"
        rows = [item] if isinstance(item[0], (int, float, np.floating)) else list(item)
        xy = _conv(rows, f)
        return [{"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                 "label": "", "score": 0.0, "id": -1} for b in xy]

    return []


def to_frames(boxes, fmt="auto"):
    """Anything -> [[box_dict, ...], ...] with one entry per frame."""
    if boxes is None:
        return []
    is_batch = (
        isinstance(boxes, (list, tuple))
        and len(boxes)
        and isinstance(boxes[0], (list, tuple, torch.Tensor))
        and not (isinstance(boxes[0], (list, tuple)) and len(boxes[0]) == 4
                 and isinstance(boxes[0][0], (int, float, np.floating)))
    )
    items = list(boxes) if is_batch else [boxes]
    return [_one_frame(it, fmt) for it in items]


def frame_at(frames, i):
    if not frames:
        return []
    return frames[i] if i < len(frames) else frames[-1]


# ------------------------------------------------------------------- export


def to_core(frames):
    """-> BOUNDING_BOX (core format). Keeps id so downstream nodes can select by it."""
    return [[{"x": b["x1"], "y": b["y1"], "width": b["x2"] - b["x1"], "height": b["y2"] - b["y1"],
              "label": b["label"], "score": b["score"], "id": b["id"]} for b in f] for f in frames]


def to_bbox(frames, equalise=True):
    """-> BBOX for SAM2. SAM2 asserts on frames with a different box count, so by
    default every frame is truncated to the smallest count in the batch."""
    out = [[[b["x1"], b["y1"], b["x2"], b["y2"]] for b in f] for f in frames]
    if equalise:
        n = min((len(f) for f in out), default=0)
        out = [f[:n] for f in out]
    return out


# ------------------------------------------------------------ sort / select


def sort_frame(f, sort_by):
    if sort_by == "detector":
        return f
    key = {
        "area": lambda b: -((b["x2"] - b["x1"]) * (b["y2"] - b["y1"])),
        "confidence": lambda b: -b["score"],
        "left_to_right": lambda b: b["x1"],
        "top_to_bottom": lambda b: b["y1"],
        "track_id": lambda b: (b["id"] if b["id"] >= 0 else 1 << 30),
    }[sort_by]
    return sorted(f, key=key)


def select_frame(f, mode, index=0, track_id=1, point=(0, 0), class_filter=""):
    if not f:
        return []
    wanted = [s.strip().lower() for s in class_filter.split(",") if s.strip()]
    if wanted:
        f = [b for b in f if b["label"].lower() in wanted]
        if not f:
            return []

    if mode == "all":
        return f
    if mode == "index":
        return [f[min(max(index, 0), len(f) - 1)]]
    if mode == "track_id":
        hit = [b for b in f if b["id"] == track_id]
        return hit[:1]
    if mode == "largest":
        return [max(f, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))]
    if mode == "smallest":
        return [min(f, key=lambda b: (b["x2"] - b["x1"]) * (b["y2"] - b["y1"]))]
    if mode == "highest_confidence":
        return [max(f, key=lambda b: b["score"])]
    if mode == "nearest_to_point":
        px, py = point
        return [min(f, key=lambda b: ((b["x1"] + b["x2"]) / 2 - px) ** 2
                                     + ((b["y1"] + b["y2"]) / 2 - py) ** 2)]
    return f


def pad_and_clip(f, pad=0, w=0, h=0):
    out = []
    for b in f:
        b = dict(b)
        b["x1"] -= pad
        b["y1"] -= pad
        b["x2"] += pad
        b["y2"] += pad
        if w and h:
            b["x1"] = float(np.clip(b["x1"], 0, w - 1))
            b["x2"] = float(np.clip(b["x2"], 0, w - 1))
            b["y1"] = float(np.clip(b["y1"], 0, h - 1))
            b["y2"] = float(np.clip(b["y2"], 0, h - 1))
        if b["x2"] > b["x1"] and b["y2"] > b["y1"]:
            out.append(b)
    return out


def fill_missing(frames, on_missing, w, h):
    """Frames with no box get filled, because SAM2 crashes on empty ones."""
    if on_missing == "empty":
        return frames
    full = [{"x1": 0.0, "y1": 0.0, "x2": float(w), "y2": float(h),
             "label": "frame", "score": 0.0, "id": -1}]
    out = []
    for f in frames:
        if f:
            out.append(f)
        elif on_missing == "previous" and out and out[-1]:
            out.append([dict(b) for b in out[-1]])
        else:
            out.append([dict(b) for b in full])
    if on_missing == "previous":
        nxt = next((f for f in out if f), None)
        if nxt:
            out = [f if f else [dict(b) for b in nxt] for f in out]
    return out


def hex_to_rgb(s):
    s = s.strip().lstrip("#")
    if len(s) == 3:
        s = "".join(c * 2 for c in s)
    if len(s) != 6:
        return (0, 255, 156)
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))
