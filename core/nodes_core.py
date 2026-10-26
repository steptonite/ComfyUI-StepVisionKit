"""StepVisionKit core — дрібні перехідники між детекторами й SAM2/HUD на ComfyUI (Mac/MPS).

Нащо: у ComfyUI три різні "типи боксів", які між собою не стикуються —
  BOUNDING_BOX  (ядро ComfyUI, RTDETR/MediaPipe): список-на-кадр зі словників {x,y,width,height,label,score}
  BOXES         (пак ComfyUI-YOLO): тензор ultralytics xywh, де x,y — ЦЕНТР бокса
  BBOX          (kijai SAM2, KJNodes): список-на-кадр з [x1,y1,x2,y2]
Ці ноди зводять усе до BBOX (для SAM2) і малюють HUD-рамки.
"""

import numpy as np
import torch


from .boxes import AnyType, any_type, _to_xyxy_frames, _select, lab_text  # noqa: F401
from ..vk.boxes import ON_MISSING, SORT_BY  # спільні списки з VK-половини паку


# ---------------------------------------------------------------- nodes


class LexBoxesToBBOX:
    """Будь-які бокси -> BBOX для kijai Sam2Segmentation (по одному списку на кадр)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "BOUNDING_BOX (core), BOXES (ComfyUI-YOLO) or BBOX"}),
                "mode": (["largest", "all", "index"], {"default": "largest",
                         "tooltip": "largest = the biggest box in the frame, CLASS IS IGNORED (saves you on CG renders where YOLO says 'kite')"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 64, "tooltip": "box number for mode=index"}),
                "pad": ("INT", {"default": 0, "min": -256, "max": 256, "tooltip": "expand the box by N pixels"}),
                "format_in": (["auto", "xyxy", "xywh_topleft", "xywh_center"], {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
                "on_miss": (["previous", "full_frame", "empty"], {"default": "previous",
                            "tooltip": "SAM2 fails if a frame has no box — keep 'previous'"}),
            },
            "optional": {"image": ("IMAGE", {"tooltip": "clamp boxes to the frame bounds"})},
        }

    DESCRIPTION = "Turns detector boxes into one BBOX per frame for SAM2."
    RETURN_TYPES = ("BBOX", "STRING")
    RETURN_NAMES = ("bboxes", "info")
    FUNCTION = "convert"
    CATEGORY = "StepVisionKit"

    def convert(self, boxes, mode, index, pad, format_in, on_miss="previous", image=None):
        h = int(image.shape[1]) if image is not None else 0
        w = int(image.shape[2]) if image is not None else 0
        frames, labels = _to_xyxy_frames(boxes, format_in)
        out, hits = [], 0
        for arr, lab in zip(frames, labels):
            arr, _ = _select(arr, lab, mode, index, pad, w, h)
            hits += 1 if arr.shape[0] else 0
            rows = [[float(v) for v in b] for b in arr]
            if not rows and on_miss != "empty":
                if on_miss == "previous" and out and out[-1]:
                    rows = [list(b) for b in out[-1]]
                elif w and h:
                    rows = [[0.0, 0.0, float(w), float(h)]]
            out.append(rows)
        if on_miss == "previous":
            nxt = next((f for f in out if f), None)
            if nxt:
                out = [f if f else [list(b) for b in nxt] for f in out]
        n = min((len(f) for f in out), default=0)
        out = [f[:n] for f in out]
        info = f"frames: {len(out)} · with box: {hits} · {n} box(es)/frame"
        print(f"[LexBoxesToBBOX] {info}")
        return (out, info)


class LexHUDBoxes:
    """HUD-рамки в стилі машинного зору: кути, підписи, товщина."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "frames to draw on"}),
                "boxes": (any_type, {"tooltip": "BOUNDING_BOX / BOXES / BBOX from a detector"}),
                "style": (["corners", "solid", "round"], {"default": "corners", "tooltip": "corners = brackets on the corners, solid = full rectangle, round = rounded rectangle"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 12, "tooltip": "line width, px"}),
                "corner_len": ("INT", {"default": 24, "min": 4, "max": 200, "tooltip": "corners style: length of each bracket arm, px"}),
                "color": ("STRING", {"default": "#00FF9C", "tooltip": "line and text colour, hex"}),
                "show_label": ("BOOLEAN", {"default": True, "tooltip": "class name above each box"}),
                "format_in": (["auto", "xyxy", "xywh_topleft", "xywh_center"], {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
            },
        }

    DESCRIPTION = "Draws HUD boxes (corner brackets, solid or round) with class labels over the frames."
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "StepVisionKit"

    def draw(self, images, boxes, style, thickness, corner_len, color, show_label, format_in):
        import cv2

        c = color.lstrip("#")
        bgr = (int(c[4:6], 16), int(c[2:4], 16), int(c[0:2], 16))
        frames, labels = _to_xyxy_frames(boxes, format_in)
        out = []
        for i in range(images.shape[0]):
            img = (images[i].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1].copy()
            arr = frames[i] if i < len(frames) else (frames[0] if frames else np.zeros((0, 4)))
            lab = labels[i] if i < len(labels) else []
            for j, b in enumerate(arr):
                x1, y1, x2, y2 = [int(v) for v in b]
                if style == "solid":
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness)
                elif style == "round":
                    cv2.rectangle(img, (x1, y1), (x2, y2), bgr, thickness, cv2.LINE_AA)
                else:
                    L = min(corner_len, max(4, (x2 - x1) // 3), max(4, (y2 - y1) // 3))
                    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
                        cv2.line(img, (px, py), (px + dx * L, py), bgr, thickness, cv2.LINE_AA)
                        cv2.line(img, (px, py), (px, py + dy * L), bgr, thickness, cv2.LINE_AA)
                if show_label:
                    t = lab_text(lab[j], True) if j < len(lab) else ""
                    t = t or f"#{j}"
                    cv2.putText(img, t, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, bgr, max(1, thickness - 1), cv2.LINE_AA)
            out.append(torch.from_numpy(img[:, :, ::-1].copy().astype(np.float32) / 255.0))
        return (torch.stack(out),)


class LexYOLODetect:
    """YOLO + ByteTrack + вибір потрібного бокса + важіль плавності — усе в одній ноді.

    Зібрано 24.08.2026 з трьох місць (`LexYOLODetect` + `VKDetect` + `VKSelectBox`), щоб не
    тягнути ланцюг із трьох нод заради одного бокса. VK-ноди лишились на місці для старих
    воркфлоу, але для нових ця нода самодостатня.

    class_filter лишай порожнім: на CG-рендерах YOLO зве фігуру 'kite', і фільтр
    по 'person' викидає ~15 кадрів із 20 (заміряно 08.08).
    """

    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        import folder_paths, os
        d = os.path.join(folder_paths.models_dir, "yolov8")
        models = sorted([f for f in os.listdir(d) if f.endswith(".pt")]) if os.path.isdir(d) else []
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "frames to detect on"}),
                "model_name": (models or ["yolo11n.pt"], {"tooltip": "files in models/yolov8; yolo11n.pt is the small fast one"}),
                "device": (["auto", "mps", "cpu", "cuda"], {"default": "auto",
                           "tooltip": "auto = picks cuda on a rented NVIDIA box and mps on a Mac. A hard-coded 'mps' in a saved workflow will fail on someone else's machine"}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "lower finds more objects and more junk"}),
                "imgsz": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32, "tooltip": "detection resolution; bigger is slower and finds small objects"}),
                "class_filter": ("STRING", {"default": "", "tooltip": "empty = NO class filter (needed for renders)"}),
                "track": ("BOOLEAN", {"default": True,
                          "tooltip": "SMOOTHNESS LEVER #1. true = ByteTrack: every object gets a stable id, and an internal Kalman filter PREDICTS the position — the box moves smoothly and lags slightly behind sudden motion. false = raw per-frame YOLO: jittery, but shows exactly what the detector saw on THIS frame; there are no ids and select=track_id does not work"}),
                "smooth": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.95, "step": 0.05,
                           "tooltip": "SMOOTHNESS LEVER #2 — smoothing (EMA) on top of the result. 0.0 = untouched. 0.3-0.5 = noticeably calmer. 0.8+ = sluggish and laggy. Also works with track=false: removes part of the jitter without enabling Kalman"}),
                "sort_by": (SORT_BY, {"default": "area",
                            "tooltip": "how to rank the boxes — this sets the #0/#1 numbers in the preview. Warning: on video 'area' reshuffles the order whenever objects change size; use track_id for stability"}),
                "select": (["all", "center", "index", "track_id", "largest", "smallest",
                            "highest_confidence", "nearest_to_point"], {"default": "all",
                           "tooltip": "which box to pass on. center = closest to the frame center: the easiest way to reject a softbox/reflector at the side. track_id = stick to ONE object through the whole clip (needs track=true). index = take #N from the ranking"}),
                "pick": ("INT", {"default": 0, "min": 0, "max": 100000,
                         "tooltip": "number for select: with 'index' it is the #N from the preview, with 'track_id' it is the id:N from the preview. Unused in other modes"}),
                "point_x": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "only for select=nearest_to_point"}),
                "point_y": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "only for select=nearest_to_point"}),
                "max_boxes": ("INT", {"default": 10, "min": 1, "max": 100,
                              "tooltip": "how many boxes to keep AFTER selection. Usually 1 for SAM2"}),
                "pad": ("INT", {"default": 0, "min": -512, "max": 512,
                        "tooltip": "grow (+) or shrink (−) the box by N pixels. A little '+' helps SAM2 not cut off hair at the box edge"}),
                "on_miss": (ON_MISSING, {"default": "previous",
                            "tooltip": "frame with no detection: reuse the previous box / use the full frame / leave empty. SAM2 fails on empty frames — keep 'previous'"}),
                "make_preview": ("BOOLEAN", {"default": False,
                                 "tooltip": "Warning: the preview is a FULL second copy of all frames in RAM (200 FullHD frames ≈ 5 GB). Enable it on 1-2 frames to read #N and id:N, then disable it for the real run"}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    DESCRIPTION = "Runs a YOLO model from models/yolov8 on every frame, with tracking and smoothing, and picks which boxes to pass on."
    RETURN_TYPES = ("BOUNDING_BOX", "BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bounding_box", "bbox_for_sam2", "preview", "info")
    FUNCTION = "detect"
    CATEGORY = "StepVisionKit"

    def detect(self, images, model_name, device, conf, imgsz, class_filter, max_boxes,
               on_miss="previous", make_preview=False, track=True, smooth=0.0,
               sort_by="area", select="all", pick=0, point_x=0, point_y=0, pad=0, unique_id=None):
        import os
        import folder_paths
        import torch
        from ultralytics import YOLO
        from ..vk.boxes import (fill_missing, pad_and_clip, preview_frames, preview_payload, select_frame,
                                sort_frame, to_bbox, to_core)

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

        path = os.path.join(folder_paths.models_dir, "yolov8", model_name)
        if path not in self._cache:
            self._cache[path] = YOLO(path)
        model = self._cache[path]

        if track:
            # стан трекера живе на об'єкті моделі — без ресету id течуть між прогонами
            try:
                for t in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                    t.reset()
            except Exception as e:                                   # noqa: BLE001
                print(f"[LexYOLODetect] tracker reset skipped: {e}")

        wanted = [s.strip().lower() for s in class_filter.split(",") if s.strip()]
        H, W = int(images.shape[1]), int(images.shape[2])
        mode = "nearest_to_point" if select == "center" else select
        point = (W / 2.0, H / 2.0) if select == "center" else (float(point_x), float(point_y))
        a = float(smooth)

        frames, ranked, ids, seen = [], [], set(), 0
        carry = {}
        shots, shot_at = [], preview_frames(images.shape[0])

        for i in range(images.shape[0]):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            if track:
                r = model.track(frame, device=device, imgsz=imgsz, conf=conf,
                                persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            else:
                r = model.predict(frame, device=device, imgsz=imgsz, conf=conf, verbose=False)[0]

            xyxy = r.boxes.xyxy.cpu().numpy()
            names = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            scores = r.boxes.conf.cpu().numpy().tolist()
            tids = (r.boxes.id.cpu().numpy().astype(int).tolist()
                    if getattr(r.boxes, "id", None) is not None else [-1] * len(names))

            raw = [{"x1": float(b[0]), "y1": float(b[1]), "x2": float(b[2]), "y2": float(b[3]),
                    "label": nm, "score": float(sc), "id": int(tid)}
                   for b, nm, sc, tid in zip(xyxy, names, scores, tids)]
            if i in shot_at:
                shots.append((i, np.ascontiguousarray(frame[:, :, ::-1]), raw))
            f = [b for b in raw if not wanted or b["label"].lower() in wanted]

            f = sort_frame(f, sort_by)
            seen += 1 if f else 0
            ids.update(b["id"] for b in f if b["id"] >= 0)

            # превʼю малює ПОВНИЙ рейтинг з номерами — щоб було по чому обрати pick
            ranked.append([dict(b, label="#%d %s%s" % (j, b["label"],
                                (" id:%d" % b["id"]) if b["id"] >= 0 else ""))
                           for j, b in enumerate(f[:10])])

            f = select_frame(f, mode, index=pick, track_id=pick, point=point)[:max_boxes]
            f = pad_and_clip(f, pad, W, H)

            if a > 0.0:
                for k, b in enumerate(f):
                    key = b["id"] if (track and b["id"] >= 0) else k
                    prev_xy = carry.get(key)
                    if prev_xy is not None:
                        for j, c in enumerate(("x1", "y1", "x2", "y2")):
                            b[c] = prev_xy[j] * a + b[c] * (1.0 - a)
                    carry[key] = [b["x1"], b["y1"], b["x2"], b["y2"]]

            frames.append(f)

        frames = fill_missing(frames, on_miss, W, H)
        core = to_core(frames)
        bbox = to_bbox(frames, equalise=True)
        n = min((len(fr) for fr in bbox), default=0)
        labels = sorted({b["label"] for fr in frames for b in fr})
        arg = " (%d)" % pick if select in ("index", "track_id") else ""
        trk = "ByteTrack" if track else "OFF (raw YOLO)"
        sm = " + smoothing %.2f" % a if a > 0 else ""
        tail = "" if make_preview else " · preview OFF (RAM)"
        info = ("frames %d · with detection %d · BBOX %d box(es)/frame · "
                "rank %s · select %s%s · track %s%s · id: %s · classes: %s%s"
                % (len(frames), seen, n, sort_by, select, arg, trk, sm,
                   sorted(ids) if ids else "—", ", ".join(labels) or "—", tail))
        print(f"[LexYOLODetect] {info}")
        if make_preview:
            (prev_img,) = LexHUDBoxes().draw(images, to_core(ranked) if ranked else core,
                                             "corners", 2, 28, "#00FF9C", True, "auto")
        else:
            prev_img = torch.zeros(1, 1, 1, 3)
        return {"ui": {"svk_yolo": [preview_payload(shots, W, H, conf, unique_id)]},
                "result": (core, bbox, prev_img, info)}


from .hud_sprite import LexHUDSprite
from .depth_curve import LexDepthCurve
