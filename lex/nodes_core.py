"""Lex Bridge — дрібні перехідники між детекторами й SAM2/HUD на ComfyUI (Mac/MPS).

Нащо: у ComfyUI три різні "типи боксів", які між собою не стикуються —
  BOUNDING_BOX  (ядро ComfyUI, RTDETR/MediaPipe): список-на-кадр зі словників {x,y,width,height,label,score}
  BOXES         (пак ComfyUI-YOLO): тензор ultralytics xywh, де x,y — ЦЕНТР бокса
  BBOX          (kijai SAM2, KJNodes): список-на-кадр з [x1,y1,x2,y2]
Ці ноди зводять усе до BBOX (для SAM2) і малюють HUD-рамки.
"""

import numpy as np
import torch


from .boxes import AnyType, any_type, _to_xyxy_frames, _select, lab_text  # noqa: F401


# ---------------------------------------------------------------- nodes


class LexBoxesToBBOX:
    """Будь-які бокси -> BBOX для kijai Sam2Segmentation (по одному списку на кадр)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "BOUNDING_BOX (ядро), BOXES (ComfyUI-YOLO) або BBOX"}),
                "mode": (["largest", "all", "index"], {"default": "largest",
                         "tooltip": "largest = найбільший бокс у кадрі, КЛАС ІГНОРУЄТЬСЯ (рятує на CG, де YOLO каже 'kite')"}),
                "index": ("INT", {"default": 0, "min": 0, "max": 64}),
                "pad": ("INT", {"default": 0, "min": -256, "max": 256, "tooltip": "розширити бокс на N пікселів"}),
                "format_in": (["auto", "xyxy", "xywh_topleft", "xywh_center"], {"default": "auto"}),
                "on_miss": (["previous", "full_frame", "empty"], {"default": "previous",
                            "tooltip": "SAM2 падає, якщо на кадрі нема бокса — тримай 'previous'"}),
            },
            "optional": {"image": ("IMAGE", {"tooltip": "щоб обрізати бокси по межах кадру"})},
        }

    RETURN_TYPES = ("BBOX", "STRING")
    RETURN_NAMES = ("bboxes", "info")
    FUNCTION = "convert"
    CATEGORY = "Lex/bridge"

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
        info = f"кадрів: {len(out)} · з боксом: {hits} · по {n} бокс(ів)/кадр"
        print(f"[LexBoxesToBBOX] {info}")
        return (out, info)


class LexHUDBoxes:
    """HUD-рамки в стилі машинного зору: кути, підписи, товщина."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "boxes": (any_type,),
                "style": (["corners", "solid", "round"], {"default": "corners"}),
                "thickness": ("INT", {"default": 2, "min": 1, "max": 12}),
                "corner_len": ("INT", {"default": 24, "min": 4, "max": 200}),
                "color": ("STRING", {"default": "#00FF9C"}),
                "show_label": ("BOOLEAN", {"default": True}),
                "format_in": (["auto", "xyxy", "xywh_topleft", "xywh_center"], {"default": "auto"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "draw"
    CATEGORY = "Lex/bridge"

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
    """YOLO з models/yolov8 напряму на MPS. Віддає ОБИДВА формати боксів одразу.

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
                "images": ("IMAGE",),
                "model_name": (models or ["yolo11n.pt"],),
                "device": (["auto", "mps", "cpu", "cuda"], {"default": "auto",
                           "tooltip": "🔴 auto = сам бере cuda на орендованій NVIDIA і mps на Маку. Жорсткий 'mps' у збереженому воркфлоу впаде на чужій машині"}),
                "conf": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.95, "step": 0.01}),
                "imgsz": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32}),
                "class_filter": ("STRING", {"default": "", "tooltip": "порожньо = БЕЗ фільтра по класу (так треба для рендерів)"}),
                "max_boxes": ("INT", {"default": 10, "min": 1, "max": 100}),
                "on_miss": (["previous", "full_frame", "empty"], {"default": "previous",
                            "tooltip": "кадр без детекції: взяти бокс попереднього / весь кадр / лишити пустим. SAM2 падає на пустих кадрах — тримай 'previous'"}),
                "make_preview": ("BOOLEAN", {"default": False,
                                 "tooltip": "🔴 прев'ю = ПОВНА друга копія всіх кадрів у RAM (200 кадрів FullHD ≈ 5 ГБ). Вмикай на 1-2 кадрах для підбору conf, вимикай на прогоні"}),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX", "BBOX", "IMAGE", "STRING")
    RETURN_NAMES = ("bounding_box", "bbox_for_sam2", "preview", "info")
    FUNCTION = "detect"
    CATEGORY = "Lex/bridge"

    def detect(self, images, model_name, device, conf, imgsz, class_filter, max_boxes,
               on_miss="previous", make_preview=False):
        import os, folder_paths
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

        path = os.path.join(folder_paths.models_dir, "yolov8", model_name)
        if path not in self._cache:
            self._cache[path] = YOLO(path)
        model = self._cache[path]
        wanted = [s.strip().lower() for s in class_filter.split(",") if s.strip()]

        core, seen = [], 0
        for i in range(images.shape[0]):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            r = model.predict(frame, device=device, imgsz=imgsz, conf=conf, verbose=False)[0]
            xyxy = r.boxes.xyxy.cpu().numpy()
            names = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            scores = r.boxes.conf.cpu().numpy().tolist()
            rows = [{"x": float(b[0]), "y": float(b[1]), "width": float(b[2] - b[0]),
                     "height": float(b[3] - b[1]), "label": n, "score": float(s)}
                    for b, n, s in zip(xyxy, names, scores)
                    if not wanted or n.lower() in wanted]
            rows.sort(key=lambda d: d["width"] * d["height"], reverse=True)
            seen += 1 if rows else 0
            if not rows and on_miss != "empty":
                if on_miss == "previous" and core and core[-1]:
                    rows = [dict(d, label=d["label"] + "*") for d in core[-1]]
                else:
                    h_, w_ = images.shape[1], images.shape[2]
                    rows = [{"x": 0.0, "y": 0.0, "width": float(w_), "height": float(h_),
                             "label": "frame", "score": 0.0}]
            core.append(rows[:max_boxes])

        # кадри на початку могли лишитись пустими — заповнюємо першим знайденим
        if on_miss == "previous":
            nxt = next((f for f in core if f), None)
            if nxt:
                core = [f if f else [dict(d, label=d["label"] + "*") for d in nxt] for f in core]

        # SAM2 вимагає ОДНАКОВУ кількість боксів на кожному кадрі
        n = min((len(f) for f in core), default=0)
        bbox = [[[d["x"], d["y"], d["x"] + d["width"], d["y"] + d["height"]] for d in f[:n]] for f in core]
        labels = sorted({d["label"] for f in core for d in f})
        info = (f"кадрів {len(core)} · з детекцією {seen} · у BBOX по {n} бокс(ів)/кадр · "
                f"класи: {', '.join(labels) or '—'}"
                f"{'' if make_preview else ' · прев\'ю ВИМКНЕНО (RAM)'}")
        print(f"[LexYOLODetect] {info}")
        if make_preview:
            (prev,) = LexHUDBoxes().draw(images, core, "corners", 2, 28, "#00FF9C", True, "auto")
        else:
            prev = torch.zeros(1, 1, 1, 3)
        return (core, bbox, prev, info)


from .hud_sprite import LexHUDSprite
from .depth_curve import LexDepthCurve
