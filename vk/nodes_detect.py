"""Detect Objects (YOLO) — detection with optional tracking on Apple Silicon."""

import numpy as np

from .boxes import (ON_MISSING, SORT_BY, fill_missing, sort_frame, to_bbox, to_core)


class VKDetect:
    _cache = {}

    @classmethod
    def INPUT_TYPES(cls):
        import os

        import folder_paths
        d = os.path.join(folder_paths.models_dir, "yolov8")
        models = sorted(f for f in os.listdir(d) if f.endswith(".pt")) if os.path.isdir(d) else []
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "frames to detect on"}),
                "model_name": (models or ["yolo11n.pt"], {"tooltip": "Files in models/yolov8. yolo11n.pt is the small fast one."}),
                "device": (["auto", "mps", "cpu", "cuda"], {"default": "auto",
                           "tooltip": "auto picks cuda on a rented NVIDIA box and mps on a Mac."}),
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.01, "max": 0.95, "step": 0.01, "tooltip": "Lower finds more objects and more junk."}),
                "image_size": ("INT", {"default": 640, "min": 320, "max": 1280, "step": 32, "tooltip": "Detection resolution. Bigger is slower and finds small objects."}),
                "classes": ("STRING", {"default": "", "tooltip": "Comma-separated class names to keep, e.g. person,car. LEAVE EMPTY for rendered or CG footage — YOLO often calls a CG figure 'kite' and a filter would drop most frames."}),
                "max_objects": ("INT", {"default": 10, "min": 1, "max": 100, "tooltip": "keep at most this many boxes per frame"}),
                "track": ("BOOLEAN", {"default": True, "tooltip": "Give every object an id that stays the same across the whole video, so Select Box can follow one object. Off = plain per-frame detection."}),
                "sort_by": (SORT_BY, {"default": "area", "tooltip": "What decides the #index order. On video, area re-shuffles the order whenever objects change size — use a track id to follow one object instead."}),
                "on_missing": (ON_MISSING, {"default": "previous", "tooltip": "Frames with no detection. SAM2 crashes on an empty frame, so keep 'previous' when feeding it."}),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX", "BBOX", "INT", "STRING")
    RETURN_NAMES = ("boxes", "bbox_for_sam2", "count", "info")
    FUNCTION = "detect"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Find objects in every frame. Feed the boxes to Draw Boxes, Overlay On Boxes, Select Box or SAM2."

    def detect(self, images, model_name, device, confidence, image_size, classes,
               max_objects, track, sort_by, on_missing):
        import os

        import folder_paths
        import torch
        from ultralytics import YOLO

        if device == "auto":
            device = ("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available() else "cpu")

        path = os.path.join(folder_paths.models_dir, "yolov8", model_name)
        if path not in self._cache:
            self._cache[path] = YOLO(path)
        model = self._cache[path]

        # Tracker state lives on the model object; without this, ids leak between runs.
        if track:
            try:
                for t in getattr(getattr(model, "predictor", None), "trackers", []) or []:
                    t.reset()
            except Exception as e:                                  # noqa: BLE001
                print(f"[VKDetect] tracker reset skipped: {e}")

        wanted = [s.strip().lower() for s in classes.split(",") if s.strip()]
        H, W = int(images.shape[1]), int(images.shape[2])
        frames, seen, ids = [], 0, set()

        for i in range(images.shape[0]):
            frame = (images[i].cpu().numpy() * 255).astype(np.uint8)[:, :, ::-1]
            if track:
                r = model.track(frame, device=device, imgsz=image_size, conf=confidence,
                                persist=True, tracker="bytetrack.yaml", verbose=False)[0]
            else:
                r = model.predict(frame, device=device, imgsz=image_size, conf=confidence,
                                  verbose=False)[0]

            xyxy = r.boxes.xyxy.cpu().numpy()
            names = [r.names[int(c)] for c in r.boxes.cls.cpu().numpy()]
            scores = r.boxes.conf.cpu().numpy().tolist()
            tids = (r.boxes.id.cpu().numpy().astype(int).tolist()
                    if getattr(r.boxes, "id", None) is not None else [-1] * len(names))

            f = [{"x1": float(bx[0]), "y1": float(bx[1]), "x2": float(bx[2]), "y2": float(bx[3]),
                  "label": nm, "score": float(sc), "id": int(tid)}
                 for bx, nm, sc, tid in zip(xyxy, names, scores, tids)
                 if not wanted or nm.lower() in wanted]

            f = sort_frame(f, sort_by)[:max_objects]
            ids.update(b["id"] for b in f if b["id"] >= 0)
            seen += 1 if f else 0
            frames.append(f)

        frames = fill_missing(frames, on_missing, W, H)
        bbox = to_bbox(frames, equalise=True)
        per = min((len(f) for f in bbox), default=0)
        labels = sorted({b["label"] for f in frames for b in f})
        info = (f"frames {len(frames)} · with detection {seen} · "
                f"{per} box(es)/frame for SAM2 · "
                f"track ids: {sorted(ids) if ids else 'off'} · classes: {', '.join(labels) or '-'}")
        print(f"[VKDetect] {info}")
        return (to_core(frames), bbox, len(ids) or per, info)


NODES = {"VKDetect": VKDetect}
DISPLAY = {"VKDetect": "Detect Objects · YOLO"}
