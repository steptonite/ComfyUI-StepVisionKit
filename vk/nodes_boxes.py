"""Select Box / Boxes To BBOX — pick which detected object the rest of the graph works on."""

from .boxes import (FORMATS, ON_MISSING, any_type, fill_missing, pad_and_clip,
                    select_frame, to_bbox, to_core, to_frames)

MODE = ["all", "index", "track_id", "largest", "smallest", "highest_confidence",
        "nearest_to_point"]


class VKSelectBox:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "Boxes from Detect Objects (boxes output) or any detector."}),
                "mode": (MODE, {"default": "largest", "tooltip": "track_id is the only mode that stays on the SAME object through a video. index re-shuffles whenever the sort order changes."}),
                "index": ("INT", {"default": 0, "min": 0, "max": 200, "tooltip": "For mode=index. Read the #N printed on the preview from Draw Boxes."}),
                "track_id": ("INT", {"default": 1, "min": 0, "max": 100000, "tooltip": "For mode=track_id. Read the id:N printed on the preview."}),
                "point_x": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "for mode=nearest_to_point: x in frame pixels"}),
                "point_y": ("INT", {"default": 0, "min": 0, "max": 16384, "tooltip": "for mode=nearest_to_point: y in frame pixels"}),
                "class_filter": ("STRING", {"default": "", "tooltip": "Keep only these classes, comma-separated. Empty = all."}),
                "pad": ("INT", {"default": 0, "min": -512, "max": 512, "tooltip": "Grow (+) or shrink (-) the box by N pixels."}),
                "format_in": (FORMATS, {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
                "on_missing": (ON_MISSING, {"default": "previous", "tooltip": "frame with no box: previous = repeat the last box, full_frame = whole frame, empty = no box (SAM2 fails on an empty frame)"}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Optional: the frames the boxes were found on (Load Video · Pro) so boxes get clipped to the frame edges."}),
            },
        }

    RETURN_TYPES = ("BOUNDING_BOX", "BBOX", "STRING")
    RETURN_NAMES = ("boxes", "bbox_for_sam2", "info")
    FUNCTION = "select"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Pick one object out of many. Cheap to re-run — change the rule without re-running detection."

    def select(self, boxes, mode, index, track_id, point_x, point_y, class_filter, pad,
               format_in, on_missing, image=None):
        H = int(image.shape[1]) if image is not None else 0
        W = int(image.shape[2]) if image is not None else 0
        frames = to_frames(boxes, format_in)
        picked, hits = [], 0
        for f in frames:
            sel = select_frame(f, mode, index, track_id, (point_x, point_y), class_filter)
            sel = pad_and_clip(sel, pad, W, H)
            hits += 1 if sel else 0
            picked.append(sel)
        picked = fill_missing(picked, on_missing, W or 512, H or 512)
        bbox = to_bbox(picked, equalise=True)
        per = min((len(f) for f in bbox), default=0)
        info = f"frames {len(picked)} · matched {hits} · {per} box(es)/frame · mode {mode}"
        if mode == "track_id" and hits == 0:
            info += f" · NO object with id {track_id} — check the preview"
        print(f"[VKSelectBox] {info}")
        return (to_core(picked), bbox, info)


class VKBoxesToBBOX:
    """Thin adapter: any box type -> BBOX for SAM2. For detectors other than VKDetect."""
    DEPRECATED = True   # сховано з меню 27.10.2026, старі воркфлоу відкриваються; замість неї: Select Box

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "boxes": (any_type, {"tooltip": "Boxes from Detect Objects, Select Box or any detector (RT-DETR, SAM3, MediaPipe, YOLO)"}),
                "format_in": (FORMATS, {"default": "auto", "tooltip": "box layout of the input: auto detects it; xyxy = corners, xywh_topleft / xywh_center = size from the top-left corner / centre"}),
                "on_missing": (ON_MISSING, {"default": "previous", "tooltip": "frame with no box: previous = repeat the last box, full_frame = whole frame, empty = no box (SAM2 fails on an empty frame)"}),
            },
            "optional": {"image": ("IMAGE", {"tooltip": "Optional: the frames the boxes were found on (Load Video · Pro) so boxes get clipped to the frame edges"})},
        }

    RETURN_TYPES = ("BBOX", "STRING")
    RETURN_NAMES = ("bbox_for_sam2", "info")
    FUNCTION = "convert"
    CATEGORY = "VisionKit"
    DESCRIPTION = "Convert boxes from any detector (RT-DETR, SAM3, MediaPipe, ComfyUI-YOLO) into the BBOX type SAM2 wants."

    def convert(self, boxes, format_in, on_missing, image=None):
        H = int(image.shape[1]) if image is not None else 512
        W = int(image.shape[2]) if image is not None else 512
        frames = fill_missing(to_frames(boxes, format_in), on_missing, W, H)
        bbox = to_bbox(frames, equalise=True)
        per = min((len(f) for f in bbox), default=0)
        info = f"frames {len(bbox)} · {per} box(es)/frame"
        print(f"[VKBoxesToBBOX] {info}")
        return (bbox, info)


NODES = {"VKSelectBox": VKSelectBox, "VKBoxesToBBOX": VKBoxesToBBOX}
DISPLAY = {"VKSelectBox": "Select Box", "VKBoxesToBBOX": "Boxes → BBOX"}
