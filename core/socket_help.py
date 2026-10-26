"""Підказки до ВИХОДІВ нод (OUTPUT_TOOLTIPS): що несе кожен сокет і куди його втикати.

Живе окремим файлом, щоб не чіпати класи нод: __init__.py паку дописує їх класам, у яких своїх немає.
Порядок і кількість = RETURN_TYPES ноди (перевіряє аудит). Українські версії - web/i18n/uk.json.
Коротко, одне речення, ≤140 символів: «що несе → куди підключити».
"""

OUTPUT_TOOLTIPS = {
    # ---- Depth Look
    "DepthLook": [
        "Frames with the look → Preview Image. Stays blocked in tuner mode until a render runs (auto_run or Render final).",
        "The look settings (DL_GRADE) → grade input of Depth Look · Video or another Depth Look node.",
        "Full-size frames with the look → Save Image, Save Video · HDR, Video Combine or Camera Realism.",
    ],
    "DepthLookSimple": [
        "Frames with the look → Preview Image, Save Image, Video Combine, Save Video · HDR or Camera Realism.",
        "The look settings (DL_GRADE) → grade input of Depth Look · Full or Depth Look · Video.",
    ],
    "DepthLookVideo": [
        "Fast preview: every Nth frame with a PREVIEW mark → Video Combine at play_fps. Not for the final file.",
        "Frame rate to play the preview (fps ÷ preview_every) → frame_rate of Video Combine. In final quality = fps.",
        "Full-size animated clip → Video Combine, Save Video · HDR or Save Image. Needs Render final (or auto_render).",
    ],
    # ---- Детекція / бокси
    "LexYOLODetect": [
        "Chosen boxes → Draw Boxes, Overlay On Boxes, HUD Sprite, HUD Boxes or Boxes → BBOX.",
        "The same boxes as BBOX, equal count on every frame → Sam2Segmentation (bboxes input).",
        "Empty (1×1) unless «Preview output» (make_preview) is on under «more». The card already shows the boxes.",
        "Text report: frames, detections, track ids, classes → Preview Any. Safe to leave unconnected.",
    ],
    "VKDetect": [
        "All detected boxes with track ids → Select Box, Draw Boxes, Overlay On Boxes or Boxes → BBOX.",
        "The same boxes as BBOX, equal count on every frame → Sam2Segmentation (bboxes input).",
        "Number of tracked objects (boxes per frame if track is off) → just a number for Preview Any or logic.",
        "Text report: frames, detections, track ids, classes → Preview Any. Safe to leave unconnected.",
    ],
    "VKSelectBox": [
        "Boxes of the chosen object → Draw Boxes, Overlay On Boxes, HUD Sprite or Boxes → BBOX.",
        "The same boxes as BBOX, equal count on every frame → Sam2Segmentation (bboxes input).",
        "Text report: matched frames, mode, boxes per frame → Preview Any. Safe to leave unconnected.",
    ],
    "VKBoxesToBBOX": [
        "Boxes of any detector as BBOX, one list per frame → Sam2Segmentation (bboxes input).",
        "Text report: frames and boxes per frame → Preview Any. Safe to leave unconnected.",
    ],
    "LexBoxesToBBOX": [
        "One BBOX list per frame → Sam2Segmentation (bboxes input).",
        "Text report: frames, frames with a box, boxes per frame → Preview Any. Safe to leave unconnected.",
    ],
    # ---- HUD
    "LexHUDBoxes": [
        "Frames with HUD boxes drawn on top → Preview Image, Video Combine, Save Image or Save Video · HDR.",
    ],
    "VKDrawBoxes": [
        "Frames with the boxes drawn on top → Preview Image, Video Combine, Save Image or Save Video · HDR.",
    ],
    "VKOverlayOnBoxes": [
        "Frames with your graphic stamped on every box → Preview Image, Video Combine or Save Image.",
        "Where the graphic landed (0..1) → mask for blur, glow or grading of just the HUD.",
        "Text report: frames, stamps, fit, anchor → Preview Any. Safe to leave unconnected.",
    ],
    "LexHUDSprite": [
        "Video with the HUD frame stamped on and the hole cut → Video Combine or Save Image. 1×1 if emit = hud_only.",
        "Alpha of composite, 0 in the cut hole → Join Image With Alpha or the mask of Save Image · HDR.",
        "The HUD frame alone (RGB) → layer it over other footage. 1×1 unless emit = hud_only or both.",
        "Alpha of hud_rgb → Join Image With Alpha, or a mask for compositing. 1×1 unless emit = hud_only or both.",
        "White fill of the YOLO box (what hole = box cuts) → Cut By Mask or SAM2. Needs emit_box_mask.",
        "Text report: frames, HUD drawn, cutout, output size → Preview Any. Safe to leave unconnected.",
    ],
    "VKCutByMask": [
        "Frames with the background replaced → Preview Image, Save Image or Video Combine.",
        "The mask actually used (after grow, feather, invert) → Join Image With Alpha for a transparent background.",
    ],
    # ---- Глибина
    "LexDepthCurve": [
        "Re-toned depth map → Depth Grade, Depth Look (depth input) or Save Depth · 16-bit / EXR.",
        "The same depth as a single-channel MASK → depth-based masking, blur or compositing.",
        "Text report: levels, gamma, tones of the result → Preview Any. Safe to leave unconnected.",
    ],
    "SVKDepthGrade": [
        "Graded, flicker-free depth map → Depth Look (depth input), Depth Grade View, Save Depth or ControlNet depth.",
        "The same depth as a single-channel MASK → depth-based masking, blur or compositing.",
        "Text report: what was applied to the depth → Preview Any. Safe to leave unconnected.",
    ],
    "SVKDepthGradeLive": [
        "Graded, flicker-free depth map → Depth Look (depth input), Depth Grade View, Save Depth or ControlNet depth.",
        "The same depth as a single-channel MASK → depth-based masking, blur or compositing.",
        "Text report: what was applied to the depth → Preview Any. Safe to leave unconnected.",
    ],
    "SVKDepthGradeView": [
        "The same depth passed through unchanged → next node (Depth Look, Save Depth). The chain stays unbroken.",
    ],
    "SVKDepthGradeVideoView": [
        "The same depth passed through unchanged → next node (Depth Look, Save Depth). The chain stays unbroken.",
    ],
    "SVKDepthReceiver": [
        "Depth image (float32, near = white, full model range) → depth input of Depth Grade.",
    ],
    "SVKRawDepthLoad": [
        "Raw DA3 geometry (float32) → Depth Receiver, or DA3 mesh / point-cloud nodes.",
    ],
    # ---- Камера й IO
    "SVKCameraRealism": [
        "Frames with the camera look (noise, optics, tone) → Save Image, Save Image · HDR or Video Combine.",
    ],
    "SVKLoadImageSequence": [
        "Sequence frames as float, values above 1.0 kept → Depth Look, Camera Realism, YOLO Detect or any image node.",
        "Alpha of the frames, 1 = opaque (all ones if files have none) → Save Video · HDR (alpha) or Join Image With Alpha.",
        "Number of frames loaded → any INT input or Preview Any.",
        "Text report: frame count, size, first and last file → Preview Any. Safe to leave unconnected.",
    ],
    "SVKSaveVideoHDR": [
        "Full path of the written file → Preview Any, or a node that takes a file path.",
    ],
}
