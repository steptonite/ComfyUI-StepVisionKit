🇺🇦 Українська версія: [README.uk.md](README.uk.md)

# StepVisionKit

Detection, tracking, HUD compositing and depth curves for **video** in ComfyUI.
Written for Mac/MPS, also runs on CUDA (`device = auto`).

Menu category: **`StepVisionKit`** → `Detect` · `HUD` · `Depth`.

## What's inside

| node | purpose |
|---|---|
| `YOLO Detect` | ultralytics called directly, returns boxes in two formats at once. No tracking — fast |
| `Detect Objects · YOLO` | the same + **ByteTrack**: returns a stable `track_id` through the whole video |
| `Select Box` | pick ONE object — by `track_id`, index, point or class |
| `Boxes → BBOX · SAM2` | convert any box format to `BBOX` for SAM2 |
| `HUD Boxes · corners` | a corner-bracket frame drawn in code |
| `HUD Sprite · PNG frame` | **your own PNG frame** follows the box; 9-slice corners, a cutout in the video, separate alpha |
| `Overlay On Boxes` | your own graphic on EVERY box, selectable by `track_id` |
| `Cut By Mask` | cut by a mask to black/white/transparent/blur/color, batched |
| `Depth Curve · DA3 → V2` | Depth Anything 3 → V2 look: Levels, clarity, deband. Does not run a model |
| `Depth Grade` and `Depth Grade View` | depth grading with anti-flicker, plus frame/video views for calibration |
| `Depth Receiver`, `Raw Depth Save` / `Load`, `Save Depth · 16-bit / EXR` | raw DA3 depth to float depth, EXR/16-bit I/O |
| `Depth Look · Full / Simple / Video` | lens blur by depth, palette, silhouette, haze, trail, halation, grain (see [depth_look/README.md](depth_look/README.md)) |

## Installation

**ComfyUI-Manager / Comfy Registry** — search for `StepVisionKit` in the Manager, or from the command line:

```bash
comfy node install stepvisionkit
```

**Manually:**

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/steptonite/ComfyUI-StepVisionKit
pip install -r ComfyUI-StepVisionKit/requirements.txt
```

Then restart ComfyUI.

YOLO weights go into `ComfyUI/models/yolov8/` (`face_yolov8n.pt`, `person_yolov8m-seg.pt`,
`yolo11n.pt` — whatever you download shows up in the dropdown). You do not need to install `torch`, ComfyUI already has it.

> **Naming history.** This used to live as two packs — `comfyui-lex-bridge` and `comfyui-visionkit` —
> which registered some keys IDENTICALLY: whichever loaded later alphabetically won,
> and the workflow silently picked up the wrong node. Now it is one pack. The internal keys (`Lex*` / `VK*`)
> are deliberately kept historical: ComfyUI looks a node up in the saved json by its `type` string, and renaming
> would turn every previously saved workflow red.

## Language

Node texts (names, tooltips, messages) are English by default. Ukrainian can be switched on in
**Settings → StepVisionKit → Language → Node language** (reload the page).

---

## Before anything else: RAM arithmetic (8 GB machine)

One **full batch of frames** in ComfyUI sits in RAM as float32:

```
frames × height × width × 3 channels × 4 bytes
```

| what | 50 FullHD frames | 200 FullHD frames |
|---|---|---|
| one IMAGE batch | **1.24 GB** | **4.98 GB** |
| one MASK batch (1 channel) | 0.41 GB | 1.66 GB |

**Every enabled output is one more such batch.** So:

- `LexYOLODetect.make_preview = False` for the real run (otherwise +5 GB per 200 frames);
- `LexHUDSprite.emit = composite` (not `both`), `emit_box_mask = False`;
- turn off preview nodes (`PreviewImage`, the preview inside `VHS_VideoCombine`) — they eat
  not only Python memory but also the **browser**;
- 50 frames at a time via `Meta Batch Manager` is a workable size (tested 10.08).

Note: ComfyUI 0.30 starts with **`Using RAM pressure cache`**. Under memory pressure it
**evicts cached node results**. This is why you sometimes just press Save Output
and it recomputes everything from scratch — the cache had already been dropped. Fix it at startup with
`--cache-classic` (aggressive cache, more RAM) or `--cache-lru N`. On 8 GB it is easier to accept this
and enable `save_output` BEFORE the run.

---

## Node `YOLO Detect` — `LexYOLODetect`

Runs ultralytics directly (e.g. on MPS) and returns boxes in two formats at once.

| parameter | what it does | how to set it |
|---|---|---|
| `images` | input batch of frames | from `Load Video` / `Meta Batch` |
| `model_name` | file from `models/yolov8/` | `face_yolov8m.pt` — faces, `hand_yolov8s.pt` — hands, `person_yolov8m-seg.pt` — people, `yolo11n.pt` — 80 COCO classes |
| `device` | `mps` / `cpu` (`auto` picks cuda or mps) | `mps` |
| `conf` | confidence threshold 0.01–0.95 | 0.25 default. Catching extras → raise to 0.4. Losing frames → 0.15 |
| `imgsz` | size the frame is shrunk to BEFORE detection | 640 default. **Important:** it does **not** affect the quality of the output video — it is for the detector only. Small face in the frame → 960/1280 |
| `class_filter` | filter by class name, comma-separated | **Keep it EMPTY.** On CG/renders YOLO calls the figure a `kite`, and a `person` filter throws away ~15 of 20 frames (measured 08.08) |
| `max_boxes` | how many boxes to keep per frame | 1 if there is a single subject |
| `on_miss` | what to do on a frame with no detection | `previous` — reuse the previous frame's box (SAM2 fails on empty frames); `full_frame` — the whole frame; `empty` — leave empty |
| `make_preview` | draw an image with the boxes | **Important:** `False` for the real run. `True` only while tuning `conf` on 1–2 frames |

**Outputs:** `bounding_box` (core format, for the HUD nodes) · `bbox_for_sam2` (for kijai SAM2) ·
`preview` (empty 1×1 if `make_preview=False`) · `info` (a statistics string).

---

## Node `HUD Sprite · PNG frame` — `LexHUDSprite`

Takes an **external PNG** with a frame and places it on the box, following the object.

### Where to put the frame

| parameter | what it does |
|---|---|
| `boxes` | boxes from `LexYOLODetect` (the `bounding_box` output) |
| `mode` | `largest` — the biggest box in the frame (class ignored) · `index` — a specific one by number · `all` — a frame on EVERY box |
| `index` | box number for `mode=index` |
| `offset_x` / `offset_y` | shift the frame by hand, in pixels |

### Size and aspect

| parameter | what it does |
|---|---|
| `margin_pct` | gap between the box and the frame, in % of the box's average size. **Equal on all four sides** — on a vertical face the frame wraps evenly. Default 25 |
| `margin_px` | the same gap in pixels; added on top of `margin_pct` |
| `aspect` | a dropdown, not text. **`box` (default)** = the frame takes the shape of the YOLO box itself: a vertical face gives a vertical frame, a hand caught as a square gives a square frame. **`png`** = the aspect of the frame file itself; the sprite scales uniformly and is not deformed at all. The rest (`1:1`, `4:5`, `9:16`, `16:9`…) force a ratio: the shorter side is **extended**, the object is never cropped |
| `min_width` / `min_height` | lower bound for the HUD size in frame pixels — so the frame does not become microscopic on tiny boxes. 0 = no limit |

Note: if the frame looks offset on a **moving** object, it is not the gap but `smooth`:
the EMA lags behind the box. If you want it strictly on the box — `smooth = 0`.
**Important:** with `aspect = box` a 3:2 sprite is squeezed into the box shape, so `nine_slice` is **mandatory** here
(otherwise the corner ornament will smear).

### Quality and behavior

| parameter | what it does |
|---|---|
| `smooth` | EMA smoothing of the frame's motion. 0 = raw box (jitters frame by frame) · 0.5 default · 0.7–0.8 = very smooth but lags on fast motion. Has no effect in `mode=all` |
| `nine_slice` | px of the corners that are **not stretched**. **Important:** measured in pixels of the **source PNG**, not of the frame — the node scales the corner down together with the sprite. Set it to roughly the size of the corner bracket: for a 1536×1024 PNG with ~230 px brackets — **240** |
| `alpha_mode` | where transparency comes from: `loadimage` (default — the MASK from `LoadImage` is **inverted**, the node accounts for it) · `mask_direct` (the MASK already is the alpha) · `luma` (brightness as alpha — for black-and-white frames without an alpha channel) · `opaque` (no transparency) |
| `tint` | recolor the frame. **Press the color-picker button under the field — the system eyedropper opens** and the hex lands in the field by itself; the clear button resets it. You can also type `#RRGGBB` by hand, and the field accepts a link — so you can hook up a `ColorPicker`. The artwork's brightness is preserved and only the hue is replaced: no need to regenerate the PNG for a different color |
| `tint_strength` | 0 = original, 1 = fully `tint`. In between = blend |

### Class label

| parameter | what it does |
|---|---|
| `label` | **`auto` (default)** = the class from the detector (`face`, `hand`, `person`) — like standard YOLO. Empty = no label. Any other text is written verbatim |
| `label_scale` | size relative to the frame height. 1.0 default, 0.6 if too big |
| `label_conf` | append the confidence: `face 0.91` |

The label color = `tint`, and if `tint` is empty, the average color of the frame itself.
The label sits above the frame; if it does not fit there (object at the top of the image) it moves inside.
In `hud_only` the label is also drawn into the alpha, so it is present on a transparent sequence.

If you want the frame NOT to deform at all, set `aspect = png`: the node takes the
proportions of the loaded file. Then the sprite scales uniformly and `nine_slice` is hardly needed,
but the frame will no longer follow the shape of the box.

### Cutout in the video

| parameter | what it does |
|---|---|
| `hole` | `none` — leave the video alone · `box` — cut out the YOLO rectangle itself · `hud_rect` — cut out the whole HUD rectangle |
| `hole_feather` | blur the cutout edge, px. 0 = hard |

### Outputs and RAM

| parameter | what it does |
|---|---|
| `emit` | `composite` (default) — compute only the composited result · `hud_only` — only the HUD on transparent · `both` — both, **twice the RAM** |
| `emit_box_mask` | also compute the mask of the rectangle. `False` for the real run |

**Outputs:** `composite` + `composite_alpha` (video with the cutout and frame) ·
`hud_rgb` + `hud_alpha` (the frame alone on transparent, at video size) ·
`box_mask` · `info` (reports how many GB of outputs it computes).

Disabled outputs return a 1×1 placeholder — do not connect them anywhere.

---

## Saving with transparency

```
hud_rgb ─┐
hud_alpha → InvertMask → JoinImageWithAlpha → VHS_VideoCombine
```

**Important: `InvertMask` is mandatory.** The core `JoinImageWithAlpha` node itself does `alpha = 1.0 - mask`
(`comfy_extras/nodes_compositing.py:207`), while these nodes' alphas are already in the normal convention (1 = opaque).
Without the inversion the cutout comes out inside out.

Format in `VHS_VideoCombine`: **`ProRes`, profile `4444`** (with alpha it sets
`yuva444p10le`) or an `8bit-png` sequence.
Note: the VHS ProRes preset hard-tags `bt709` — irrelevant for HUD graphics, but do not push an HDR/HLG master
through it; assemble that in Resolve.

---

## The whole chain

```
Load Video (VHS) ──┬─────────────────────────────┐
                   ↓                             ↓
        LexYOLODetect (make_preview=False)    images
                   ↓ bounding_box                ↓
LoadImage (HUD.png) ──────────────→ LexHUDSprite
                                          ↓
                                    composite → VHS_VideoCombine
```

`Meta Batch Manager` = 50 frames. Enable `save_output` **before** the run.

---

## Node `Depth Curve · DA3 → V2` — `LexDepthCurve`

**Why.** Depth Anything **V2** outputs disparity (≈1/distance) — a non-linearity that stretches
the foreground. **V3** outputs DISTANCE, and the `v2_style` mode in `DA3Render` stretches it **linearly**
(`comfy/ldm/depth_anything_3/preprocess.py:93`). On a portrait with the face at 1.0–1.3 m and a background at 3 m,
the face takes ~15% of the tonal range instead of ~34% — hence "V3 is flatter, although more accurate".

This node is arithmetic on a finished depth map. **It does not run a model.** Measured:
**32–43 ms/frame** (512 and FullHD) versus ~16 s of mono_large inference — that is **0.2–0.3%** of the time.

| parameter | what it does |
|---|---|
| `depth` | output of `DA3Render` (or any depth-map IMAGE) |
| `source` | `near_is_white` — a regular depth map where near is bright (`v2_style`, `min_max`, ControlNet depth). `raw_distance` — `DA3Render` in `raw` mode, where values are distances |
| `mode` | `disparity` = like V2 · `gamma` = simple curve · `none` = renormalize only |
| `near_bias` | strength of the non-linearity. Note: **NOT the main control** — see below. Keep 0.5–1.0, do the sharp separation with levels |
| `black_point` / `white_point` | **the LEFT and RIGHT Levels sliders from Photoshop.** This is what spreads the gaps between planes |
| `gamma` | the middle Levels slider |
| `clarity` | local contrast with a large radius (like the Clarity slider in Photoshop) — sculpts form inside a shape. 0.3–0.5. Note: it used to be CLAHE — removed 10.08 because it produced blotches, see below |
| `detail` | micro-contrast (unsharp). 0.3–0.5. Note: from 0.8 it draws ring outlines on smooth gradients |
| `deband` | remove stair-step banding WITHOUT blurring the image. Bilateral filter: smooths small steps inside a plane and keeps the silhouette sharp. Runs BEFORE `clarity`. 0.3–0.5 |
| `deband_edge` | what counts as noise rather than an edge. 0.06 — banding only · **0.12 default** · 0.2–0.35 catches ragged leaf clutter but erodes the silhouette |
| `normalize` | **VIDEO:** `batch` (default) — black/white points are computed ONCE for the whole batch. This is what removes depth brightness flicker. `per_frame` — the old behavior, for single stills |
| `clip_low` / `clip_high` | clipping percentiles, % (1 / 99 default) |
| `invert` | invert if near came out dark |

**Outputs:** `depth` (IMAGE) · `depth_mask` (MASK) · `info` (reports where the tones actually lie —
`p2/25/50/75/98`, so you know where to set the levels).

### Why `near_bias` does not add contrast, it only darkens the image

`1/(distance + bias)` gives the whole tonal budget to the **nearest** object. Lower the `bias` and
the foreground takes even more while the rest sticks to black. It looks like "the picture is dying".
That is a curve collapse, not levels.

**Levels work differently:** they do not darken, they **cut out the band where the subject lives and stretch it
over the full range.** Measured 10.08.2026 on a frame with two figures (tone range on the body of the
right figure, black background excluded):

| | range | |
|---|---|---|
| V2, 23.2 s | 0.165 | reference |
| V3 + `near_bias 0.1` | **0.094** | 1.8× flatter — what you saw by eye |
| V3 + levels 0.22–0.62 + `clarity 0.35` | **0.262** | better than V2 |
| + `detail 0.8` | 0.294 | but already with rings |

Note — **the cost of levels:** the narrower the `black↔white` gap, the more planes blow out to pure white and
pure black. In the test the left (nearer) figure was completely blown to white. If you want both planes sculpted,
widen the gap and tune `clarity`.

### How to set it (in Photoshop terms)

1. `mode` = `disparity`, `near_bias` **0.5–1.0** (or `mode: none`) — a soft base.
2. Look at `info`: `p2/25/50/75/98` tells you which band the tones lie in.
3. Raise `black_point` until the background drops to black.
4. Lower `white_point` until the subject starts to blow out — then step back.
5. `clarity` 0.3–0.5 — this is what gives the cloth folds.
6. `detail` 0.3–0.5 at the end, no more.
7. If banding appears — `deband` 0.3–0.5.

### Stair-step banding: the main cause was in the node itself

Caught 10.08.2026: `clarity` was computed through **uint8**, so CLAHE itself cut the depth map
into 256 steps. With a narrow subject band (0.09 of the range) that is ~23 levels for the whole body — hence
"like 8 bit", and the banding carried over into generation. Switched to **uint16** (65,536 levels).

Measurement of "steppiness" (99th percentile of jump / median, on a smooth gradient):

| | |
|---|---|
| clean gradient | 1.7 |
| clarity 8-bit + detail 0.5 | **5.0** |
| clarity 16-bit | **2.7** |
| 8-bit + `deband 0.5` | 5.0 |

Note: **`deband` alone does NOT cure 8-bit banding** — 16 bits does. `deband` is for
the rest: steps from the model itself and from heavy stretching with levels.

Chain: `LoadDA3Model` (mono_large) → `DA3Inference` → `DA3Render` (`depth`, `v2_style`)
→ **`LexDepthCurve`** → onward to visualization.

**You can attach the node to V2 too** — but with `mode` = **`none`**, because V2 already outputs disparity,
and applying it a second time crumples it again. Levels + `clarity` work the same way on V2.


### Blotches / ragged clutter on the background and flicker on video (fixed 10.08.2026)

Two separate problems, both were in the node itself.

**1. Blotches.** `clarity` was on CLAHE. CLAHE equalizes the histogram **in each tile separately**;
at 1080×1920 a tile was 90×160 px, and a tile that landed on a flat background stretched micro-noise
over the whole range — hence the ragged clutter on bushes and walls. Replaced with an unsharp mask with a large
radius (`v + (v − heavily_blurred) × strength`) — this is the real Photoshop Clarity: linear,
it does not make blotches.

**2. Brightness flicker on video.** The 1%/99% quantiles were computed **on each frame separately**.
As soon as an object enters or leaves the frame, the black and white points move and the whole picture jerks.
The parameter `normalize = batch` computes them once for the whole batch. For `mode: disparity` the bounds are now
taken analytically (`1/(1+bias)`…`1/bias`), i.e. they are always shared across all frames.

Note — **what the node will NOT cure.** Ragged clutter on leaves and thin branches is mostly **real noise
of the depth model itself**: thin branches at different distances produce speckle. It is visible on V2
and on V3. Measured on a 1080×1920 frame: `deband_edge` 0.06 → 0.35 reduces the clutter 0.0397 → 0.0294,
but the silhouette sharpness drops 0.056 → 0.036. A trade-off, not a solution. Above all — **do not amplify**:
`detail` is the main culprit here.

## "The volume doesn't come out" + sharpening on the contour (fixed 10.08.2026)

Two different problems, both mine.

### 1. Halo along the silhouette — from the clarity base
`clarity` was computed as `v + (v − heavily_blurred) × strength`, and the blur was **Gaussian**.
A Gaussian blurs **across the edge** of the figure, so near the silhouette the difference is huge →
a light rim outside, a dark one inside. That is the "edge gets sharpened".
Replaced with a **guided filter** (`cv2.ximgproc.guidedFilter`), which does not cross the edge.

| clarity base | halo near the silhouette, mean \|Δ\| | max | time at 1080×1920 |
|---|---|---|---|
| Gaussian (before) | 0.102 | 0.459 | 0.068 s |
| guided (now) | **0.035** | 0.338 | 0.055 s |

So the halo dropped threefold and it **did not get more expensive**. If opencv has no contrib — fallback to
a bilateral filter on a downscaled copy (0.015 s versus 1.4 s for a full bilateral).

### 2. Plane volume — `focus` + `focus_strength` (new)
`black_point`/`white_point` are Levels: they **clip** the other planes to black and white.
So as soon as you stretched yourself (the middle plane), the laptop in front and the building behind
became flat blobs — and the sense of scene volume disappeared with them.

`focus` is **Curves**: a logistic S-curve around the chosen tone. The tones of your plane
are stretched, the rest is **softly compressed**, and the curve holds the ends 0 and 1.

Measured on the same 1080×1920 frame (the figure is the box 700:1500 × 150:800, tone 0.545):

| what | range on the figure | how much background crushed to black |
|---|---|---|
| nothing | 0.529 | 0 % |
| levels 0.55–0.95 | 0.85 | **18 %** |
| focus 0.55 × 4 | 0.626 | 0 % |
| focus 0.55 × 8 | 0.784 | 0 % |
| focus 0.55 × 12 | **0.891** | **0 %** |

The S-curve gives **more** volume than Levels and kills no plane at all.

**Where to put `focus`.** The console now prints a histogram of the planes:
```
[LexDepthCurve] planes 0.0→1.0: | ▁▂▁█▂▄▃▃|  (put focus on the hump you want)
```
Each hump = a separate plane of the scene. Put `focus` at the position of your hump (left 0.0 = farthest).

### Recipe for "make myself stand out on the middle plane"
1. `black_point 0.0`, `white_point 1.0` — **do not touch** Levels, they are for something else.
2. Look at the histogram, set `focus` on your hump.
3. `focus_strength` 4 → 8 → 12 until the plane "stands up".
4. `clarity` 0.4–0.7 — sculpting inside the figure (now without a rim).
5. `detail` 0.3–0.5 — fine folds.

### Why V2 with `mode: none` got worse than before
Previously `clarity` was CLAHE — it **equalized the histogram** and by itself gave V2
a boost in local contrast (together with blotches and flicker). Now clarity is honest
local contrast; it makes no blotches, but it does not give a "free" stretch either.
That stretch now lives in `focus_strength` — on V2 you need to enable it too.
