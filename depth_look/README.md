🇺🇦 Українська версія: [README.uk.md](README.uk.md)

# Depth Look — nodes "Depth Look · Full / Simple / Video" (StepVisionKit/Depth)

Lens blur by depth + color palette, silhouette, haze, trail, halation, grain.
Since 24.10.2026 it lives inside the **ComfyUI-StepVisionKit** pack (`depth_look/`); before that it was a standalone pack, `comfy_depth_look`.
The engine `look.py` / `look_params.py` / `look_api.py` / `tune.html` sits in this same folder; the author's separate
prototyping folder consumes it through symlinks (the standalone `look_tool.py` CLI works as before). Only numpy + cv2
(+ torch from Comfy) are required.

## Installation

Nothing separate: the nodes come with ComfyUI-StepVisionKit, menu category `StepVisionKit/Depth`.

- ComfyUI-Manager / Comfy Registry: `comfy node install stepvisionkit`
- Manually: `git clone https://github.com/steptonite/ComfyUI-StepVisionKit` into `ComfyUI/custom_nodes`, then
  `pip install -r requirements.txt`.

The node keys (`DepthLook`, `DepthLookSimple`, `DepthLookVideo`) are unchanged, so saved workflows open as they were.

## Language

Node texts are English by default. Ukrainian can be switched on in Settings → StepVisionKit → Language → Node language
(reload the page).

## Node UI (24.10.2026)

The interface follows the pack's rules in `../CONVENTIONS_UI.md` (shared: `web/svk_ui.css` + `web/svk_ui.js`):
native widgets are hidden (their values are still saved in the workflow), explanations live only in hover tooltips,
buttons are icons in a single row, rarely used controls are under "more". Tested in Nodes 2.0 mode.

- **Simple:** frame (hold = original) → look tiles with human-friendly names → effect chips (wrap onto lines,
  tap = on/off, drag or wheel = strength 0..2) → "Look strength" → "more". The card scrolls with the wheel inside.
- **Full:** the tuner fills the whole node; below the preview there is a row of icons (focus, hero, background, depth, original,
  zoom, loupe), at the bottom — full screen, "preview / final" and "more" (size, `max_side`, depth levels, cleanup, depth direction).
- **Video:** a frame strip with playback, `fps`, "preview / final" and the 4 main sliders
  (color motion, motion speed, light pulse, live grain); the rest of the animation is under "more". The node is an output node
  (`OUTPUT_NODE`), so it is computed even without saving after it.
- **Look strength** (`look_mix`) from the Simple node travels through `grade` into the Full and Video nodes: all three blend with the original
  the same way (the log says "look strength 0.70").

## Inputs

- `image` — frames (B,H,W,3).
- `depth` — depth map (1 frame for all or B frames; channel 0 is used; the size does not have to match).
- `preset` — a preset from `depth_look/presets` (default `look_v1`).
- `clean_depth` — True: raw depth is cleaned up (`prepare_depth`, frame = guide); False: the depth is already clean 0..1.
- `depth_light_is_near` — True: bright = near; False — invert.
- `overrides` (optional) — JSON applied over the preset, e.g. `{"silhouette":1.0,"trail":0.5}`.
- `palette` (optional) — replaces the palette: JSON `[[t,[r,g,b]],...]` (0..1) or `0:#0b0604, 0.5:#c54202, 1:#ffc9a0`.

- `fps` (optional, 24) and `start_time` (optional, 0) — frame i gets `time = start_time + i/fps` seconds; the animation
  sliders depend on `time`: `grain_animate` (new grain every frame), `warp_speed` (the wave "breathes"),
  `glow_pulse`/`pulse_hz` (pulse of glow and sparks), `haze_drift` (haze drifts), `trail_wobble` (trail wobble).
  Set them through `overrides`, e.g. `{"grain_animate":1,"glow":0.8,"glow_pulse":0.4,"warp_speed":0.5}`.

The output is an `image` of the same size. Progress is printed every 10 frames.

## How to connect depth

The `depth` input is any IMAGE with a depth map: Depth Anything / DA3 / MiDaS / Marigold nodes, etc.
(channel 0; 8/16-bit, 1/3/4 channels; a different size or even a slightly different aspect ratio is not a problem — the depth
is stretched to the frame). **Bright = near**; if your node outputs the opposite (bright = far), turn off
`depth_light_is_near`. Frames can have any aspect (portrait, landscape, square); effect radii
are computed from the long side. The orientation of depth must match the frame (a portrait depth on a landscape frame = error).
New in `overrides`: `blur_amount` (0..3, blur multiplier), `blur_target` (0..1: 1 = hero sharp, 0 = sharp background,
-1 = as in the preset), `glow*`, `spark*` (point glow, `spark_rays` 4/6/8 = star).

## Presets

Edited in the tuner: `look_tool.py tune` (in the author's prototyping folder). New presets appear in the list after
restarting Comfy. `look_v1` is locked (LOCKED) — save under a different name.

## Built-in tuner in the node (23.10.2026)

The node has the same tuner inside as `look_tool.py tune`: sliders, a gradient editor with a library,
an eyedropper, a preview. No separate browser tab is needed.

- **The frame** comes only from the `image` + `depth` input sockets (you cannot pick files). After the first graph run the node
  caches up to 8 evenly sampled frames of the batch (float16, `_cache/comfy_node_<id>_*`), and the preview is computed from them.
  Before the first run the tuner shows "run the graph once so the preview picks up your frame". After each run the preview picks up
  the new frame by itself.
- The **"frame N"** slider scrubs through the cached frames; the `time` slider = `start_time + N/fps`.
- Parameters are written into the hidden `params_json` widget and saved in the workflow. A non-empty `params_json` takes
  priority over `preset`/`overrides`/`palette`; changing `preset` in the node clears `params_json`.
- Gradients and presets are the same `gradients/` and `presets/` as in the standalone tuner.
- `max_side` (0 = native): the look is computed on a downscaled frame and grows back — for long videos on 8 GB.
- `depth_levels`: `clip` (default) — shared depth levels across the whole clip (`look.clip_levels`), no flicker;
  `frame` — per-frame stretch, as before. Works only with `clean_depth`.
- Grain is live: the frame's `seed` = `seed` + the batch frame number; `warp_seed` is constant.
- Comfy routes: `/depth_look/ui`, `/depth_look/api/<node>/...` (logic in `look_api.py`, shared with the standalone tuner).
