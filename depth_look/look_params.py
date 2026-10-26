"""look_params — опис повзунків depth_look_lens, пресети і завантаження кадру.
Один источник правди для тюнера (tune), CLI (look_tool.py) і Comfy-ноди (ComfyUI-StepVisionKit/depth_look/).
Формат пресету = як presets/look_v1.json: плоскі ключі + vp_x/vp_y/shift_x/shift_y + palette [[t,[r,g,b]],...]."""
import hashlib, json, os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # EXR-глибина
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))          # look_proto, якщо імпорт через симлінк: фото/глибини стенду
CODE_DIR = os.path.dirname(os.path.realpath(__file__))    # справжня тека: StepVisionKit/depth_look
PRESETS_DIR = os.path.join(CODE_DIR, "presets")
LOCKED = {"look_v1", "00_original"}            # зафіксовані автором — тюнер не перезаписує
PHOTOS_DIR = os.path.join(HERE, "..", "..", "edited_pngs")
RAW_DEPTH = os.environ.get("DEPTH_LOOK_RAW_DEPTH") or os.path.join(HERE, "..", "..", "..", "Comfy", "Images", "Text2Img", "raw_depth")   # сирі глибини стенду
WORK_HW = (1424, 2144)                       # (W, H) робочої роздільності кадрів Nikon, як у run.py

# (name, lo, hi, step, group, label). kind: i — ціле; решта float.
# Порядок груп = порядок роботи дизайнера (GROUP_ORDER нижче); перший повзунок групи — її «сила» (GROUP_POWER).
# Назви параметрів (name) не міняти: пресети і workflow ключуються по них. Міняти можна group/label/порядок.
SPEC = [
    # 1 · Глибина і лінза — основа: що різке, що розмите; усе нижче бере глибину звідси
    ("blur_amount", 0.0, 3.0, 0.01, "Depth & Lens", "blur strength"),
    ("blur_target", -1.0, 1.0, 0.01, "Depth & Lens", "what stays sharp: background ↔ subject (−1 = behind the focal plane)"),
    ("focus", 0.5, 1.0, 0.005, "Depth & Lens", "focal plane"),
    ("r_min", 0.0, 0.01, 0.0002, "Depth & Lens", "min blur"),
    ("r_max", 0.0, 0.12, 0.001, "Depth & Lens", "max blur"),
    ("lens_gamma", 0.4, 2.5, 0.01, "Depth & Lens", "blur curve"),
    ("dispersion", 0.0, 0.4, 0.01, "Depth & Lens", "channel dispersion"),
    ("n_layers", 4, 16, 1, "Depth & Lens", "depth layers", "i"),

    # 2 · Колір і градієнт
    ("far_gradient", 0.0, 1.0, 0.01, "Color & Gradient", "strength: far gradient"),
    ("gradient_onset", 0.0, 0.4, 0.005, "Color & Gradient", "where it starts"),
    ("near_tint", 0.0, 1.0, 0.01, "Color & Gradient", "near tint"),
    ("luma_keep", 0.0, 1.0, 0.01, "Color & Gradient", "keep luminance"),
    ("gradient_gain", 0.5, 2.0, 0.01, "Color & Gradient", "gradient strength"),
    ("gradient_point", 0.0, 1.0, 0.01, "Color & Gradient", "gradient point (0 near … 1 far)"),
    ("gradient_squeeze", 0.0, 1.0, 0.01, "Color & Gradient", "squeeze the gradient around the point"),
    ("gradient_hue", -180.0, 180.0, 1.0, "Color & Gradient", "hue shift at the point, °"),
    ("gradient_tone", -1.0, 1.0, 0.01, "Color & Gradient", "tone at the point (− darker, + lighter)"),
    ("subject_map", 0.0, 1.0, 0.01, "Color & Gradient", "subject = map"),
    ("subject_contrast", 0.5, 3.0, 0.01, "Color & Gradient", "subject contrast"),
    ("bg_bands", 0.0, 1.0, 0.01, "Color & Gradient", "background bands"),
    ("band_flip", 0.0, 1.0, 1.0, "Color & Gradient", "flip bands"),

    # 3 · Силует
    ("silhouette", 0.0, 1.0, 0.01, "Silhouette", "strength"),
    ("sil_lo", 0.3, 0.95, 0.005, "Silhouette", "edge: from"),
    ("sil_hi", 0.35, 1.0, 0.005, "Silhouette", "edge: to"),
    ("sil_bg", 0.0, 1.0, 0.01, "Silhouette", "light background"),
    ("sil_bg_lo", 0.0, 1.0, 0.01, "Silhouette", "background no darker than"),
    ("sil_detail", 0.0, 0.6, 0.01, "Silhouette", "detail in silhouette"),
    ("sil_rim", 0.0, 2.0, 0.01, "Silhouette", "rim glow"),
    ("sil_rim_radius", 0.002, 0.05, 0.001, "Silhouette", "rim glow width"),

    # 4 · Деформація (дві незалежні дії, кожна зі своєю силою)
    ("warp", 0.0, 0.45, 0.005, "Warp · Wave", "wave strength (max 0.45)"),
    ("warp_scale", 0.02, 0.4, 0.005, "Warp · Wave", "wave scale"),
    ("warp_seed", 1, 999, 1, "Warp · Wave", "wave seed", "i"),
    ("smear", 0.0, 1.0, 0.01, "Warp · Smear", "subject smear strength"),
    ("smear_radial", 0.0, 1.0, 1.0, "Warp · Smear", "radial (1) / linear (0)"),
    ("smear_angle", 0.0, 360.0, 1.0, "Warp · Smear", "linear angle"),

    # 5 · Шлейф
    ("trail", 0.0, 1.0, 0.01, "Trail", "strength"),
    ("trail_angle", 0.0, 360.0, 1.0, "Trail", "direction, °"),
    ("trail_length", 0.0, 0.8, 0.005, "Trail", "length"),
    ("trail_source", 0.0, 1.0, 0.01, "Trail", "light ↔ subject"),
    ("trail_threshold", 0.2, 0.95, 0.01, "Trail", "light threshold"),
    ("trail_decay", 0.0, 6.0, 0.05, "Trail", "decay"),
    ("trail_tint", 0.0, 1.0, 0.01, "Trail", "color from palette"),

    # 6 · Копії в глибину (сила = видимість копій; 0 копій або 0 видимості = група вимкнена)
    ("stretch_opacity", 0.0, 1.0, 0.01, "Depth Copies", "strength: copy visibility (main control for the halo)"),
    ("stretch_length", 0.0, 1.0, 0.01, "Depth Copies", "length"),
    ("stretch_steps", 0, 48, 1, "Depth Copies", "count", "i"),
    ("stretch_lighten", 0.0, 1.0, 0.01, "Depth Copies", "lighten only"),
    ("stretch_fade", 0.2, 3.0, 0.01, "Depth Copies", "fade"),
    ("stretch_blur", 0.0, 0.05, 0.001, "Depth Copies", "copy blur"),
    ("stretch_color_shift", 0.0, 1.0, 0.01, "Depth Copies", "color from palette"),
    ("vp_x", 0.0, 1.0, 0.005, "Depth Copies", "vanishing point X"),
    ("vp_y", 0.0, 1.0, 0.005, "Depth Copies", "vanishing point Y"),
    ("shift_x", -0.3, 0.3, 0.005, "Depth Copies", "offset X"),
    ("shift_y", -0.3, 0.3, 0.005, "Depth Copies", "offset Y"),

    # 7 · Серпанок
    ("haze", 0.0, 1.0, 0.01, "Haze", "strength"),
    ("haze_start", 0.0, 0.9, 0.01, "Haze", "starting depth"),
    ("haze_curve", 0.3, 4.0, 0.05, "Haze", "density curve"),
    ("haze_t", 0.0, 1.0, 0.01, "Haze", "color (palette position)"),
    ("haze_scatter", 0.0, 1.0, 0.01, "Haze", "light scatter"),

    # 8 · Світло (чотири окремі ефекти, кожен зі своєю силою)
    ("glow", 0.0, 2.0, 0.01, "Light · Glow", "strength"),
    ("glow_radius", 0.005, 0.15, 0.001, "Light · Glow", "radius"),
    ("glow_threshold", 0.0, 1.0, 0.01, "Light · Glow", "brightness threshold"),
    ("glow_warmth", 0.0, 1.0, 0.01, "Light · Glow", "color: source ↔ palette"),

    ("spark", 0.0, 2.0, 0.01, "Light · Point Glow", "strength"),
    ("spark_threshold", 0.5, 1.0, 0.005, "Light · Point Glow", "brightness threshold"),
    ("spark_size", 0.002, 0.05, 0.001, "Light · Point Glow", "flare size"),
    ("spark_rays", 0, 8, 1, "Light · Point Glow", "rays (0 = round)", "i"),
    ("spark_angle", 0.0, 180.0, 1.0, "Light · Point Glow", "ray angle, °"),
    ("spark_length", 0.01, 0.3, 0.005, "Light · Point Glow", "ray length"),

    ("halation_strength", 0.0, 2.5, 0.01, "Light · Halation", "halation strength"),
    ("halation_radius", 0.002, 0.05, 0.0005, "Light · Halation", "radius"),
    ("halation_threshold", 0.2, 0.95, 0.01, "Light · Halation", "brightness threshold"),

    ("chroma_ab", 0.0, 15.0, 0.1, "Light · Chromatic Aberration", "aberration strength"),

    # 9 · Зерно (останній шар, поверх усього)
    ("grain_amount", 0.0, 100.0, 0.5, "Grain", "grain amount"),
    ("grain_size", 0.0, 100.0, 0.5, "Grain", "grain size"),
    ("grain_roughness", 0.0, 100.0, 0.5, "Grain", "grain roughness"),
    ("seed", 1, 9999, 1, "Grain", "grain seed", "i"),

    # Відео-групи: у статичній ноді НЕ показуються (див. ANIM_GROUPS). Записи й назви груп не чіпати —
    # їх забирає наступна відео-нода (DepthLookVideo).
    ("time", 0.0, 10.0, 0.05, "Animation (video)", "preview time, s"),
    ("grain_animate", 0, 1, 1, "Animation (video)", "grain: new every frame", "i"),
    ("grain_fps", 1.0, 60.0, 1.0, "Animation (video)", "grain: frames/s"),
    ("warp_speed", 0.0, 2.0, 0.01, "Animation (video)", "wave breathes"),
    ("glow_pulse", 0.0, 1.0, 0.01, "Animation (video)", "glow/spark pulse"),
    ("pulse_hz", 0.05, 4.0, 0.05, "Animation (video)", "pulse frequency, Hz"),
    ("haze_drift", 0.0, 1.0, 0.01, "Animation (video)", "haze drifts"),
    ("trail_wobble", 0.0, 45.0, 0.5, "Animation (video)", "trail wobble, °"),

    ("grad_mode", 0, 5, 1, "Gradient Animation", "mode: 0 static · 1 camera · 2 drift · 3 flow · 4 travel through depth · 5 wave", "i"),
    ("grad_amount", 0.0, 1.0, 0.01, "Gradient Animation", "animation strength (0 = static)"),
    ("grad_speed", -2.0, 2.0, 0.01, "Gradient Animation", "speed, cycles/s (sign = direction)"),
    ("grad_phase", 0.0, 1.0, 0.01, "Gradient Animation", "start phase"),
    ("grad_loop", 0.0, 20.0, 0.1, "Gradient Animation", "loop, s (0 = off; >0 = seamless loop)"),
    ("grad_angle", 0.0, 360.0, 1.0, "Gradient Animation", "motion/wave angle, ° (90 = vertical)"),
    ("grad_wavelength", 0.05, 2.0, 0.01, "Gradient Animation", "wavelength (mode 4: depth, mode 5: fraction of the long side)"),
    ("grad_amp", 0.0, 0.5, 0.005, "Gradient Animation", "wave amplitude (mode 5)"),
    ("grad_offset_x", -1.0, 1.0, 0.001, "Gradient Animation", "camera: offset X (fraction of the long side)"),
    ("grad_offset_y", -1.0, 1.0, 0.001, "Gradient Animation", "camera: offset Y (fraction of the long side)"),
]
NAMES = [s[0] for s in SPEC]
LENS_KEYS = ("focus", "n_layers", "r_min", "r_max", "lens_gamma", "dispersion", "blur_amount", "blur_target")   # міняєш — перерахунок шарів


INT_KEYS = ("n_layers", "stretch_steps", "warp_seed", "seed", "spark_rays", "grain_animate", "grad_mode")

# ---------------------------------------------------------------- групи: порядок, іконки, підказки
# Спільна логіка для всіх нод нодпаку (статична, відео, ...): ноді потрібні лише GROUP_ORDER / GROUP_ICON /
# GROUP_HINT / GROUP_POWER і groups_json(). Порядок — як працює дизайнер: від форми кадру до плівки.
#   1 глибина і лінза (що різке) -> 2 колір (фарбуємо за глибиною) -> 3 силует людини -> 4 деформація (форма)
#   -> 5 шлейф -> 6 копії в глибину -> 7 серпанок (повітря) -> 8 світло (глоу, іскри, halation, аберація)
#   -> 9 зерно (останнім, поверх усього). Відео-групи — в кінці й лише для відео-ноди.
ANIM_GROUPS = ("Animation (video)", "Gradient Animation")
ANIM_KEYS = [s[0] for s in SPEC if s[4] in ANIM_GROUPS]     # time, grain_animate, grain_fps, warp_speed, glow_pulse, ..., grad_*

GROUP_ORDER = [
    "Depth & Lens", "Color & Gradient", "Silhouette", "Warp · Wave", "Warp · Smear", "Trail",
    "Depth Copies", "Haze", "Light · Glow", "Light · Point Glow", "Light · Halation",
    "Light · Chromatic Aberration", "Grain",
    "Animation (video)", "Gradient Animation",
]

# «Сила» групи = перший повзунок. Якщо будь-який із перелічених ключів = 0, група вимкнена: решта повзунків
# приглушені («вимкнено — не рахується»; рушій такі гілки пропускає). Немає запису = група завжди «ввімкнена».
GROUP_POWER = {
    "Silhouette": ("silhouette",),
    "Warp · Wave": ("warp",),
    "Warp · Smear": ("smear",),
    "Trail": ("trail",),
    "Depth Copies": ("stretch_opacity", "stretch_steps"),
    "Haze": ("haze",),
    "Light · Glow": ("glow",),
    "Light · Point Glow": ("spark",),
    "Light · Halation": ("halation_strength",),
    "Grain": ("grain_amount",),
}


def _ico(body):
    """Іконка групи: однаковий лінійний стиль (24×24, штрих currentColor 1.7, круглі кінці)."""
    return ('<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + body + '</svg>')


GROUP_ICON = {
    "Preview": _ico('<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
    "Depth & Lens": _ico('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3" opacity=".55"/>'),
    "Color & Gradient": _ico('<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z"/><path d="M8.5 14.5a3.5 3.5 0 0 0 3.5 3.5" opacity=".6"/>'),
    "Silhouette": _ico('<circle cx="12" cy="8" r="3.6"/><path d="M5 20.5c0-4.2 3.4-6.5 7-6.5s7 2.3 7 6.5z"/>'),
    "Warp · Wave": _ico('<path d="M2 9c2.5-4 4.5-4 6.5 0s4 4 6.5 0 3.5-3 7-1"/><path d="M2 16c2.5-4 4.5-4 6.5 0s4 4 6.5 0 3.5-3 7-1" opacity=".55"/>'),
    "Warp · Smear": _ico('<circle cx="17" cy="12" r="3.6"/><path d="M3 8h8M2 12h9M3 16h8" opacity=".75"/>'),
    "Trail": _ico('<circle cx="18" cy="6" r="2.6"/><path d="M16 8.2 4 20M12.5 5.5 3 15M19 12.5 10 21" opacity=".8"/>'),
    "Depth Copies": _ico('<rect x="3" y="3" width="18" height="18" rx="2"/><rect x="7" y="7" width="10" height="10" rx="1.5" opacity=".75"/><rect x="10.2" y="10.2" width="3.6" height="3.6" opacity=".55"/>'),
    "Haze": _ico('<path d="M3 8h13M6 12h15M3 16h11M17 8h4M17 16h4"/>'),
    "Light · Glow": _ico('<circle cx="12" cy="12" r="4"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1"/>'),
    "Light · Point Glow": _ico('<path d="M12 2.5l2 7.5 7.5 2-7.5 2-2 7.5-2-7.5-7.5-2 7.5-2z"/>'),
    "Light · Halation": _ico('<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="6.5" stroke-dasharray="2.2 2.6"/><circle cx="12" cy="12" r="10" opacity=".4"/>'),
    "Light · Chromatic Aberration": _ico('<circle cx="9.5" cy="12" r="6"/><circle cx="14.5" cy="12" r="6" opacity=".6"/>'),
    "Grain": _ico('<circle cx="6" cy="7" r="1" fill="currentColor"/><circle cx="12" cy="5" r="1" fill="currentColor"/><circle cx="18" cy="8" r="1" fill="currentColor"/>'
                  '<circle cx="9" cy="12" r="1" fill="currentColor"/><circle cx="15" cy="13" r="1" fill="currentColor"/><circle cx="5" cy="17" r="1" fill="currentColor"/>'
                  '<circle cx="11" cy="18" r="1" fill="currentColor"/><circle cx="18" cy="18" r="1" fill="currentColor"/>'),
    "Animation (video)": _ico('<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>'),
    "Gradient Animation": _ico('<rect x="3" y="6" width="18" height="12" rx="2.5"/><path d="M7 12h8m-2.5-2.5L15 12l-2.5 2.5"/>'),
}

GROUP_HINT = {
    "Preview": "View settings. They do not affect the frame itself or the final render.",
    "Depth & Lens": "What is sharp and what is blurred. The base: color and the effects below take their depth from here.",
    "Color & Gradient": "Colors the frame by depth: the far areas get the gradient colors, the subject stays almost natural.",
    "Silhouette": "The subject as a dark silhouette on a light background.",
    "Warp · Wave": "A light heat-haze shimmer: the frame ripples slightly.",
    "Warp · Smear": "The subject is dragged along, like a long exposure.",
    "Trail": "Light or the subject streaks into a tail in the chosen direction.",
    "Depth Copies": "Semi-transparent copies recede from the subject into the depth of the frame, like a tunnel.",
    "Haze": "Air between the layers: the farther, the brighter and denser.",
    "Light · Glow": "A soft glow around bright areas.",
    "Light · Point Glow": "Little stars on the brightest points.",
    "Light · Halation": "A reddish glow around bright areas, like on motion-picture film.",
    "Light · Chromatic Aberration": "Colored fringing at the frame edges, like vintage optics.",
    "Grain": "Film grain — the last layer on top of everything.",
    "Animation (video)": "Everything that moves over time: grain, wave, light pulse, haze, trail.",
    "Gradient Animation": "Gradient motion: camera, drift, flow, travel through depth, wave.",
}


def spec_json(anim=False):
    """Повзунки для тюнера. anim=False (статична нода) ховає відео-групи; їхні записи у SPEC лишаються."""
    return [dict(name=s[0], lo=s[1], hi=s[2], step=s[3], group=s[4], label=s[5],
                 kind=(s[6] if len(s) > 6 else "f")) for s in SPEC if anim or s[4] not in ANIM_GROUPS]


def groups_json(anim=False):
    """Групи в порядку роботи: [{name, icon, hint, power:[ключі сили]}] — лише ті, що є у spec_json(anim)."""
    present = {s[4] for s in SPEC if anim or s[4] not in ANIM_GROUPS}
    names = [g for g in GROUP_ORDER if g in present] + sorted(present - set(GROUP_ORDER))
    return [dict(name=g, icon=GROUP_ICON.get(g, ""), hint=GROUP_HINT.get(g, ""), power=list(GROUP_POWER.get(g, ()))) for g in names]


def defaults():
    """Дефолти depth_look_lens у форматі пресету (щоб у тюнері було видно всі повзунки)."""
    import inspect, look
    sig = inspect.signature(look.depth_look_lens)
    p = {k: v.default for k, v in sig.parameters.items() if k in NAMES}
    vp, sh = sig.parameters["vanishing_point"].default, sig.parameters["stretch_shift"].default
    p.update(vp_x=vp[0], vp_y=vp[1], shift_x=sh[0], shift_y=sh[1])
    p["palette"] = [[float(t), [float(x) for x in c]] for t, c in look.LENS_STOPS]
    return p


# старі workflow/стан тюнера памʼятають колишні імена файлів пресетів
LEGACY_PRESETS = {"00_оригінал": "00_original"}


def preset_alias(name):
    """Старе імʼя пресета з суфіксом (<база>_...), файлу якого вже нема -> базовий пресет."""
    known = list_presets()
    if not isinstance(name, str) or name in known:
        return name
    for p in sorted(known, key=len, reverse=True):
        if name.startswith(p + "_"):
            return p
    return name


def load_preset(name_or_path):
    if not os.path.exists(name_or_path):
        name_or_path = preset_alias(LEGACY_PRESETS.get(name_or_path, name_or_path))
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PRESETS_DIR, name_or_path.removesuffix(".json") + ".json")
    p = defaults()
    p.update(json.load(open(path)))
    return p


def _preset_name(name):
    return "".join(ch for ch in name if ch.isalnum() or ch in "_-.").strip(".") or "look"


def save_preset(name, p):
    name = _preset_name(name)
    if name in LOCKED:
        raise ValueError("preset %s is locked, save it under a different name" % name)
    path = os.path.join(PRESETS_DIR, name + ".json")
    json.dump(p, open(path, "w"), indent=1, ensure_ascii=False)
    return path


def delete_preset(name):
    """Видаляє збережений користувачем пресет; вбудовані (LOCKED) і імена зі «сміттям» (../x) відхиляє."""
    if not isinstance(name, str) or _preset_name(name) != name:
        raise ValueError("invalid preset name %r" % (name,))
    if name in LOCKED:
        raise ValueError("preset %s is locked, it cannot be deleted" % name)
    path = os.path.join(PRESETS_DIR, name + ".json")
    if os.path.dirname(os.path.abspath(path)) != os.path.abspath(PRESETS_DIR) or not os.path.isfile(path):
        raise ValueError("preset %s not found" % name)
    os.remove(path)
    return path


def list_presets():
    return sorted(f[:-5] for f in os.listdir(PRESETS_DIR) if f.endswith(".json"))


def kwargs(p):
    """Пресет -> kwargs depth_look_lens (як variants.to_kwargs, але для всіх відомих ключів)."""
    kw = {}
    for k in NAMES:
        if k in p and k not in ("vp_x", "vp_y", "shift_x", "shift_y"):
            kw[k] = int(round(p[k])) if k in INT_KEYS else float(p[k])
    kw["vanishing_point"] = (float(p["vp_x"]), float(p["vp_y"]))
    kw["stretch_shift"] = (float(p["shift_x"]), float(p["shift_y"]))
    pal = sorted(p["palette"], key=lambda s: s[0])
    kw["gradient_stops"] = [(float(t), tuple(float(x) for x in c)) for t, c in pal]
    return kw


# ---------------------------------------------------------------- кадри
def photo_list():
    dm = json.load(open(os.path.join(HERE, "depth_map.json")))
    return [v for v in dm.values() if v]


def _read_rgb(path):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise FileNotFoundError(path)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    im = im[..., :3]
    scale = 65535.0 if im.dtype == np.uint16 else 255.0
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float32) / scale


def load_frame(photo=None, image=None, depth=None):
    """-> (rgb float HxWx3, depth_clean float HxW). photo = імʼя кадру стенду з depth_map.json,
    або image+depth — будь-який кадр (перший кадр відео) зі своєю картою глибини.
    Чиста глибина кешується: для кадрів стенду — як у variants (<photo>/depth_clean.npy)."""
    import look
    if photo:
        dm = {v: k for k, v in json.load(open(os.path.join(HERE, "depth_map.json"))).items() if v}
        image = os.path.join(PHOTOS_DIR, photo + ".png")
        depth = os.path.join(RAW_DEPTH, dm[photo])
        rgb = _read_rgb(image)
        H0, W0 = rgb.shape[:2]
        f = 2144.0 / max(H0, W0)                     # довга сторона 2144, співвідношення зберігаємо (горизонталь теж)
        W1, H1 = int(round(W0 * f)), int(round(H0 * f))
        if H0 >= W0 and abs(W1 - WORK_HW[0]) <= 2 and H1 == WORK_HW[1]:
            W1 = WORK_HW[0]                          # звичайні портретні кадри Nikon -> рівно 1424x2144 (кеші, паритет)
        rgb = cv2.resize(rgb, (W1, H1), interpolation=cv2.INTER_AREA)
        cache = os.path.join(HERE, photo, "depth_clean.npy")
    else:
        rgb = _read_rgb(image)
        H, W = rgb.shape[:2]
        if max(H, W) > 2144:
            f = 2144 / max(H, W)
            rgb = cv2.resize(rgb, (int(round(W * f)), int(round(H * f))), interpolation=cv2.INTER_AREA)
        key = hashlib.md5(("%s|%s|%s" % (os.path.abspath(image), os.path.abspath(depth), rgb.shape)).encode()).hexdigest()[:12]
        cache = os.path.join(HERE, "_cache", key + "_depth.npy")
    d = np.load(cache) if os.path.exists(cache) else None
    if d is not None and d.shape != rgb.shape[:2]:   # старий кеш під інший розмір/орієнтацію - перерахувати
        d = None
    if d is None:
        d = look.prepare_depth(cv2.imread(depth, cv2.IMREAD_UNCHANGED), rgb)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.save(cache, d)
    return rgb, d


def fit(rgb, d, long_side):
    H, W = d.shape
    if not long_side or max(H, W) <= long_side:
        return rgb, d
    f = long_side / max(H, W)
    sz = (max(2, int(round(W * f))), max(2, int(round(H * f))))
    return cv2.resize(rgb, sz, interpolation=cv2.INTER_AREA), cv2.resize(d, sz, interpolation=cv2.INTER_AREA)


def to8(rgb):
    return cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
