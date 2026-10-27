"""Camera Realism: опис повзунків і профілі камер. Один джерело правди для ноди (node.py), рушія (engine.py)
і гейта видимості віджетів (depth_look/tests/ui_widgets_gate.py читає SPEC звідси).
Імена параметрів не міняти: workflow зберігає значення віджетів по них.
SPEC: (name, lo, hi, step, group, label, tooltip[, kind]). kind 'i' - ціле."""

import json
import os

PROFILE_CUSTOM = "Custom"
DEFAULT_PROFILE = "ARRI Alexa 35"

SPEC = [
    # 1 · Сенсор
    ("noise_luma", 0.0, 1.0, 0.01, "Sensor", "luma noise",
     "Brightness noise (shot + read), stronger in the shadows. 0 = off."),
    ("noise_chroma", 0.0, 1.0, 0.01, "Sensor", "colour noise",
     "Colour noise: softer and blotchier than luma noise, mostly in the shadows. 0 = off."),
    ("noise_size", 0.0, 1.0, 0.01, "Sensor", "grain size",
     "0 = pixel-sharp noise, 1 = coarse clumps (film grain)."),
    ("iso", 100, 25600, 100, "Sensor", "ISO",
     "Virtual ISO: raises both noise types relative to ISO 800. Has no effect when both noise sliders are 0.", "i"),
    ("fixed_pattern", 0.0, 1.0, 0.01, "Sensor", "fixed pattern",
     "Faint fixed sensor pattern (rows, columns, per-pixel gain). Identical on every frame of the clip. 0 = off."),
    # 2 · Байєр
    ("bayer_amount", 0.0, 1.0, 0.01, "Bayer", "Bayer mosaic",
     "Colour-filter mosaic plus simple demosaicing: colour moiré and zipper edges on fine detail. Mixed with the original by this amount. 0 = off."),
    # 3 · ISP
    ("isp_denoise", 0.0, 1.0, 0.01, "Camera processing", "noise reduction",
     "In-camera noise reduction: smooths flat areas and fine texture (waxy skin at high values). 0 = off."),
    ("isp_sharpen", 0.0, 1.0, 0.01, "Camera processing", "sharpening",
     "In-camera sharpening with bright and dark halos along edges. 0 = off."),
    ("isp_sharpen_radius", 0.5, 4.0, 0.1, "Camera processing", "halo width",
     "Width of the sharpening halo, px. Used only when sharpening is above 0."),
    ("isp_blocks", 0.0, 1.0, 0.01, "Camera processing", "block compression",
     "Light 8×8 block quantization (JPEG / H.264-like). 0 = off."),
    # 4 · Оптика
    ("vignette", 0.0, 1.0, 0.01, "Optics", "vignette",
     "Corner darkening by the cos⁴ law. 0 = off."),
    ("distortion", -1.0, 1.0, 0.01, "Optics", "distortion",
     "Lens distortion: positive = barrel, negative = pincushion. 0 = off."),
    ("chroma_ab", 0.0, 1.0, 0.01, "Optics", "colour fringes",
     "Lateral chromatic aberration: red and blue fringes that grow toward the edges. 0 = off."),
    ("corner_softness", 0.0, 1.0, 0.01, "Optics", "soft corners",
     "Blur that grows toward the corners of the frame. 0 = off."),
    ("veil", 0.0, 1.0, 0.01, "Optics", "veiling glare",
     "Light haze from internal lens reflections: lowers contrast and lifts the blacks. 0 = off."),
    # 5 · Глибина
    ("depth_haze", 0.0, 1.0, 0.01, "Depth", "aerial haze",
     "Aerial perspective: far areas lose contrast and saturation and drift toward a sky tone. Needs the depth input; skipped without it."),
    ("depth_detail", 0.0, 1.0, 0.01, "Depth", "near detail",
     "Local contrast, stronger on near areas. Needs the depth input; skipped without it."),
    # 7 · Тон і колір (ядро v2)
    ("highlight_rolloff", 0.0, 1.0, 0.01, "Tone & colour", "highlight rolloff",
     "Soft shoulder: highlights fade into white gradually and lose colour instead of clipping (ARRI, film). 0 = off."),
    ("highlight_clip", 0.0, 1.0, 0.01, "Tone & colour", "highlight clip",
     "Hard per-channel clipping: flat blown patches with a hue shift, as in 8-bit video. 0 = off."),
    ("blacks", -1.0, 1.0, 0.01, "Tone & colour", "blacks",
     "Positive = crushed blacks (old DSLR video), negative = lifted shadows (film fade, phone HDR). 0 = off."),
    ("warmth", -1.0, 1.0, 0.01, "Tone & colour", "warmth",
     "Camera white-balance bias: positive = warmer, negative = cooler. 0 = off."),
    ("saturation", -1.0, 1.0, 0.01, "Tone & colour", "saturation",
     "Camera colour saturation: positive = richer, negative = more muted. 0 = off."),
    ("skin_hue", -1.0, 1.0, 0.01, "Tone & colour", "skin hue",
     "Shifts skin and orange tones: positive = toward yellow-green (Sony), negative = toward magenta-red (Canon). 0 = off."),
    # 8 · Плівка
    ("film_grain", 0.0, 1.0, 0.01, "Film", "film grain",
     "Grain of the three emulsion layers: the blue layer is coarsest, grain lives in shadows and mid-tones and fades in highlights. 0 = off."),
    ("halation", 0.0, 1.0, 0.01, "Film", "halation",
     "Red-orange glow around bright areas, light scattered back from the film base. 0 = off."),
    ("gate_weave", 0.0, 1.0, 0.01, "Film", "gate weave",
     "Frame-to-frame jitter of the film in the camera gate, mostly vertical. Skipped for a single still frame."),
    ("dust", 0.0, 1.0, 0.01, "Film", "dust & scratches",
     "Dust specks and hairs on the print (new on every frame) and a faint vertical scratch that lasts tens of frames. 0 = off."),
    # 9 · Телефон
    ("smart_hdr", 0.0, 1.0, 0.01, "Phone", "Smart HDR",
     "Phone local tone mapping: lifted shadows, flattened highlights, halos along high-contrast edges. 0 = off."),
    ("deep_fusion", 0.0, 1.0, 0.01, "Phone", "Deep Fusion",
     "Over-crisp texture on hair, fabric and skin from multi-frame fusion; flat areas stay untouched. 0 = off."),
    ("shadow_smear", 0.0, 1.0, 0.01, "Phone", "shadow smear",
     "Phone noise reduction in the shadows: fine texture smears into watercolour blotches. 0 = off."),
    ("portrait_bokeh", 0.0, 1.0, 0.01, "Phone", "portrait bokeh",
     "Portrait-mode fake background blur from the depth map, with the typical cut-out errors on hair. Needs the depth input; skipped without it."),
    ("ghost_flare", 0.0, 1.0, 0.01, "Phone", "ghost flare",
     "Green ghost of bright points mirrored through the frame centre, the classic phone lens reflection. 0 = off."),
    # 10 · Старе відео
    ("line_skip", 0.0, 1.0, 0.01, "Old video", "line skipping",
     "DSLR video readout that skips sensor rows: aliasing, moiré on fine patterns and a soft image. 0 = off."),
    ("chroma_blotch", 0.0, 1.0, 0.01, "Old video", "colour blotches",
     "Large magenta-green colour noise blotches in the shadows. 0 = off."),
    ("banding", 0.0, 1.0, 0.01, "Old video", "banding",
     "8-bit steps in smooth gradients such as sky and walls; textured areas are not affected. 0 = off."),
    # 6 · Відео
    ("shake", 0.0, 1.0, 0.01, "Video", "hand shake",
     "Micro hand-shake: smooth sub-pixel shift and a tiny rotation that change from frame to frame. Skipped for a single still frame."),
    ("shake_speed", 0.25, 3.0, 0.05, "Video", "shake tempo",
     "Hand-shake tempo: higher = faster jitter."),
    ("flicker", 0.0, 1.0, 0.01, "Video", "exposure flicker",
     "Frame-to-frame exposure flicker. Skipped for a single still frame."),
    ("rolling_shutter", 0.0, 1.0, 0.01, "Video", "rolling shutter",
     "Rows shift in proportion to the shake speed (jelly skew). Needs hand shake above 0."),
    ("motion_blur", 0.0, 1.0, 0.01, "Video", "motion blur",
     "Shutter blur along real motion between frames: 0.5 = 180° shutter. Skipped for a single still frame."),
    # власне зерно: ручне, поверх будь-якого профілю; профіль його не перезаписує, майстер-сила не множить
    ("og_amount", 0.0, 1.0, 0.01, "Own grain", "own grain",
     "Your own grain on top of any profile. The profile does not overwrite it and the master strength does not scale it. 0 = off."),
    ("og_size", 0.3, 6.0, 0.05, "Own grain", "grain size",
     "Grain diameter in pixels of a 1000 px wide frame: it scales with resolution, so 4K looks like 1080p."),
    ("og_clump", 0.0, 1.0, 0.01, "Own grain", "clumping",
     "Loose/clumpy grain: 0 = even fine dots, 1 = grains gather into coarse clusters (pushed film)."),
    ("og_softness", 0.0, 1.0, 0.01, "Own grain", "softness",
     "0 = crisp grain edges (sharp scan), 1 = soft, smeared grain (print, soft lens)."),
    ("og_colour", 0.0, 1.0, 0.01, "Own grain", "colour",
     "0 = monochrome grain, 1 = every colour layer has its own grain (colour negative)."),
    ("og_blue", 0.5, 2.5, 0.05, "Own grain", "blue layer size",
     "Size of the blue layer grain relative to red and green (real film: about 1.4). Works when colour > 0."),
    ("og_shadows", 0.0, 2.0, 0.01, "Own grain", "in shadows",
     "Grain amount in dark tones (1 = as in mid-tones)."),
    ("og_mids", 0.0, 2.0, 0.01, "Own grain", "in mid-tones",
     "Grain amount in mid-tones."),
    ("og_highlights", 0.0, 2.0, 0.01, "Own grain", "in highlights",
     "Grain amount in bright tones (film: low, a digital grain plug-in: about 1)."),
    ("og_animate", 0.0, 1.0, 0.01, "Own grain", "changes per frame",
     "Video: 1 = new grain on every frame (film), 0 = the same grain stuck to the screen (a texture overlay)."),
    ("og_replace", 0, 1, 1, "Own grain", "replace camera noise",
     "1 = while own grain is on, the profile's sensor noise and film grain are turned off; 0 = own grain is added on top.", "i"),
    ("frame_offset", 0, 1000000000, 1, "Video", "first frame number",
     "Number of this batch's first frame within the whole clip. Set it when a clip is processed in chunks so shake and flicker continue seamlessly.", "i"),
]
NAMES = [s[0] for s in SPEC]
GROUP_ORDER = ["Sensor", "Bayer", "Camera processing", "Optics", "Depth", "Tone & colour", "Film", "Phone", "Old video", "Own grain", "Video"]
RUNTIME = {"frame_offset"}            # не частина профілю, не масштабується майстер-силою
# власне зерно: не частина профілю (вибір профілю його не чіпає), завжди з повзунків, майстер-сила не множить
OWN = tuple(s[0] for s in SPEC if s[4] == "Own grain")
NOISE_KEYS = ("noise_luma", "noise_chroma", "film_grain")   # що вимикає og_replace

# параметри-«сили»: їх множить майстер-сила; решта (iso, розмір зерна, радіус, темп, кадр) - ні
STRENGTH_KEYS = ("noise_luma", "noise_chroma", "fixed_pattern", "bayer_amount", "isp_denoise", "isp_sharpen", "isp_blocks",
                 "vignette", "distortion", "chroma_ab", "corner_softness", "veil", "depth_haze", "depth_detail",
                 "shake", "flicker", "rolling_shutter",
                 "highlight_rolloff", "highlight_clip", "blacks", "warmth", "saturation", "skin_hue",
                 "film_grain", "halation", "gate_weave", "dust", "smart_hdr", "deep_fusion", "shadow_smear", "portrait_bokeh",
                 "ghost_flare", "line_skip", "chroma_blotch", "banding", "motion_blur")

# нейтрально: усі сили 0 (нічого не рахується), решта - середні значення
NEUTRAL = {s[0]: 0.0 for s in SPEC}
NEUTRAL.update({"iso": 800, "noise_size": 0.25, "isp_sharpen_radius": 1.2, "shake_speed": 1.0, "frame_offset": 0,
                "og_size": 1.2, "og_clump": 0.35, "og_softness": 0.3, "og_colour": 0.25, "og_blue": 1.4,
                "og_shadows": 1.0, "og_mids": 1.0, "og_highlights": 0.4, "og_animate": 1.0, "og_replace": 0})

# значення профілів при майстер-силі 1. «Сила» 1 (типова) - рівно профіль, 0.5 - половина кожної сили.
# v4 (26.10.2026, відгук: усе, крім ARRI, пережарювало, кольорові точки, айфон знищував кадр): ефекти, яких не виміряти
# на справжніх кадрах (Smart HDR, акварель у тінях, байєр, пропуск рядків, плями, смуги, кліп), зменшено до рівня
# делікатності ARRI; шум, шумодав, шарп, текстура і блоки - із замірів (calib/*.json)
PROFILES = {
    # ARRI Alexa 35: дрібний luma-шум, мʼяке плече світлих, теплувата шкіра, без шарпу; затвор 180°
    "ARRI Alexa 35": dict(iso=800, noise_luma=0.35, noise_chroma=0.03, noise_size=0.15, fixed_pattern=0.02,
                          bayer_amount=0.15, vignette=0.15, chroma_ab=0.08, corner_softness=0.25, veil=0.10,
                          depth_haze=0.15, shake=0.06, highlight_rolloff=0.65, blacks=-0.10, warmth=0.15,
                          saturation=-0.05, skin_hue=-0.10, motion_blur=0.50),
    # RED / Sony Venice: трохи більше шуму, легкий шарп, помірне плече
    "Cinema digital": dict(iso=800, noise_luma=0.45, noise_chroma=0.03, noise_size=0.20, fixed_pattern=0.04,
                           bayer_amount=0.10, isp_sharpen=0.08, isp_sharpen_radius=1.0, vignette=0.12, distortion=-0.02,
                           chroma_ab=0.12, corner_softness=0.20, veil=0.08, depth_haze=0.15, depth_detail=0.10,
                           shake=0.10, rolling_shutter=0.15, highlight_rolloff=0.40, saturation=0.05, skin_hue=0.05,
                           motion_blur=0.50),
    # Sony/Canon бездзеркалка: жорсткий кліп, смуги 8 біт, блоки, жовтувата шкіра Sony, rolling shutter
    "Mirrorless": dict(iso=1600, noise_luma=0.45, noise_chroma=0.10, noise_size=0.20, fixed_pattern=0.05,
                       bayer_amount=0.10, isp_denoise=0.15, isp_sharpen=0.35, isp_sharpen_radius=1.0, isp_blocks=0.20,
                       vignette=0.20, distortion=0.05, chroma_ab=0.18, corner_softness=0.25, veil=0.05,
                       depth_haze=0.15, depth_detail=0.15, shake=0.20, flicker=0.05, rolling_shutter=0.40,
                       highlight_clip=0.25, blacks=0.05, warmth=-0.05, saturation=0.05, skin_hue=0.15, banding=0.10,
                       motion_blur=0.35),
    # iPhone: Smart HDR, Deep Fusion, акварель у тінях; короткий затвор. Портретне боке і зелений привид вимкнено (26.10.2026):
    # на справжніх кадрах їх немає, повзунки лишились для ручного вибору
    "iPhone": dict(iso=640, noise_luma=0.25, noise_chroma=0.10, noise_size=0.30, fixed_pattern=0.02,
                   bayer_amount=0.0, isp_denoise=0.50, isp_sharpen=0.40, isp_sharpen_radius=1.4, isp_blocks=0.25,
                   vignette=0.08, chroma_ab=0.06, corner_softness=0.15, veil=0.06, depth_detail=0.05,
                   shake=0.30, flicker=0.05, rolling_shutter=0.50, smart_hdr=0.15, deep_fusion=0.65, shadow_smear=0.15,
                   blacks=-0.08, warmth=0.05, saturation=0.08, motion_blur=0.15),
    # дзеркалка 2012 у відео (Canon 5D Mk III / 7D): пропуск рядків, плями в тінях, кліп світлих, вбиті чорні, тепла Canon
    "DSLR 2012": dict(iso=1600, noise_luma=0.30, noise_chroma=0.08, noise_size=0.25, fixed_pattern=0.08,
                      bayer_amount=0.20, isp_sharpen=0.15, isp_sharpen_radius=1.2, isp_blocks=0.35,
                      vignette=0.30, distortion=0.08, chroma_ab=0.30, corner_softness=0.30, veil=0.08,
                      depth_haze=0.10, shake=0.25, flicker=0.05, rolling_shutter=0.45, line_skip=0.30,
                      chroma_blotch=0.20, banding=0.15, highlight_clip=0.30, blacks=0.12, warmth=0.12,
                      saturation=0.05, skin_hue=-0.12, motion_blur=0.50),
    # 16 мм плівка: зерно емульсії, halation, плече й підняті тіні, gate weave; без цифрового шуму, байєра й шарпу.
    # Пил вимкнено (26.10.2026): білі цятки читаються як брак, а не як плівка
    "16mm film": dict(iso=800, noise_size=0.65, vignette=0.40, distortion=0.04, chroma_ab=0.20,
                      corner_softness=0.45, veil=0.20, depth_haze=0.20, shake=0.15, flicker=0.15,
                      film_grain=0.50, halation=0.35, gate_weave=0.35, highlight_rolloff=0.80,
                      blacks=-0.20, warmth=0.20, saturation=-0.15, motion_blur=0.50),
}
# v3 (26.10.2026): шум і шумодав - із замірів на справжніх кадрах камер (calib/*.json, звідки й як - там само), не на око.
# Беремо лише ключі, які заміри справді визначають; решта профілю (тон, оптика, відео) поки ручна.
CALIBRATED = ("iso", "noise_luma", "noise_chroma", "noise_size", "isp_denoise")
for _name, _fn in (("iPhone", "iphone.json"), ("DSLR 2012", "dslr2012.json")):
    with open(os.path.join(os.path.dirname(__file__), "calib", _fn), encoding="utf-8") as _f:
        _cal = json.load(_f)
    PROFILES[_name].update({k: _cal["iso"] if k == "iso" else _cal["values"][k] for k in CALIBRATED})
PROFILE_NAMES = list(PROFILES) + [PROFILE_CUSTOM]


def profile_values(name):
    """Повний набір значень профілю (всі ключі, крім службових) - для ноди й для JS."""
    base = {k: v for k, v in NEUTRAL.items() if k not in RUNTIME and k not in OWN}
    base.update(PROFILES.get(name, {}))
    return base


# початкові значення віджетів ноди = профіль за замовчуванням: «Custom» стартує з того самого, що бачить користувач
WIDGET_DEFAULTS = dict(NEUTRAL, **profile_values(DEFAULT_PROFILE))


def resolve(profile, strength, custom=None):
    """Профіль + майстер-сила + (для Custom) значення повзунків -> готові параметри для engine.apply_realism."""
    custom = custom or {}
    if profile == PROFILE_CUSTOM:
        vals = {k: custom.get(k, WIDGET_DEFAULTS[k]) for k in NAMES if k not in RUNTIME}
    else:
        vals = profile_values(profile if profile in PROFILES else DEFAULT_PROFILE)
    vals.update({k: custom.get(k, NEUTRAL[k]) for k in OWN})
    s = min(max(float(strength), 0.0), 1.0)
    out = {k: (float(v) * s if k in STRENGTH_KEYS else v) for k, v in vals.items()}
    if out["og_amount"] > 1e-6 and int(out["og_replace"]) >= 1:
        out.update({k: 0.0 for k in NOISE_KEYS})
    return out
