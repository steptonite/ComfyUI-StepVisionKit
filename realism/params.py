"""Camera Realism: опис повзунків і профілі камер. Один джерело правди для ноди (node.py), рушія (engine.py)
і гейта видимості віджетів (depth_look/tests/ui_widgets_gate.py читає SPEC звідси).
Імена параметрів не міняти: workflow зберігає значення віджетів по них.
SPEC: (name, lo, hi, step, group, label, tooltip[, kind]). kind 'i' - ціле."""

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
    # 6 · Відео
    ("shake", 0.0, 1.0, 0.01, "Video", "hand shake",
     "Micro hand-shake: smooth sub-pixel shift and a tiny rotation that change from frame to frame. Skipped for a single still frame."),
    ("shake_speed", 0.25, 3.0, 0.05, "Video", "shake tempo",
     "Hand-shake tempo: higher = faster jitter."),
    ("flicker", 0.0, 1.0, 0.01, "Video", "exposure flicker",
     "Frame-to-frame exposure flicker. Skipped for a single still frame."),
    ("rolling_shutter", 0.0, 1.0, 0.01, "Video", "rolling shutter",
     "Rows shift in proportion to the shake speed (jelly skew). Needs hand shake above 0."),
    ("frame_offset", 0, 1000000000, 1, "Video", "first frame number",
     "Number of this batch's first frame within the whole clip. Set it when a clip is processed in chunks so shake and flicker continue seamlessly.", "i"),
]
NAMES = [s[0] for s in SPEC]
GROUP_ORDER = ["Sensor", "Bayer", "Camera processing", "Optics", "Depth", "Video"]
RUNTIME = {"frame_offset"}            # не частина профілю, не масштабується майстер-силою

# параметри-«сили»: їх множить майстер-сила; решта (iso, розмір зерна, радіус, темп, кадр) - ні
STRENGTH_KEYS = ("noise_luma", "noise_chroma", "fixed_pattern", "bayer_amount", "isp_denoise", "isp_sharpen", "isp_blocks",
                 "vignette", "distortion", "chroma_ab", "corner_softness", "veil", "depth_haze", "depth_detail",
                 "shake", "flicker", "rolling_shutter")

# нейтрально: усі сили 0 (нічого не рахується), решта - середні значення
NEUTRAL = {s[0]: 0.0 for s in SPEC}
NEUTRAL.update({"iso": 800, "noise_size": 0.25, "isp_sharpen_radius": 1.2, "shake_speed": 1.0, "frame_offset": 0})

# значення профілів при майстер-силі 1. «Сила» 0.5 (типова) дає рівно половину кожної сили.
PROFILES = {
    # ARRI Alexa 35: дрібний низький luma-шум, майже без chroma, мʼяка оптика, без шарпу
    "ARRI Alexa 35": dict(iso=800, noise_luma=0.40, noise_chroma=0.03, noise_size=0.15, fixed_pattern=0.03,
                          bayer_amount=0.20, vignette=0.15, chroma_ab=0.08, corner_softness=0.25, veil=0.12,
                          depth_haze=0.15, shake=0.06),
    # RED / Sony Venice: трохи більше шуму, легкий шарп, стримана оптика
    "Cinema digital": dict(iso=800, noise_luma=0.50, noise_chroma=0.08, noise_size=0.20, fixed_pattern=0.05,
                           bayer_amount=0.30, isp_sharpen=0.15, isp_sharpen_radius=1.0, vignette=0.12, distortion=-0.02,
                           chroma_ab=0.12, corner_softness=0.20, veil=0.08, depth_haze=0.15, depth_detail=0.10,
                           shake=0.10, rolling_shutter=0.15),
    # Sony/Canon бездзеркалка: шумодав + шарп, помітна оптика
    "Mirrorless": dict(iso=1600, noise_luma=0.60, noise_chroma=0.15, noise_size=0.20, fixed_pattern=0.06,
                       bayer_amount=0.40, isp_denoise=0.15, isp_sharpen=0.35, isp_sharpen_radius=1.0, isp_blocks=0.10,
                       vignette=0.25, distortion=0.05, chroma_ab=0.20, corner_softness=0.30, veil=0.05,
                       depth_haze=0.20, depth_detail=0.15, shake=0.20, flicker=0.05, rolling_shutter=0.30),
    # телефон: сильний шумодав-«віск», шарп-ореоли, chroma-шум у тінях, блоки, обчислювальна мікродеталь
    "iPhone": dict(iso=640, noise_luma=0.45, noise_chroma=0.35, noise_size=0.30, fixed_pattern=0.04,
                   bayer_amount=0.50, isp_denoise=0.70, isp_sharpen=0.60, isp_sharpen_radius=1.6, isp_blocks=0.35,
                   vignette=0.10, distortion=0.03, chroma_ab=0.08, corner_softness=0.20, veil=0.10,
                   depth_haze=0.10, depth_detail=0.30, shake=0.30, flicker=0.10, rolling_shutter=0.50),
    # дзеркалка 2012: chroma-шум, муар байєра, JPEG, виразна оптика
    "DSLR 2012": dict(iso=1600, noise_luma=0.70, noise_chroma=0.50, noise_size=0.25, fixed_pattern=0.12,
                      bayer_amount=0.70, isp_denoise=0.10, isp_sharpen=0.45, isp_sharpen_radius=1.2, isp_blocks=0.40,
                      vignette=0.35, distortion=0.10, chroma_ab=0.35, corner_softness=0.35, veil=0.10,
                      depth_haze=0.15, depth_detail=0.15, shake=0.25, flicker=0.05, rolling_shutter=0.45),
    # 16 мм плівка: грубе сильне зерно, віньєтка, мʼякі кути, гейт-тремтіння, мерехтіння проектора; без байєра й блоків
    "16mm film": dict(iso=800, noise_luma=1.00, noise_chroma=0.15, noise_size=0.65,
                      isp_sharpen=0.20, isp_sharpen_radius=1.4, vignette=0.40, distortion=0.04, chroma_ab=0.30,
                      corner_softness=0.55, veil=0.25, depth_haze=0.20, shake=0.35, flicker=0.35),
}
PROFILE_NAMES = list(PROFILES) + [PROFILE_CUSTOM]


def profile_values(name):
    """Повний набір значень профілю (всі ключі, крім службових) - для ноди й для JS."""
    base = {k: v for k, v in NEUTRAL.items() if k not in RUNTIME}
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
    s = min(max(float(strength), 0.0), 1.0)
    return {k: (float(v) * s if k in STRENGTH_KEYS else v) for k, v in vals.items()}
