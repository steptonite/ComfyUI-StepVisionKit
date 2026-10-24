"""look_params — опис повзунків depth_look_lens, пресети і завантаження кадру.
Один источник правди для тюнера (tune), CLI (look_tool.py) і Comfy-ноди (ComfyUI-StepVisionKit/depth_look/).
Формат пресету = як presets/look_v1.json: плоскі ключі + vp_x/vp_y/shift_x/shift_y + palette [[t,[r,g,b]],...]."""
import hashlib, json, os
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")  # EXR-глибина
import numpy as np, cv2

HERE = os.path.dirname(os.path.abspath(__file__))          # look_proto, якщо імпорт через симлінк: фото/глибини стенду
CODE_DIR = os.path.dirname(os.path.realpath(__file__))    # справжня тека: StepVisionKit/depth_look
PRESETS_DIR = os.path.join(CODE_DIR, "presets")
LOCKED = {"look_v1", "00_оригінал"}            # зафіксовані автором — тюнер не перезаписує
PHOTOS_DIR = os.path.join(HERE, "..", "..", "edited_pngs")
RAW_DEPTH = os.environ.get("DEPTH_LOOK_RAW_DEPTH") or os.path.join(HERE, "..", "..", "..", "Comfy", "Images", "Text2Img", "raw_depth")   # сирі глибини стенду
WORK_HW = (1424, 2144)                       # (W, H) робочої роздільності кадрів Nikon, як у run.py

# (name, lo, hi, step, group, label). kind: i — ціле; решта float.
# Порядок груп = порядок роботи дизайнера (GROUP_ORDER нижче); перший повзунок групи — її «сила» (GROUP_POWER).
# Назви параметрів (name) не міняти: пресети і workflow ключуються по них. Міняти можна group/label/порядок.
SPEC = [
    # 1 · Глибина і лінза — основа: що різке, що розмите; усе нижче бере глибину звідси
    ("blur_amount", 0.0, 3.0, 0.01, "Глибина і лінза", "сила блюру"),
    ("blur_target", -1.0, 1.0, 0.01, "Глибина і лінза", "що різке: фон ↔ герой (−1 = за площиною фокуса)"),
    ("focus", 0.5, 1.0, 0.005, "Глибина і лінза", "площина фокуса"),
    ("r_min", 0.0, 0.01, 0.0002, "Глибина і лінза", "мін. блюр"),
    ("r_max", 0.0, 0.12, 0.001, "Глибина і лінза", "макс. блюр"),
    ("lens_gamma", 0.4, 2.5, 0.01, "Глибина і лінза", "крива блюру"),
    ("dispersion", 0.0, 0.4, 0.01, "Глибина і лінза", "дисперсія каналів"),
    ("n_layers", 4, 16, 1, "Глибина і лінза", "шари глибини", "i"),

    # 2 · Колір і градієнт
    ("far_gradient", 0.0, 1.0, 0.01, "Колір і градієнт", "сила: градієнт далекого"),
    ("gradient_onset", 0.0, 0.4, 0.005, "Колір і градієнт", "де починається"),
    ("near_tint", 0.0, 1.0, 0.01, "Колір і градієнт", "тон близького"),
    ("luma_keep", 0.0, 1.0, 0.01, "Колір і градієнт", "лишити яскравість"),
    ("gradient_gain", 0.5, 2.0, 0.01, "Колір і градієнт", "сила градієнта"),
    ("subject_map", 0.0, 1.0, 0.01, "Колір і градієнт", "людина = мапа"),
    ("subject_contrast", 0.5, 3.0, 0.01, "Колір і градієнт", "контраст людини"),
    ("bg_bands", 0.0, 1.0, 0.01, "Колір і градієнт", "смуги тла"),
    ("band_flip", 0.0, 1.0, 1.0, "Колір і градієнт", "перевернути смуги"),

    # 3 · Силует
    ("silhouette", 0.0, 1.0, 0.01, "Силует", "сила"),
    ("sil_lo", 0.3, 0.95, 0.005, "Силует", "межа: від"),
    ("sil_hi", 0.35, 1.0, 0.005, "Силует", "межа: до"),
    ("sil_bg", 0.0, 1.0, 0.01, "Силует", "світле тло"),
    ("sil_bg_lo", 0.0, 1.0, 0.01, "Силует", "тло не темніше"),
    ("sil_detail", 0.0, 0.6, 0.01, "Силует", "деталь у силуеті"),
    ("sil_rim", 0.0, 2.0, 0.01, "Силует", "сяйво по контуру"),
    ("sil_rim_radius", 0.002, 0.05, 0.001, "Силует", "ширина сяйва"),

    # 4 · Деформація (дві незалежні дії, кожна зі своєю силою)
    ("warp", 0.0, 0.45, 0.005, "Деформація · хвиля", "сила хвилі (макс. 0.45)"),
    ("warp_scale", 0.02, 0.4, 0.005, "Деформація · хвиля", "масштаб хвилі"),
    ("warp_seed", 1, 999, 1, "Деформація · хвиля", "сід хвилі", "i"),
    ("smear", 0.0, 1.0, 0.01, "Деформація · розмаз", "сила розмазу людини"),
    ("smear_radial", 0.0, 1.0, 1.0, "Деформація · розмаз", "радіальний (1) / лінійний (0)"),
    ("smear_angle", 0.0, 360.0, 1.0, "Деформація · розмаз", "кут лінійного"),

    # 5 · Шлейф
    ("trail", 0.0, 1.0, 0.01, "Шлейф", "сила"),
    ("trail_angle", 0.0, 360.0, 1.0, "Шлейф", "напрям, °"),
    ("trail_length", 0.0, 0.8, 0.005, "Шлейф", "довжина"),
    ("trail_source", 0.0, 1.0, 0.01, "Шлейф", "світло ↔ людина"),
    ("trail_threshold", 0.2, 0.95, 0.01, "Шлейф", "поріг світла"),
    ("trail_decay", 0.0, 6.0, 0.05, "Шлейф", "загасання"),
    ("trail_tint", 0.0, 1.0, 0.01, "Шлейф", "колір з палітри"),

    # 6 · Копії в глибину (сила = видимість копій; 0 копій або 0 видимості = група вимкнена)
    ("stretch_opacity", 0.0, 1.0, 0.01, "Копії в глибину", "сила: видимість копій (головне для ореолу)"),
    ("stretch_length", 0.0, 1.0, 0.01, "Копії в глибину", "довжина"),
    ("stretch_steps", 0, 48, 1, "Копії в глибину", "кількість", "i"),
    ("stretch_lighten", 0.0, 1.0, 0.01, "Копії в глибину", "лише освітлювати"),
    ("stretch_fade", 0.2, 3.0, 0.01, "Копії в глибину", "згасання"),
    ("stretch_blur", 0.0, 0.05, 0.001, "Копії в глибину", "блюр копій"),
    ("stretch_color_shift", 0.0, 1.0, 0.01, "Копії в глибину", "колір з палітри"),
    ("vp_x", 0.0, 1.0, 0.005, "Копії в глибину", "точка сходу X"),
    ("vp_y", 0.0, 1.0, 0.005, "Копії в глибину", "точка сходу Y"),
    ("shift_x", -0.3, 0.3, 0.005, "Копії в глибину", "зсув X"),
    ("shift_y", -0.3, 0.3, 0.005, "Копії в глибину", "зсув Y"),

    # 7 · Серпанок
    ("haze", 0.0, 1.0, 0.01, "Серпанок", "сила"),
    ("haze_start", 0.0, 0.9, 0.01, "Серпанок", "з якої глибини"),
    ("haze_curve", 0.3, 4.0, 0.05, "Серпанок", "крива густини"),
    ("haze_t", 0.0, 1.0, 0.01, "Серпанок", "колір (точка палітри)"),
    ("haze_scatter", 0.0, 1.0, 0.01, "Серпанок", "розсіяння світел"),

    # 8 · Світло (чотири окремі ефекти, кожен зі своєю силою)
    ("glow", 0.0, 2.0, 0.01, "Світло · глоу", "сила"),
    ("glow_radius", 0.005, 0.15, 0.001, "Світло · глоу", "радіус"),
    ("glow_threshold", 0.0, 1.0, 0.01, "Світло · глоу", "поріг яскравості"),
    ("glow_warmth", 0.0, 1.0, 0.01, "Світло · глоу", "колір: джерело ↔ палітра"),

    ("spark", 0.0, 2.0, 0.01, "Світло · точковий глоу", "сила"),
    ("spark_threshold", 0.5, 1.0, 0.005, "Світло · точковий глоу", "поріг яскравості"),
    ("spark_size", 0.002, 0.05, 0.001, "Світло · точковий глоу", "розмір блиску"),
    ("spark_rays", 0, 8, 1, "Світло · точковий глоу", "промені (0 = круглий)", "i"),
    ("spark_angle", 0.0, 180.0, 1.0, "Світло · точковий глоу", "кут променів, °"),
    ("spark_length", 0.01, 0.3, 0.005, "Світло · точковий глоу", "довжина променів"),

    ("halation_strength", 0.0, 2.5, 0.01, "Світло · halation", "сила halation"),
    ("halation_radius", 0.002, 0.05, 0.0005, "Світло · halation", "радіус"),
    ("halation_threshold", 0.2, 0.95, 0.01, "Світло · halation", "поріг яскравості"),

    ("chroma_ab", 0.0, 15.0, 0.1, "Світло · хром. аберація", "сила аберації"),

    # 9 · Зерно (останній шар, поверх усього)
    ("grain_amount", 0.0, 100.0, 0.5, "Зерно", "кількість зерна"),
    ("grain_size", 0.0, 100.0, 0.5, "Зерно", "розмір зерна"),
    ("grain_roughness", 0.0, 100.0, 0.5, "Зерно", "грубість зерна"),
    ("seed", 1, 9999, 1, "Зерно", "сід зерна", "i"),

    # Відео-групи: у статичній ноді НЕ показуються (див. ANIM_GROUPS). Записи й назви груп не чіпати —
    # їх забирає наступна відео-нода (DepthLookVideo).
    ("time", 0.0, 10.0, 0.05, "Анімація (відео)", "час превʼю, с"),
    ("grain_animate", 0, 1, 1, "Анімація (відео)", "зерно: нове щокадру", "i"),
    ("grain_fps", 1.0, 60.0, 1.0, "Анімація (відео)", "зерно: кадрів/с"),
    ("warp_speed", 0.0, 2.0, 0.01, "Анімація (відео)", "хвиля дихає"),
    ("glow_pulse", 0.0, 1.0, 0.01, "Анімація (відео)", "пульс глоу/іскор"),
    ("pulse_hz", 0.05, 4.0, 0.05, "Анімація (відео)", "частота пульсу, Гц"),
    ("haze_drift", 0.0, 1.0, 0.01, "Анімація (відео)", "серпанок пливе"),
    ("trail_wobble", 0.0, 45.0, 0.5, "Анімація (відео)", "хитання шлейфу, °"),

    ("grad_mode", 0, 5, 1, "Анімація градієнта", "режим: 0 static · 1 camera · 2 drift · 3 перелив · 4 хід по глибині · 5 хвиля", "i"),
    ("grad_amount", 0.0, 1.0, 0.01, "Анімація градієнта", "сила анімації (0 = статика)"),
    ("grad_speed", -2.0, 2.0, 0.01, "Анімація градієнта", "швидкість, циклів/с (знак = напрям)"),
    ("grad_phase", 0.0, 1.0, 0.01, "Анімація градієнта", "початкова фаза"),
    ("grad_loop", 0.0, 20.0, 0.1, "Анімація градієнта", "петля, с (0 = вимк.; >0 = безшовний цикл)"),
    ("grad_angle", 0.0, 360.0, 1.0, "Анімація градієнта", "кут руху/хвилі, ° (90 = по вертикалі)"),
    ("grad_wavelength", 0.05, 2.0, 0.01, "Анімація градієнта", "довжина хвилі (режим 4: глибина, 5: частка довгої сторони)"),
    ("grad_amp", 0.0, 0.5, 0.005, "Анімація градієнта", "амплітуда хвилі (режим 5)"),
    ("grad_offset_x", -1.0, 1.0, 0.001, "Анімація градієнта", "камера: зсув X (частка довгої сторони)"),
    ("grad_offset_y", -1.0, 1.0, 0.001, "Анімація градієнта", "камера: зсув Y (частка довгої сторони)"),
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
ANIM_GROUPS = ("Анімація (відео)", "Анімація градієнта")
ANIM_KEYS = [s[0] for s in SPEC if s[4] in ANIM_GROUPS]     # time, grain_animate, grain_fps, warp_speed, glow_pulse, ..., grad_*

GROUP_ORDER = [
    "Глибина і лінза", "Колір і градієнт", "Силует", "Деформація · хвиля", "Деформація · розмаз", "Шлейф",
    "Копії в глибину", "Серпанок", "Світло · глоу", "Світло · точковий глоу", "Світло · halation",
    "Світло · хром. аберація", "Зерно",
    "Анімація (відео)", "Анімація градієнта",
]

# «Сила» групи = перший повзунок. Якщо будь-який із перелічених ключів = 0, група вимкнена: решта повзунків
# приглушені («вимкнено — не рахується»; рушій такі гілки пропускає). Немає запису = група завжди «ввімкнена».
GROUP_POWER = {
    "Силует": ("silhouette",),
    "Деформація · хвиля": ("warp",),
    "Деформація · розмаз": ("smear",),
    "Шлейф": ("trail",),
    "Копії в глибину": ("stretch_opacity", "stretch_steps"),
    "Серпанок": ("haze",),
    "Світло · глоу": ("glow",),
    "Світло · точковий глоу": ("spark",),
    "Світло · halation": ("halation_strength",),
    "Зерно": ("grain_amount",),
}


def _ico(body):
    """Іконка групи: однаковий лінійний стиль (24×24, штрих currentColor 1.7, круглі кінці)."""
    return ('<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + body + '</svg>')


GROUP_ICON = {
    "Превʼю": _ico('<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>'),
    "Глибина і лінза": _ico('<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 3v3M12 18v3M3 12h3M18 12h3" opacity=".55"/>'),
    "Колір і градієнт": _ico('<path d="M12 3c4 5 6 8 6 11a6 6 0 0 1-12 0c0-3 2-6 6-11z"/><path d="M8.5 14.5a3.5 3.5 0 0 0 3.5 3.5" opacity=".6"/>'),
    "Силует": _ico('<circle cx="12" cy="8" r="3.6"/><path d="M5 20.5c0-4.2 3.4-6.5 7-6.5s7 2.3 7 6.5z"/>'),
    "Деформація · хвиля": _ico('<path d="M2 9c2.5-4 4.5-4 6.5 0s4 4 6.5 0 3.5-3 7-1"/><path d="M2 16c2.5-4 4.5-4 6.5 0s4 4 6.5 0 3.5-3 7-1" opacity=".55"/>'),
    "Деформація · розмаз": _ico('<circle cx="17" cy="12" r="3.6"/><path d="M3 8h8M2 12h9M3 16h8" opacity=".75"/>'),
    "Шлейф": _ico('<circle cx="18" cy="6" r="2.6"/><path d="M16 8.2 4 20M12.5 5.5 3 15M19 12.5 10 21" opacity=".8"/>'),
    "Копії в глибину": _ico('<rect x="3" y="3" width="18" height="18" rx="2"/><rect x="7" y="7" width="10" height="10" rx="1.5" opacity=".75"/><rect x="10.2" y="10.2" width="3.6" height="3.6" opacity=".55"/>'),
    "Серпанок": _ico('<path d="M3 8h13M6 12h15M3 16h11M17 8h4M17 16h4"/>'),
    "Світло · глоу": _ico('<circle cx="12" cy="12" r="4"/><path d="M12 2.5v3M12 18.5v3M2.5 12h3M18.5 12h3M5.3 5.3l2.1 2.1M16.6 16.6l2.1 2.1M18.7 5.3l-2.1 2.1M7.4 16.6l-2.1 2.1"/>'),
    "Світло · точковий глоу": _ico('<path d="M12 2.5l2 7.5 7.5 2-7.5 2-2 7.5-2-7.5-7.5-2 7.5-2z"/>'),
    "Світло · halation": _ico('<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="6.5" stroke-dasharray="2.2 2.6"/><circle cx="12" cy="12" r="10" opacity=".4"/>'),
    "Світло · хром. аберація": _ico('<circle cx="9.5" cy="12" r="6"/><circle cx="14.5" cy="12" r="6" opacity=".6"/>'),
    "Зерно": _ico('<circle cx="6" cy="7" r="1" fill="currentColor"/><circle cx="12" cy="5" r="1" fill="currentColor"/><circle cx="18" cy="8" r="1" fill="currentColor"/>'
                  '<circle cx="9" cy="12" r="1" fill="currentColor"/><circle cx="15" cy="13" r="1" fill="currentColor"/><circle cx="5" cy="17" r="1" fill="currentColor"/>'
                  '<circle cx="11" cy="18" r="1" fill="currentColor"/><circle cx="18" cy="18" r="1" fill="currentColor"/>'),
    "Анімація (відео)": _ico('<circle cx="12" cy="12" r="9"/><path d="M10 8.5v7l6-3.5z"/>'),
    "Анімація градієнта": _ico('<rect x="3" y="6" width="18" height="12" rx="2.5"/><path d="M7 12h8m-2.5-2.5L15 12l-2.5 2.5"/>'),
}

GROUP_HINT = {
    "Превʼю": "Налаштування перегляду. На сам кадр і фінальний рендер не впливають.",
    "Глибина і лінза": "Що різке, а що розмите. Основа: колір і ефекти нижче беруть глибину звідси.",
    "Колір і градієнт": "Фарбує кадр за глибиною: далеке — кольорами градієнта, людина — майже своя.",
    "Силует": "Людина темним силуетом на світлому тлі.",
    "Деформація · хвиля": "Легке «марево»: кадр дрібно пливе.",
    "Деформація · розмаз": "Людину тягне слідом, як на довгій витримці.",
    "Шлейф": "Світло або людина тягнеться хвостом у вибраний бік.",
    "Копії в глибину": "Від людини вглиб кадру йдуть напівпрозорі копії, як у тунелі.",
    "Серпанок": "Повітря між шарами: чим далі, тим світліше й густіше.",
    "Світло · глоу": "М'яке світіння навколо яскравих місць.",
    "Світло · точковий глоу": "Зірочки на найяскравіших точках.",
    "Світло · halation": "Червонувате світіння навколо яскравого, як на кіноплівці.",
    "Світло · хром. аберація": "Кольорова кайма по краях кадру, як у старої оптики.",
    "Зерно": "Плівкове зерно — останній шар поверх усього.",
    "Анімація (відео)": "Те, що рухається в часі: зерно, хвиля, пульс світла, серпанок, шлейф.",
    "Анімація градієнта": "Рух градієнта: камера, дрейф, перелив, хід по глибині, хвиля.",
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


def load_preset(name_or_path):
    path = name_or_path if os.path.exists(name_or_path) else os.path.join(PRESETS_DIR, name_or_path.removesuffix(".json") + ".json")
    p = defaults()
    p.update(json.load(open(path)))
    return p


def save_preset(name, p):
    name = "".join(ch for ch in name if ch.isalnum() or ch in "_-.").strip(".") or "look"
    if name in LOCKED:
        raise ValueError("пресет %s зафіксований, збережи під іншим імʼям" % name)
    path = os.path.join(PRESETS_DIR, name + ".json")
    json.dump(p, open(path, "w"), indent=1, ensure_ascii=False)
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
