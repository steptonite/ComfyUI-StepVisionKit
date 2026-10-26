"""StepVisionKit — детекція, трекінг, HUD-композ і глибина для відео в ComfyUI (Mac/MPS + CUDA).

Один пак замість двох. Раніше це жило як `comfyui-lex-bridge` і `comfyui-visionkit`,
які реєстрували ЧАСТИНУ ключів однаково — вигравав той, що вантажився пізніше за алфавітом,
і воркфлоу мовчки брав чужу ноду. Тут обидві родини зведені, ключі не перетинаються.

Дві родини нод, різні за призначенням:

  · **HUD / глибина** (`Lex*`) — те, на чому їде формат «говоряща голова над картою глибини»:
    детект без трекінгу, кастомна PNG-рамка на боксі з діркою у відео, крива глибини.
  · **Трекінг** (`VK*`) — ByteTrack: `VKDetect` віддає стабільний `track_id`, `VKSelectBox`
    тримається ОДНОГО об'єкта через усе відео (а не «найбільшого в кадрі» покадрово).

Ключі нод лишились історичні (`Lex*` / `VK*`) СВІДОМО: ComfyUI шукає ноду в збереженому json
за рядком `type`, і перейменування зробило б червоними всі раніше збережені воркфлоу.
Змінилась тільки менюшна категорія — тепер усе під `StepVisionKit`.
"""

from .core.nodes_core import LexBoxesToBBOX, LexHUDBoxes, LexYOLODetect
from .core.depth_curve import LexDepthCurve
from .core.depth_grade import SVKDepthGrade, SVKDepthGradeLive, SVKDepthGradeView, SVKDepthGradeVideoView
from .core.depth_raw import SVKDepthReceiver, SVKRawDepthLoad, SVKRawDepthSave, SVKSaveDepth16
from .core.hud_sprite import LexHUDSprite
# Depth Look (look v1): лінзовий блюр за глибиною + палітра. Ключі DepthLook* лишились з окремого паку -
# збережені воркфлоу відкриваються як були, змінилась лише категорія меню.
from .depth_look import NODE_CLASS_MAPPINGS as DL_NODES, NODE_DISPLAY_NAME_MAPPINGS as DL_DISPLAY

# Термо-сторож: пауза при перегріві SoC (м'яка — всередині процесу, жорстка — окремий наглядач).
# Помилка сторожа не має ламати пакет: install() сам ловить усе, це страховка на імпорт.
try:
    from .core import thermal_guard as _svk_thermal_guard
    _svk_thermal_guard.install()
except Exception as _e:
    print(f"[Thermal guard] not started: {_e}")

from .vk.nodes_boxes import NODES as VK_BOX, DISPLAY as VKD_BOX
from .vk.nodes_detect import NODES as VK_DET, DISPLAY as VKD_DET
from .vk.nodes_draw import NODES as VK_DRAW, DISPLAY as VKD_DRAW
from .vk.nodes_mask import NODES as VK_MASK, DISPLAY as VKD_MASK
from .vk.nodes_overlay import NODES as VK_OVL, DISPLAY as VKD_OVL
from .media import NODES as IO_NODES, DISPLAY as IO_DISPLAY
# Camera Realism: програмний реалізм камери (шум сенсора, байєр, ISP, оптика, тремтіння); план - docs/realism_plan.md
from .realism import NODES as RL_NODES, DISPLAY as RL_DISPLAY

WEB_DIRECTORY = "./web"

# HLG-формати для VHS Video Combine: VHS бере json і з теки VHS_video_formats (свої формати він мітить bt709)
import os as _os, folder_paths as _fp
_vf = _fp.folder_names_and_paths.get("VHS_video_formats", ((), set()))
_fp.folder_names_and_paths["VHS_video_formats"] = ([*_vf[0], _os.path.join(_os.path.dirname(__file__), "video_formats")], set(_vf[1]) | {".json"})

NODE_CLASS_MAPPINGS = {
    "LexYOLODetect": LexYOLODetect,
    "LexBoxesToBBOX": LexBoxesToBBOX,
    "LexHUDBoxes": LexHUDBoxes,
    "LexHUDSprite": LexHUDSprite,
    "LexDepthCurve": LexDepthCurve,
    "SVKDepthGrade": SVKDepthGrade,
    "SVKDepthGradeLive": SVKDepthGradeLive,
    "SVKDepthGradeView": SVKDepthGradeView,
    "SVKDepthGradeVideoView": SVKDepthGradeVideoView,
    "SVKDepthReceiver": SVKDepthReceiver,
    "SVKRawDepthSave": SVKRawDepthSave,
    "SVKRawDepthLoad": SVKRawDepthLoad,
    "SVKSaveDepth16": SVKSaveDepth16,
    **DL_NODES,
    **VK_DET, **VK_BOX, **VK_DRAW, **VK_OVL, **VK_MASK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LexYOLODetect": "YOLO Detect",
    "LexBoxesToBBOX": "Boxes → BBOX (for SAM2)",
    "LexHUDBoxes": "HUD Boxes (corners/labels)",
    "LexHUDSprite": "HUD Sprite (custom PNG frame + cutout)",
    "LexDepthCurve": "Depth Curve (DA3 → V2 look)",
    "SVKDepthGrade": "Depth Grade (grade + anti-flicker)",
    "SVKDepthGradeLive": "Depth Grade Live (legacy)",
    "SVKDepthGradeView": "Depth Grade View · Frame (calibration)",
    "SVKDepthGradeVideoView": "Depth Grade View · Video (live preview)",
    "SVKDepthReceiver": "Depth Receiver (raw DA3 → float depth)",
    "SVKRawDepthSave": "Raw Depth Save (DA3 geometry → EXR float32)",
    "SVKRawDepthLoad": "Raw Depth Load (EXR float32 → DA3 geometry)",
    "SVKSaveDepth16": "Save Depth 16-bit / EXR",
    **DL_DISPLAY,
    **VKD_DET, **VKD_BOX, **VKD_DRAW, **VKD_OVL, **VKD_MASK,
}

# Одна гілка меню на весь пак, дві підгілки за призначенням — щоб не шукати ноди
# в двох різних місцях, як було з двома паками.
_SUB = {"LexHUDSprite": "HUD", "LexHUDBoxes": "HUD", "VKDrawBoxes": "HUD",
        "VKOverlayOnBoxes": "HUD", "VKCutByMask": "HUD",
        "LexDepthCurve": "Depth", "SVKDepthGrade": "Depth", "SVKDepthGradeLive": "Depth", "SVKDepthGradeView": "Depth", "SVKDepthGradeVideoView": "Depth",
        "DepthLook": "Depth", "DepthLookSimple": "Depth", "DepthLookVideo": "Depth", "SVKVideoDepthStream": "Depth",
        "SVKDepthReceiver": "Depth", "SVKRawDepthSave": "Depth", "SVKRawDepthLoad": "Depth", "SVKSaveDepth16": "Depth",
        "LexYOLODetect": "Detect", "VKDetect": "Detect",
        "LexBoxesToBBOX": "Detect", "VKBoxesToBBOX": "Detect", "VKSelectBox": "Detect"}
NODE_CLASS_MAPPINGS.update(IO_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(IO_DISPLAY)
_SUB.update({k: "IO" for k in IO_NODES})
NODE_CLASS_MAPPINGS.update(RL_NODES)
NODE_DISPLAY_NAME_MAPPINGS.update(RL_DISPLAY)
_SUB.update({k: "Realism" for k in RL_NODES})
for _key, _cls in NODE_CLASS_MAPPINGS.items():
    _cls.CATEGORY = f"StepVisionKit/{_SUB.get(_key, 'misc')}"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
