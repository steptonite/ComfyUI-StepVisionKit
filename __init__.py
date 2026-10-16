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

from .lex.nodes_core import LexBoxesToBBOX, LexHUDBoxes, LexYOLODetect
from .lex.depth_curve import LexDepthCurve
from .lex.depth_grade import SVKDepthGrade, SVKDepthGradeLive, SVKDepthGradeView
from .lex.hud_sprite import LexHUDSprite

from .vk.nodes_boxes import NODES as VK_BOX, DISPLAY as VKD_BOX
from .vk.nodes_detect import NODES as VK_DET, DISPLAY as VKD_DET
from .vk.nodes_draw import NODES as VK_DRAW, DISPLAY as VKD_DRAW
from .vk.nodes_mask import NODES as VK_MASK, DISPLAY as VKD_MASK
from .vk.nodes_overlay import NODES as VK_OVL, DISPLAY as VKD_OVL

WEB_DIRECTORY = "./web"

NODE_CLASS_MAPPINGS = {
    "LexYOLODetect": LexYOLODetect,
    "LexBoxesToBBOX": LexBoxesToBBOX,
    "LexHUDBoxes": LexHUDBoxes,
    "LexHUDSprite": LexHUDSprite,
    "LexDepthCurve": LexDepthCurve,
    "SVKDepthGrade": SVKDepthGrade,
    "SVKDepthGradeLive": SVKDepthGradeLive,
    "SVKDepthGradeView": SVKDepthGradeView,
    **VK_DET, **VK_BOX, **VK_DRAW, **VK_OVL, **VK_MASK,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LexYOLODetect": "YOLO Detect",
    "LexBoxesToBBOX": "Boxes → BBOX (для SAM2)",
    "LexHUDBoxes": "HUD Boxes (кути/підписи)",
    "LexHUDSprite": "HUD Sprite (кастомна рамка PNG + дірка)",
    "LexDepthCurve": "Depth Curve (DA3 → вигляд як V2)",
    "SVKDepthGrade": "Depth Grade 🎚 (грейд глибини + антимерехтіння)",
    "SVKDepthGradeLive": "Depth Grade Live (стара, заміни на Depth Grade + Depth Grade View)",
    "SVKDepthGradeView": "Depth Grade View 👁 (живе прев'ю: повзунки без перегенерації)",
    **VKD_DET, **VKD_BOX, **VKD_DRAW, **VKD_OVL, **VKD_MASK,
}

# Одна гілка меню на весь пак, дві підгілки за призначенням — щоб не шукати ноди
# в двох різних місцях, як було з двома паками.
_SUB = {"LexHUDSprite": "HUD", "LexHUDBoxes": "HUD", "VKDrawBoxes": "HUD",
        "VKOverlayOnBoxes": "HUD", "VKCutByMask": "HUD",
        "LexDepthCurve": "Depth", "SVKDepthGrade": "Depth", "SVKDepthGradeLive": "Depth", "SVKDepthGradeView": "Depth",
        "LexYOLODetect": "Detect", "VKDetect": "Detect",
        "LexBoxesToBBOX": "Detect", "VKBoxesToBBOX": "Detect", "VKSelectBox": "Detect"}
for _key, _cls in NODE_CLASS_MAPPINGS.items():
    _cls.CATEGORY = f"StepVisionKit/{_SUB.get(_key, 'misc')}"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
