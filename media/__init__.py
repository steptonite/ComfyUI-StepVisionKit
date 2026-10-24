"""IO-ноди: RAW-фото, HDR-зображення (EXR / PNG 16), професійне відео (ProRes / DNxHR / HEVC 10 біт / EXR)."""
from . import raw, save_hdr, video

NODES = {**raw.NODES, **save_hdr.NODES, **video.NODES}
DISPLAY = {**raw.DISPLAY, **save_hdr.DISPLAY, **video.DISPLAY}
