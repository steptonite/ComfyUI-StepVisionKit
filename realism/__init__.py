from .node import NODES, DISPLAY

try:   # у Comfy PromptServer.instance є; у мікротесті нема - тоді маршрути не реєструємо
    from server import PromptServer  # type: ignore
    if getattr(PromptServer, "instance", None) is not None:
        from .routes import register_routes
        register_routes(PromptServer.instance.routes)
except Exception as _e:   # noqa: BLE001
    if "No module named 'server'" not in str(_e):
        print("[CameraRealism] routes not registered: %s" % _e)
