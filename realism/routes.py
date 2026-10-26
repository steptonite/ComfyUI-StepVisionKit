"""aiohttp-маршрути Camera Realism (превʼю, лупа, власні пресети). Важке - в executor, як у depth_look."""
import asyncio

from . import preview, presets


def register_routes(routes):
    from aiohttp import web

    def _jr(code, obj):
        return web.json_response(obj, status=code, headers={"Cache-Control": "no-store"})

    async def _call(fn, *a):
        try:
            return _jr(200, await asyncio.get_running_loop().run_in_executor(None, fn, *a))
        except preview.NeedRun:
            return _jr(409, {"err": "run it once first", "need_run": True})
        except ValueError as e:
            return _jr(400, {"err": str(e)})
        except Exception as e:   # noqa: BLE001 - картка показує текст помилки, Comfy не падає
            print("[CameraRealism] preview error: %r" % e)
            return _jr(500, {"err": str(e)})

    @routes.get("/svk_realism/presets")
    async def rl_presets(request):
        return await _call(lambda: {"presets": presets.list_presets()})

    @routes.post("/svk_realism/presets/save")
    async def rl_save(request):
        b = await request.json()
        return await _call(lambda: {"name": presets.save_preset(b.get("name"), b.get("values")), "presets": presets.list_presets()})

    @routes.post("/svk_realism/presets/delete")
    async def rl_delete(request):
        b = await request.json()
        return await _call(lambda: {"name": presets.delete_preset(b.get("name")), "presets": presets.list_presets()})

    @routes.post("/svk_realism/{node}/render")
    async def rl_render(request):
        return await _call(preview.render, request.match_info["node"], await request.json())

    @routes.post("/svk_realism/{node}/loupe")
    async def rl_loupe(request):
        return await _call(preview.loupe, request.match_info["node"], await request.json())
