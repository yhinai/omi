"""Local journal routes; generation runs in a separate bounded subprocess."""
import asyncio
from datetime import date
import json
from pathlib import Path
import re
import sys

from aiohttp import web

SERVICE = web.AppKey("edition_service", object)
HEADERS = {"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
           "Referrer-Policy": "no-referrer", "X-Frame-Options": "DENY"}


def day_value(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise web.HTTPBadRequest(text="Use a date in YYYY-MM-DD format")
    try:
        date.fromisoformat(value)
    except ValueError:
        raise web.HTTPBadRequest(text="Invalid edition date") from None
    return value


class EditionService:
    def __init__(self, base_dir):
        self.base = Path(base_dir)
        self.root = self.base / "logs" / "editions"
        self.job = {"status": "idle"}
        self.task = None
        self.process = None

    async def focus(self, request):
        return web.FileResponse(self.base / "omi_focus.html", headers=HEADERS)

    async def preview_asset(self, request):
        return web.FileResponse(self.base / "assets" / "ocean-editorial.png", headers=HEADERS)

    async def edition(self, request):
        day = day_value(request.query.get("date", date.today().isoformat()))
        path = self.root / day / "edition.json"
        if not path.exists() or not path.resolve().is_relative_to(self.root.resolve()):
            raise web.HTTPNotFound(text="No edition for this date")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            raise web.HTTPServiceUnavailable(text="Edition is unavailable") from None
        return web.json_response(data, headers=HEADERS)

    async def asset(self, request):
        day = day_value(request.match_info["day"])
        name = request.match_info["name"]
        if not re.fullmatch(r"[a-f0-9]{32}\.(?:png|jpg|webp)", name):
            raise web.HTTPNotFound()
        path = self.root / day / name
        if not path.is_file() or not path.resolve().is_relative_to(self.root.resolve()):
            raise web.HTTPNotFound()
        return web.FileResponse(path, headers=HEADERS)

    async def status(self, request):
        return web.json_response(self.job, headers=HEADERS)

    async def start(self, request):
        # JSON + same-origin checks prevent a random website/form starting paid jobs.
        if request.host.rsplit(":", 1)[0] not in ("127.0.0.1", "localhost"):
            raise web.HTTPForbidden(text="Use the local journal address")
        origin = request.headers.get("Origin")
        if origin and origin != str(request.url.origin()):
            raise web.HTTPForbidden(text="Generation requires the local journal")
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType(text="Use JSON")
        if request.content_length and request.content_length > 4096:
            raise web.HTTPRequestEntityTooLarge(max_size=4096, actual_size=request.content_length)
        try:
            data = await request.json()
        except (ValueError, UnicodeError):
            raise web.HTTPBadRequest(text="Invalid JSON") from None
        if not isinstance(data, dict):
            raise web.HTTPBadRequest(text="Expected an object")
        day = day_value(data.get("day"))
        mode = data.get("images", "web")
        if mode not in ("web", "ai", "none"):
            raise web.HTTPBadRequest(text="Choose web, ai or none")
        if self.task is not None and not self.task.done():
            raise web.HTTPConflict(text="An edition is already being prepared")
        self.job = {"status": "building", "day": day, "images": mode}
        self.task = asyncio.create_task(self.run(day, mode))
        return web.json_response(self.job, status=202, headers=HEADERS)

    async def run(self, day, mode):
        try:
            self.process = await asyncio.create_subprocess_exec(
                sys.executable, "-B", str(self.base / "magazine_engine.py"), "--date", day, "--images", mode,
                cwd=str(self.base), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
            async with asyncio.timeout(270):
                code = await self.process.wait()
            self.job["status"] = "ready" if code == 0 else "failed"
        except asyncio.CancelledError:
            self.job["status"] = "interrupted"
            raise
        except (OSError, TimeoutError):
            self.job["status"] = "failed"
        finally:
            if self.process is not None and self.process.returncode is None:
                try:
                    self.process.terminate()
                except ProcessLookupError:
                    pass
                try:
                    await asyncio.wait_for(self.process.wait(), 5)
                except TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            self.process = None

    async def cleanup(self, app):
        if self.task is not None and not self.task.done():
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)


def install(app, base_dir):
    service = EditionService(base_dir)
    app[SERVICE] = service
    app.router.add_get("/", service.focus)
    app.router.add_get("/focus", service.focus)
    app.router.add_get("/focus-assets/ocean-editorial.png", service.preview_asset)
    app.router.add_get("/api/edition", service.edition)
    app.router.add_get("/api/edition-job", service.status)
    app.router.add_post("/api/edition-build", service.start)
    app.router.add_get("/edition-assets/{day}/{name}", service.asset)
    app.on_cleanup.append(service.cleanup)
