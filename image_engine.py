#!/usr/bin/env python3.15
"""Bounded editorial images: Gemini illustrations or credited Wikimedia photographs."""
import base64
from html import unescape
import json
from pathlib import Path
import re
import uuid
from urllib.parse import urlsplit

import aiohttp

MAX_IMAGE_BYTES = 12 * 1024 * 1024


def plain(value):
    return unescape(re.sub(r"<[^>]+>", "", value or "")).strip()[:500]


async def read_bounded(response, limit):
    data = bytearray()
    async for chunk in response.content.iter_chunked(65536):
        data.extend(chunk)
        if len(data) > limit:
            raise ValueError("Image service response exceeds size limit")
    return bytes(data)


async def checked_json(response, limit=24 * 1024 * 1024):
    if response.status != 200:
        raise RuntimeError(f"Image service returned HTTP {response.status}")
    value = json.loads(await read_bounded(response, limit))
    if not isinstance(value, dict):
        raise ValueError("Image service returned an invalid object")
    return value


def save_image(data, mime, directory):
    extensions = {"image/png": ("png", data.startswith(b"\x89PNG\r\n\x1a\n")),
                  "image/jpeg": ("jpg", data.startswith(b"\xff\xd8\xff")),
                  "image/webp": ("webp", data[:4] == b"RIFF" and data[8:12] == b"WEBP")}
    if mime not in extensions or not extensions[mime][1] or len(data) > MAX_IMAGE_BYTES:
        raise ValueError("Unsupported or oversized editorial image")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + "." + extensions[mime][0]
    target = directory / name
    target.write_bytes(data)
    target.chmod(0o600)
    return name


async def generate_image(session, *, api_key, model, prompt, directory):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", model):
        raise ValueError("Invalid image model ID")
    request = {"contents": [{"parts": [{"text": (
        "Create a refined, photographic editorial illustration for a personal daily journal. "
        "No words, logos, watermarks or depictions of identifiable people. Do not imply this is "
        "documentary evidence of an actual event. Landscape composition, natural light. Subject: " + prompt[:1600])}]}],
        "generationConfig": {"responseModalities": ["TEXT", "IMAGE"],
                             "imageConfig": {"aspectRatio": "16:9"}}}
    async with session.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            json=request, headers={"x-goog-api-key": api_key}, allow_redirects=False,
            timeout=aiohttp.ClientTimeout(total=120)) as response:
        data = await checked_json(response)
    for candidate in data.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if isinstance(inline, dict) and isinstance(inline.get("data"), str):
                raw = base64.b64decode(inline["data"], validate=True)
                name = save_image(raw, inline.get("mimeType", inline.get("mime_type")), directory)
                return {"file": name, "kind": "generated", "credit": "AI-generated illustration · Gemini",
                        "caption": "An editorial illustration, not a photograph of the recorded event.",
                        "source_url": "", "license": "AI-generated"}
    raise ValueError("Image model returned no image")


async def find_web_image(session, *, query, directory):
    # Only public topic keywords go to Commons, never raw transcript text.
    query = " ".join(query.split())[:120]
    params = {"action": "query", "generator": "search", "gsrsearch": query + " filetype:bitmap",
              "gsrnamespace": "6", "gsrlimit": "6", "prop": "imageinfo", "iiprop": "url|extmetadata",
              "iiurlwidth": "1600", "format": "json"}
    async with session.get("https://commons.wikimedia.org/w/api.php", params=params,
            headers={"User-Agent": "OmiPersonalJournal/1.0 (https://github.com/yhinai/omi)"},
            allow_redirects=False, timeout=aiohttp.ClientTimeout(total=30)) as response:
        data = await checked_json(response, 2 * 1024 * 1024)
    pages = sorted(data.get("query", {}).get("pages", {}).values(), key=lambda page: page.get("index", 999))
    for page in pages:
        for info in page.get("imageinfo", []):
            meta = info.get("extmetadata", {})
            license_name = plain(meta.get("LicenseShortName", {}).get("value", ""))
            if not re.fullmatch(r"(?:CC0|Public domain|CC BY(?:-SA)? [\d.]+)", license_name, re.I):
                continue
            url = info.get("thumburl", info.get("url", ""))
            parsed = urlsplit(url)
            source = info.get("descriptionurl", "")
            if parsed.scheme != "https" or parsed.hostname != "upload.wikimedia.org":
                continue
            if not source.startswith("https://commons.wikimedia.org/wiki/"):
                continue
            try:
                async with session.get(url, allow_redirects=False,
                        headers={"User-Agent": "OmiPersonalJournal/1.0 (https://github.com/yhinai/omi)"},
                        timeout=aiohttp.ClientTimeout(total=30)) as response:
                    if response.status != 200:
                        continue
                    raw = await read_bounded(response, MAX_IMAGE_BYTES)
                    name = save_image(raw, response.content_type, directory)
            except (aiohttp.ClientError, TimeoutError, ValueError):
                continue
            return {"file": name, "kind": "web", "source_url": source, "license": license_name,
                    "credit": plain(meta.get("Artist", {}).get("value", "Wikimedia Commons")),
                    "caption": plain(meta.get("ObjectName", {}).get("value", page.get("title", "")))[:220]}
    raise ValueError("No suitable licensed photograph found for this topic")


async def create_image(session, *, mode, query, prompt, directory, env):
    if mode == "ai":
        return await generate_image(session, api_key=env["GEMINI_API_KEY"],
            model=env.get("OMI_IMAGE_MODEL", "gemini-3.1-flash-image"), prompt=prompt, directory=directory)
    if mode == "web":
        return await find_web_image(session, query=query, directory=directory)
    return None


def main():
    """Prepare a fresh cover without re-running the conversation analysis."""
    import argparse
    import asyncio
    import fcntl
    from magazine_engine import EDITION_DIR, checked_date, publish_edition
    from omi_bridge import _load_dotenv, require_python315
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--mode", choices=("web", "ai"), default="web")
    args = parser.parse_args()
    async def run():
        require_python315()
        directory = EDITION_DIR / checked_date(args.date)
        with (directory / ".build.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            edition = json.loads((directory / "edition.json").read_text())
            story = edition["stories"][0]
            async with aiohttp.ClientSession(cookie_jar=aiohttp.DummyCookieJar()) as session:
                edition["cover"] = await create_image(session, mode=args.mode, query=story["image_query"],
                    prompt=story["image_prompt"], directory=directory, env=_load_dotenv())
            edition.pop("image_warning", None)
            publish_edition(edition, directory)
            print(json.dumps({"status": "ready", "day": args.date, "mode": args.mode}))
    try:
        asyncio.run(run())
    except Exception as error:
        print(json.dumps({"status": "failed", "error": type(error).__name__}))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
