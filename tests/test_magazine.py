"""Offline coverage for source integrity, image provenance and local-only edition controls."""
import asyncio
import base64
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import edition_server
import image_engine
import magazine_engine as engine

SOURCE = {"id": "s1", "timestamp": "2026-10-24 12:00:00", "text": "We discussed the ocean. There are more questions to explore."}
DRAFT = {"title": "The questions below the surface", "subtitle": "A day of discussion about ocean research.",
    "themes": ["Ocean research"], "takeaways": [{"text": "Keep exploring the questions.", "source_ids": ["s1"]}],
    "stories": [{"title": "Listening to the ocean", "kicker": "Discovery", "paragraphs": ["The conversation turned to ocean research."],
                 "source_ids": ["s1"], "image_query": "ocean underwater", "image_prompt": "Sunlight through a calm ocean."}],
    "quote": {"text": "There are more questions to explore.", "source_id": "s1"}}


def context(payload, status=200, mime="application/json"):
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
    response = mock.Mock(status=status, content_type=mime)
    async def chunks(size):
        for pos in range(0, len(raw), size):
            yield raw[pos:pos+size]
    response.content.iter_chunked = chunks
    cm = mock.MagicMock()
    cm.__aenter__ = mock.AsyncMock(return_value=response)
    cm.__aexit__ = mock.AsyncMock(return_value=False)
    return cm


class SourceTests(unittest.TestCase):
    def test_rotations_are_combined_deduplicated_and_gaps_are_not_treated_as_speech(self):
        with tempfile.TemporaryDirectory() as root:
            source = {"event_id": "s1", "timestamp": SOURCE["timestamp"], "text": SOURCE["text"], "type": "transcript"}
            Path(root, "omi_transcript_events.jsonl.1").write_text(json.dumps(source) + "\n")
            gap = {"event_id": "g1", "type": "gap", "timestamp": SOURCE["timestamp"], "duration_sec": 2.5}
            Path(root, "omi_transcript_events.jsonl").write_text(json.dumps(source) + "\n" + json.dumps(gap) + '\n{"incomplete"')
            sources, gaps = engine.load_day(root, "2026-10-24")
            self.assertEqual(sources, [SOURCE])
            self.assertEqual(gaps, 2.5)

    def test_sources_and_quotes_are_checked_against_actual_evidence(self):
        result = engine.validate_edition(DRAFT, [SOURCE])
        self.assertEqual(result["sources"], [SOURCE])
        for mutation in ("reference", "quote"):
            draft = copy.deepcopy(DRAFT)
            if mutation == "reference":
                draft["stories"][0]["source_ids"] = ["invented-source"]
            else:
                draft["quote"]["text"] = "A quote that was never spoken."
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                engine.validate_edition(draft, [SOURCE])

    def test_missing_day_and_oversized_source_fail_explicitly(self):
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(ValueError):
                engine.load_day(root, "2026-10-24")
            Path(root, "omi_transcript_events.jsonl").write_text(json.dumps(SOURCE) + "\n")
            with mock.patch.object(engine, "MAX_SOURCE_CHARS", 10), self.assertRaises(ValueError):
                engine.load_day(root, "2026-10-24")

    def test_publish_preserves_private_file_mode_and_replaces_complete_json(self):
        with tempfile.TemporaryDirectory() as root:
            engine.publish_edition({"old": True}, root)
            engine.publish_edition({"new": True}, root)
            path = Path(root, "edition.json")
            self.assertEqual(json.loads(path.read_text()), {"new": True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertFalse(list(Path(root).glob(".edition-*.json")))


class GenerationTests(unittest.IsolatedAsyncioTestCase):
    async def test_analysis_rejects_truncated_results_and_uses_json_mode(self):
        session = mock.Mock()
        payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(DRAFT)}]}}]}
        session.post.return_value = context(payload)
        result = await engine.analyze_day(session, [SOURCE], {"GEMINI_API_KEY": "test-key"})
        self.assertEqual(result["sources"], [SOURCE])
        self.assertEqual(session.post.call_args.kwargs["json"]["generationConfig"]["responseMimeType"], "application/json")
        self.assertNotIn("test-key", session.post.call_args.args[0])
        payload["candidates"][0]["finishReason"] = "MAX_TOKENS"
        session.post.return_value = context(payload)
        with self.assertRaises(ValueError):
            await engine.analyze_day(session, [SOURCE], {"GEMINI_API_KEY": "test-key"})

    async def test_failed_regeneration_keeps_previous_edition(self):
        with tempfile.TemporaryDirectory() as root:
            source_dir = Path(root, "logs")
            source_dir.mkdir()
            Path(source_dir, "omi_transcript_events.jsonl").write_text(json.dumps(SOURCE) + "\n")
            output = Path(root, "editions")
            engine.publish_edition({"title": "Previous edition"}, output / "2026-10-24")
            with mock.patch.object(engine, "analyze_day", side_effect=ValueError("invalid")):
                with self.assertRaises(ValueError):
                    await engine.build("2026-10-24", "none", source_dir, output, {"GEMINI_API_KEY": "test"})
            self.assertEqual(json.loads((output / "2026-10-24" / "edition.json").read_text())["title"], "Previous edition")

    async def test_ai_images_are_saved_with_explicit_provenance(self):
        png = b"\x89PNG\r\n\x1a\n" + bytes(64)
        payload = {"candidates": [{"content": {"parts": [{"inlineData": {"mimeType": "image/png", "data": base64.b64encode(png).decode()}}]}}]}
        session = mock.Mock()
        session.post.return_value = context(payload)
        with tempfile.TemporaryDirectory() as root:
            result = await image_engine.generate_image(session, api_key="test", model="gemini-3.1-flash-image", prompt="Ocean", directory=root)
            self.assertEqual(result["kind"], "generated")
            self.assertEqual(Path(root, result["file"]).read_bytes(), png)
            self.assertIn("not a photograph", result["caption"])

    async def test_web_images_keep_credit_and_skip_untrusted_download_hosts(self):
        meta = {"LicenseShortName": {"value": "CC BY-SA 4.0"}, "Artist": {"value": "<a>Photographer</a>"}}
        bad = {"imageinfo": [{"extmetadata": meta, "thumburl": "http://127.0.0.1/private", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Bad"}], "index": 1}
        good = {"imageinfo": [{"extmetadata": meta, "thumburl": "https://upload.wikimedia.org/good.jpg", "descriptionurl": "https://commons.wikimedia.org/wiki/File:Good"}], "index": 2}
        session = mock.Mock()
        session.get.side_effect = [context({"query": {"pages": {"1": bad, "2": good}}}), context(b"\xff\xd8\xff" + bytes(64), mime="image/jpeg")]
        with tempfile.TemporaryDirectory() as root:
            result = await image_engine.find_web_image(session, query="ocean", directory=root)
            self.assertEqual(result["credit"], "Photographer")
            self.assertEqual(result["license"], "CC BY-SA 4.0")
            self.assertEqual(result["source_url"], "https://commons.wikimedia.org/wiki/File:Good")
            self.assertEqual(session.get.call_count, 2)
            self.assertEqual(session.get.call_args.args[0], "https://upload.wikimedia.org/good.jpg")

    async def test_oversized_download_is_rejected(self):
        cm = context(bytes(129))
        async with cm as response:
            with self.assertRaises(ValueError):
                await image_engine.read_bounded(response, 128)


class LocalJournalTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "omi_focus.html").write_text("<!doctype html><title>Omi Focus</title>")
        app = web.Application(client_max_size=4096)
        edition_server.install(app, self.root)
        self.service = app[edition_server.SERVICE]
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_focus_and_editions_are_served_without_caching_personal_content(self):
        response = await self.client.get("/focus")
        self.assertEqual(response.status, 200)
        self.assertIn("Omi Focus", await response.text())
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        engine.publish_edition({"day": "2026-10-24", "title": "Private edition"}, self.service.root / "2026-10-24")
        response = await self.client.get("/api/edition?date=2026-10-24")
        self.assertEqual((await response.json())["title"], "Private edition")
        self.assertEqual(response.headers["Cache-Control"], "no-store")
        response = await self.client.get("/api/edition?date=2026-99-99")
        self.assertEqual(response.status, 400)

    async def test_cross_origin_or_non_json_generation_is_rejected(self):
        with mock.patch.object(asyncio, "create_subprocess_exec") as spawn:
            response = await self.client.post("/api/edition-build", json={"day": "2026-10-24"}, headers={"Origin": "https://example.invalid"})
            self.assertEqual(response.status, 403)
            response = await self.client.post("/api/edition-build", data="day=2026-10-24")
            self.assertEqual(response.status, 415)
            spawn.assert_not_called()

    async def test_only_one_generation_job_can_run_and_cleanup_cancels_it(self):
        entered = asyncio.Event()
        async def run(day, mode):
            entered.set()
            await asyncio.Event().wait()
        with mock.patch.object(self.service, "run", side_effect=run):
            response = await self.client.post("/api/edition-build", json={"day": "2026-10-24", "images": "none"})
            self.assertEqual(response.status, 202)
            await asyncio.wait_for(entered.wait(), 1)
            response = await self.client.post("/api/edition-build", json={"day": "2026-10-24"})
            self.assertEqual(response.status, 409)
            response = await self.client.get("/api/edition-job")
            self.assertEqual((await response.json())["status"], "building")
            await self.service.cleanup(None)
            self.assertTrue(self.service.task.cancelled())

    async def test_asset_paths_cannot_escape_the_edition_directory(self):
        response = await self.client.get("/edition-assets/2026-10-24/edition.json")
        self.assertEqual(response.status, 404)
        day = self.service.root / "2026-10-24"
        day.mkdir(parents=True)
        secret = self.root / "outside.png"
        secret.write_bytes(b"outside")
        (day / ("a" * 32 + ".png")).symlink_to(secret)
        response = await self.client.get("/edition-assets/2026-10-24/" + "a" * 32 + ".png")
        self.assertEqual(response.status, 404)
