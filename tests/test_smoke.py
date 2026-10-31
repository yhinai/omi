"""Verify the manual smoke check against a local provider without paid API calls."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
import logging
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

from aiohttp import web
from aiohttp.test_utils import TestServer
import live_stt
import omi_bridge as bridge

spec = importlib.util.spec_from_file_location("omi_smoke", Path(__file__).resolve().parents[1] / "scripts/smoke_test.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_isolated_check_verifies_real_files_and_leaves_normal_logs_untouched(self):
        async def provider(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for message in ws:
                data = json.loads(message.data)
                if "setup" in data:
                    await ws.send_json({"setupComplete": {}})
                realtime = data.get("realtimeInput", {})
                if "audio" in realtime:
                    await ws.send_json({"serverContent": {"interimInputTranscription": {"text": "Test phrase"}}})
                if realtime.get("audioStreamEnd"):
                    await ws.send_json({"serverContent": {"inputTranscription": {"text": "Test phrase."}}})
                    await ws.send_json({"serverContent": {"generationComplete": True}})
            return ws

        app = web.Application()
        app.router.add_get("/provider", provider)
        with tempfile.TemporaryDirectory() as directory:
            wav = Path(directory, "speech.wav")
            with wave.open(str(wav), "wb") as audio:
                audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                audio.writeframes(b"\x00\x10" * 1600)
            original = Path(directory, "normal-logs")
            original.mkdir()
            sentinel = original / "keep.txt"
            sentinel.write_text("existing history")
            events, diagnostics = logging.Logger("smoke-events"), logging.Logger("smoke-diagnostics")
            verify = smoke.verify_saved
            def verify_outputs(root, finals, expected):
                self.assertFalse(Path(root, "omi_live_transcript.log").exists())
                verify(root, finals, expected)
            try:
                async with TestServer(app) as server:
                    with mock.patch.object(bridge, "state", bridge.BridgeState()), \
                         mock.patch.object(bridge, "configure"), \
                         mock.patch.object(bridge, "GEMINI_API_KEY", "offline"), \
                         mock.patch.object(bridge, "STREAM_TOKEN", "test-token"), \
                         mock.patch.object(bridge, "LOG_DIR", str(original)), \
                         mock.patch.object(bridge, "logger", diagnostics), \
                         mock.patch.object(bridge, "event_logger", events), \
                         mock.patch.object(live_stt, "LIVE_URL", str(server.make_url("/provider"))), \
                         mock.patch.object(smoke, "verify_saved", side_effect=verify_outputs) as verification, \
                         mock.patch.object(smoke.logging, "shutdown"), redirect_stdout(io.StringIO()) as output:
                        await smoke.check(SimpleNamespace(wav=wav, expect="Test phrase", timeout=8))
                        verification.assert_called_once()
                        self.assertEqual(json.loads(output.getvalue())["status"], "passed")
                        self.assertIsNone(bridge.state.http_session)
                self.assertEqual(list(original.iterdir()), [sentinel])
                self.assertEqual(sentinel.read_text(), "existing history")
            finally:
                for target in (events, diagnostics):
                    for handler in target.handlers:
                        handler.close()

    def test_saved_check_rejects_missing_markdown_partial_and_gap(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "omi_transcript_events.jsonl")
            final = {"type": "transcript", "is_final": True, "event_id": "one", "text": "Test phrase"}
            path.write_text(json.dumps(final) + "\n")
            with self.assertRaisesRegex(RuntimeError, "Markdown"):
                smoke.verify_saved(directory, [final], "Test phrase")
            Path(directory, "2026-10-31.md").write_text("Test phrase")
            smoke.verify_saved(directory, [final], "Test phrase")
            for kind in ("partial", "gap"):
                path.write_text(json.dumps(final) + "\n" + json.dumps({"type": kind}) + "\n")
                with self.subTest(kind=kind), self.assertRaisesRegex(RuntimeError, "finalized"):
                    smoke.verify_saved(directory, [final], "Test phrase")
