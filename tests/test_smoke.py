"""Verify the manual smoke check against a fake local worker without loading a model."""
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
import sys

import bridge
import stt

spec = importlib.util.spec_from_file_location("omi_smoke", Path(__file__).resolve().parents[1] / "scripts/smoke_test.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class SmokeTests(unittest.IsolatedAsyncioTestCase):
    async def test_isolated_check_verifies_real_files_and_leaves_normal_logs_untouched(self):
        worker = """import sys,json,base64
print(json.dumps({'type':'ready','load_warm_sec':0}),flush=True)
first=None;end=0
for line in sys.stdin:
 r=json.loads(line)
 if r['op']=='packet':
  if first is None:first=r
  end=r['sample_offset']+len(base64.b64decode(r['pcm_base64']))//2
  print(json.dumps({'type':'partial','utterance_id':'test','text':'Test phrase'}),flush=True)
 elif r['op'] in ('flush','shutdown'):
  if first is not None:
   print(json.dumps({'type':'final','utterance_id':'test','session_id':first['session_id'],'sample_offset':first['sample_offset'],'sample_end_offset':end,'received_at':first['received_at'],'audio_start_at':first['audio_start_at'],'text':'Test phrase.'}),flush=True)
   first=None
  print(json.dumps({'type':'covered','sample_end_offset':end}),flush=True)
  print(json.dumps({'type':'flushed','request_id':r.get('request_id')}),flush=True)
  if r['op']=='shutdown':break
"""
        factory = stt.LocalWhisperSTT
        def make_worker(**kwargs):
            stream = factory(**kwargs)
            stream.command = [sys.executable, "-u", "-c", worker]
            return stream
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
                with mock.patch.object(bridge, "state", bridge.BridgeState()), \
                     mock.patch.object(bridge, "configure"), \
                     mock.patch.object(bridge, "STREAM_TOKEN", "test-token"), \
                     mock.patch.object(bridge, "LOG_DIR", str(original)), \
                     mock.patch.object(bridge, "logger", diagnostics), \
                     mock.patch.object(bridge, "event_logger", events), \
                     mock.patch.object(stt, "LocalWhisperSTT", side_effect=make_worker), \
                     mock.patch.object(smoke, "verify_saved", side_effect=verify_outputs) as verification, \
                     mock.patch.object(smoke.logging, "shutdown"), redirect_stdout(io.StringIO()) as output:
                    await smoke.check(SimpleNamespace(wav=wav, expect="Test phrase", timeout=8))
                    verification.assert_called_once()
                    self.assertEqual(json.loads(output.getvalue())["status"], "passed")
                    self.assertFalse(hasattr(bridge.state, "http_session"))
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
