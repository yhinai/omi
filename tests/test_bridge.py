"""Offline regressions for audio retention, configuration, and shutdown."""
import asyncio
from collections import deque, namedtuple
from datetime import datetime
import importlib.util
import io
import json
import logging
import os
from pathlib import Path
import aiohttp
import subprocess
import sys
import struct
import wave
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from unittest.mock import patch

from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer, make_mocked_request

if os.environ.get("OMI_BRIDGE_PATH"):
    spec = importlib.util.spec_from_file_location("omi_bridge", os.environ["OMI_BRIDGE_PATH"])
    bridge = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bridge
    spec.loader.exec_module(bridge)
else:
    import omi_bridge as bridge

bridge.logger.addHandler(logging.NullHandler())
bridge.logger.propagate = False

SPEECH = b"\x00\x10" * 8000  # Half a second above the silence gate.
VersionInfo = namedtuple("VersionInfo", "major minor micro releaselevel serial")


def http_error(status):
    return aiohttp.ClientResponseError(SimpleNamespace(real_url="https://example.invalid"), (), status=status, message="mock")


def http_context(status=200, text="Recovered speech"):
    payload = {"candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": text}]}}]}
    response = mock.Mock(status=status, request_info=SimpleNamespace(real_url="https://example.invalid"), history=())
    response.json = mock.AsyncMock(return_value=payload)
    response.read = mock.AsyncMock(return_value=json.dumps(payload).encode())
    response.text = mock.AsyncMock(return_value=json.dumps(payload))
    response.raise_for_status = mock.Mock(side_effect=http_error(status) if status >= 400 else None)
    context = mock.MagicMock()
    context.__aenter__ = mock.AsyncMock(return_value=response)
    context.__aexit__ = mock.AsyncMock(return_value=False)
    return context


class RuntimeTests(unittest.TestCase):
    def runtime(self, version=(3, 15, 0, "final", 0), implementation="cpython", gil_disabled=0):
        runtime = SimpleNamespace(version_info=VersionInfo(*version),
                                  implementation=SimpleNamespace(name=implementation))
        config = SimpleNamespace(get_config_var=lambda name: gil_disabled)
        return mock.patch.multiple(bridge, sys=runtime, sysconfig=config)

    def test_standard_final_cpython315_patch_releases_are_supported(self):
        for micro in (0, 1, 99):
            for gil_disabled in (None, 0):
                with self.subTest(micro=micro, gil_disabled=gil_disabled), \
                     self.runtime(version=(3, 15, micro, "final", 0), gil_disabled=gil_disabled):
                    self.assertIsNone(bridge.require_python315())

    def test_other_python_versions_and_prereleases_are_rejected(self):
        versions = [(3, 14, 99, "final", 0), (3, 16, 0, "final", 0), (2, 7, 18, "final", 0)]
        versions += [(3, 15, 0, level, 1) for level in ("alpha", "beta", "candidate")]
        for version in versions:
            with self.subTest(version=version), self.runtime(version=version):
                with self.assertRaisesRegex(RuntimeError, "CPython 3\\.15"):
                    bridge.require_python315()

    def test_alternative_implementation_and_free_threaded_build_are_rejected(self):
        for implementation, gil_disabled in (("pypy", 0), ("cpython", 1)):
            with self.subTest(implementation=implementation, gil_disabled=gil_disabled), \
                 self.runtime(implementation=implementation, gil_disabled=gil_disabled):
                with self.assertRaisesRegex(RuntimeError, "CPython 3\\.15"):
                    bridge.require_python315()


class ImportTests(unittest.TestCase):
    def test_import_does_not_read_configuration_or_create_files(self):
        script = """
import aiohttp.web, importlib.util, os, sys
from unittest import mock
from unittest.mock import patch
sys.dont_write_bytecode = True
before = dict(os.environ)
spec = importlib.util.spec_from_file_location('omi_bridge_import_probe', sys.argv[1])
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
with mock.patch('builtins.open', side_effect=AssertionError('file access during import')), \\
     mock.patch('os.makedirs', side_effect=AssertionError('mkdir during import')):
    spec.loader.exec_module(module)
assert dict(os.environ) == before, 'import changed environment'
"""
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(bridge.__file__).resolve())],
            capture_output=True, text=True, timeout=10,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONDONTWRITEBYTECODE": "1"},
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_configuration_precedence_includes_gemini_key_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            local, fallback = Path(directory) / "local.env", Path(directory) / "fallback.env"
            local.write_text("GOOGLE_API_KEY=local-key\nOMI_CORE_WINDOW_SEC=6\n")
            fallback.write_text("GEMINI_API_KEY=fallback-key\nOMI_CORE_WINDOW_SEC=7\n")
            merged = bridge._load_dotenv(local_path=local, fallback_path=fallback, environ={})
            self.assertEqual(merged["GEMINI_API_KEY"], "local-key")
            self.assertEqual(merged["OMI_CORE_WINDOW_SEC"], "6")
            merged = bridge._load_dotenv(local_path=local, fallback_path=fallback,
                environ={"GOOGLE_API_KEY": "shell-key", "OMI_CORE_WINDOW_SEC": "5"})
            self.assertEqual(merged["GEMINI_API_KEY"], "shell-key")
            self.assertEqual(merged["OMI_CORE_WINDOW_SEC"], "5")

    def test_thinking_level_defaults_to_low_and_allows_high_override(self):
        with mock.patch.object(bridge, "GEMINI_API_KEY", "offline-key"), \
             mock.patch.object(bridge, "GEMINI_THINKING_LEVEL", "low"):
            bridge.configure({"GEMINI_API_KEY": "offline-key"})
            self.assertEqual(bridge.GEMINI_THINKING_LEVEL, "low")
            self.assertEqual(bridge.CORE_WINDOW_SEC, 8.0)
            bridge.configure({"GEMINI_API_KEY": "offline-key", "OMI_GEMINI_THINKING_LEVEL": "high"})
            self.assertEqual(bridge.GEMINI_THINKING_LEVEL, "high")

    def test_unsupported_minimal_or_unknown_thinking_levels_fail_configuration(self):
        with mock.patch.object(bridge, "GEMINI_API_KEY", "offline-key"), \
             mock.patch.object(bridge, "GEMINI_THINKING_LEVEL", "low"):
            for level in ("minimal", "fake"):
                with self.subTest(level=level), self.assertRaises(ValueError):
                    bridge.configure({"GEMINI_API_KEY": "offline-key", "OMI_GEMINI_THINKING_LEVEL": level})


class ResponseTests(unittest.TestCase):
    def response(self, text="spoken words", reason="STOP", parts=None):
        return {"candidates": [{"finishReason": reason,
                "content": {"parts": [{"text": text}] if parts is None else parts}}]}

    def assert_invalid(self, data, kind):
        with self.assertRaises(bridge.GeminiResponseError) as raised:
            bridge.parse_gemini_response(data)
        self.assertEqual(raised.exception.kind, kind)
        self.assertFalse(raised.exception.retryable)

    def test_complete_transcript_and_explicit_empty_output_have_distinct_outcomes(self):
        result = bridge.parse_gemini_response(self.response())
        self.assertEqual((result.text, result.outcome, result.finish_reason), ("spoken words", "speech", "STOP"))
        for text in ("", '""', "''", "   "):
            with self.subTest(text=text):
                result = bridge.parse_gemini_response(self.response(text=text))
                self.assertEqual((result.text, result.outcome), ("", "no_speech"))

    def test_ordinary_words_none_and_silence_are_preserved_as_speech(self):
        for text in ("none", "silence", "[silence]"):
            with self.subTest(text=text):
                result = bridge.parse_gemini_response(self.response(text=text))
                self.assertEqual((result.text, result.outcome), (text, "speech"))

    def test_null_prompt_feedback_is_accepted_for_a_completed_transcript(self):
        data = self.response()
        data["promptFeedback"] = None
        self.assertEqual(bridge.parse_gemini_response(data).text, "spoken words")

    def test_explicit_zero_reply_tokens_allow_omitted_empty_content(self):
        candidates = [{"finishReason": "STOP"}, {"finishReason": "STOP", "content": None},
                      {"finishReason": "STOP", "content": {}},
                      {"finishReason": "STOP", "content": {"parts": []}},
                      {"finishReason": "STOP", "content": {"parts": [{"text": ""}]}}]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                result = bridge.parse_gemini_response({"candidates": [candidate],
                    "promptFeedback": None, "usageMetadata": {"candidatesTokenCount": 0}})
                self.assertEqual((result.text, result.outcome), ("", "no_speech"))

    def test_zero_token_metadata_does_not_hide_contradictions_or_bad_field_types(self):
        candidates = [self.response()["candidates"][0],
                      {"finishReason": "STOP", "content": []},
                      {"finishReason": "STOP", "content": {"parts": None}},
                      self.response(parts=[{"text": 1}])["candidates"][0]]
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                self.assert_invalid({"candidates": [candidate],
                    "usageMetadata": {"candidatesTokenCount": 0}}, "invalid_response")
        self.assert_invalid({"candidates": [], "usageMetadata": {"candidatesTokenCount": 0}}, "invalid_response")
        for count in ("0", False):
            with self.subTest(count=count):
                self.assert_invalid({"candidates": [{"finishReason": "STOP"}],
                    "usageMetadata": {"candidatesTokenCount": count}}, "invalid_response")
        blocked = self.response(reason="SAFETY")
        blocked["usageMetadata"] = {"candidatesTokenCount": 0}
        self.assert_invalid(blocked, "blocked")
        truncated = self.response(reason="MAX_TOKENS")
        truncated["usageMetadata"] = {"candidatesTokenCount": 0}
        self.assert_invalid(truncated, "truncated")

    def test_omitted_reply_count_allows_empty_stop_only_when_token_totals_reconcile(self):
        usages = [{"promptTokenCount": 100, "totalTokenCount": 100},
                  {"promptTokenCount": 100, "thoughtsTokenCount": 20, "totalTokenCount": 120}]
        for usage in usages:
            for content in (None, {}, {"role": "model"}, {"parts": []}):
                with self.subTest(usage=usage, content=content):
                    result = bridge.parse_gemini_response({"candidates": [{"finishReason": "STOP", "content": content}],
                                                          "usageMetadata": usage})
                    self.assertEqual((result.text, result.outcome), ("", "no_speech"))

    def test_omitted_reply_count_rejects_missing_inconsistent_or_wrong_type_totals(self):
        usages = [{}, {"totalTokenCount": 100}, {"promptTokenCount": 100},
                  {"promptTokenCount": 100, "thoughtsTokenCount": 20, "totalTokenCount": 121},
                  {"promptTokenCount": True, "totalTokenCount": True},
                  {"promptTokenCount": 100, "thoughtsTokenCount": False, "totalTokenCount": 100},
                  {"promptTokenCount": -1, "totalTokenCount": -1},
                  {"promptTokenCount": 100.0, "totalTokenCount": 100.0},
                  {"promptTokenCount": 100, "totalTokenCount": 100, "candidatesTokenCount": None}]
        for usage in usages:
            with self.subTest(usage=usage):
                self.assert_invalid({"candidates": [{"finishReason": "STOP"}], "usageMetadata": usage},
                                    "invalid_response")
        reconciled = {"promptTokenCount": 100, "thoughtsTokenCount": 20, "totalTokenCount": 120}
        contradictory = self.response()
        contradictory["usageMetadata"] = reconciled
        self.assert_invalid(contradictory, "invalid_response")
        for reason, kind in (("SAFETY", "blocked"), ("MAX_TOKENS", "truncated")):
            data = {"candidates": [{"finishReason": reason}], "usageMetadata": reconciled}
            self.assert_invalid(data, kind)

    def test_thought_parts_are_excluded_from_transcript(self):
        result = bridge.parse_gemini_response(self.response(parts=[
            {"thought": True, "text": "internal reasoning"},
            {"text": "spoken"}, {"text": "words"},
        ]))
        self.assertEqual(result.text, "spoken words")

    def test_prompt_and_candidate_blocks_are_reported_as_blocked(self):
        self.assert_invalid({"promptFeedback": {"blockReason": "SAFETY"}}, "blocked")
        for reason in ("SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"):
            with self.subTest(reason=reason):
                self.assert_invalid(self.response(reason=reason), "blocked")

    def test_truncated_output_is_not_accepted_as_a_complete_transcript(self):
        self.assert_invalid(self.response(reason="MAX_TOKENS"), "truncated")

    def test_missing_candidates_or_text_are_invalid_instead_of_silence(self):
        for data in ({}, {"candidates": []}, self.response(parts=[]),
                     self.response(parts=[{"thought": True, "text": "reasoning only"}])):
            with self.subTest(data=data):
                self.assert_invalid(data, "invalid_response")

    def test_wrong_response_field_types_are_rejected(self):
        values = [None, [], {"candidates": "bad"}, {"candidates": [None]},
                  {"candidates": [{"finishReason": "STOP", "content": []}]},
                  self.response(parts="bad"), self.response(parts=["bad"]),
                  self.response(parts=[{"text": 1}]), self.response(reason=None)]
        for data in values:
            with self.subTest(data=data):
                self.assert_invalid(data, "invalid_response")


class AudioTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [
            mock.patch.object(bridge, "stt_queue", asyncio.Queue(maxsize=24)),
            mock.patch.object(bridge, "GEMINI_API_KEY", "offline-test-key"),
            mock.patch.object(bridge, "record_transcript_line"),
            mock.patch.object(bridge, "flush_daily_paragraph"),
        ]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        bridge.audio_pcm_buffer.clear()
        bridge.audio_pre_tail.clear()
        bridge.audio_sessions.clear()
        bridge.current_session = None
        bridge._next_session_id = 0
        bridge.stopping = False
        bridge.background_tasks.clear()
        bridge.webhook_tasks.clear()
        bridge.live_listeners.clear()
        bridge.rolling_research_buffer.clear()
        bridge.audio_ring.clear()
        bridge.audio_ring_bytes = 0
        bridge.audio_sample_rate = 16000
        for name in ("windows_processed", "windows_retried", "windows_dropped"):
            bridge.metrics[name] = 0
        for name in ("last_packet_at", "last_stt_success_at", "last_error"):
            bridge.metrics[name] = None
        for name in bridge.metrics:
            if name.startswith("windows_"):
                bridge.metrics[name] = 0
        bridge.metrics["last_transcript_at"] = None
        bridge.metrics["transcripts_emitted"] = 0
        bridge.metrics["last_outcome"] = None

    async def enqueue(self, reset=False):
        await bridge.enqueue_stt_window(SPEECH, 16000, 0, 0.5, 0, reset_context_after=reset)

    async def consume(self):
        await bridge.stt_queue.put(None)
        await asyncio.wait_for(bridge.stt_consumer_worker(), 2)
        await asyncio.wait_for(bridge.stt_queue.join(), 1)

    async def test_full_queue_evicts_oldest_and_balances_task_accounting(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        await bridge.enqueue_stt_window(b"\x01\x02", 16000, 0, 1/16000, 0)
        self.assertEqual(bridge.stt_queue.get_nowait().pcm, b"\x01\x02")
        bridge.stt_queue.task_done()
        await asyncio.wait_for(bridge.stt_queue.join(), 1)
        self.assertEqual(bridge.metrics["windows_dropped"], 1)
        self.assertEqual(bridge.metrics["windows_evicted"], 1)

    async def test_shutdown_full_queue_waits_instead_of_evicting(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        bridge.stopping = True
        waiting = asyncio.create_task(self.enqueue())
        try:
            await asyncio.sleep(0)
            self.assertFalse(waiting.done())
            bridge.stt_queue.get_nowait()
            bridge.stt_queue.task_done()
            await asyncio.wait_for(waiting, 1)
            bridge.stt_queue.get_nowait()
            bridge.stt_queue.task_done()
            await asyncio.wait_for(bridge.stt_queue.join(), 1)
            self.assertEqual(bridge.metrics["windows_dropped"], 0)
        finally:
            waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)

    async def test_evicted_window_resets_context_before_next_retained_audio(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        entered, release = asyncio.Event(), asyncio.Event()
        contexts = []
        async def transcribe(*args):
            contexts.append(args[-1])
            if len(contexts) == 1:
                entered.set()
                await release.wait()
            return bridge.STTResult("earlier spoken words")
        with mock.patch.object(bridge, "gemini_stt_request", side_effect=transcribe):
            worker = asyncio.create_task(bridge.stt_consumer_worker())
            try:
                await bridge.enqueue_stt_window(SPEECH, 16000, 0, .5, 0, session_id=1, sample_offset=0)
                await asyncio.wait_for(entered.wait(), 1)
                for offset in (8000, 16000):
                    await bridge.enqueue_stt_window(SPEECH, 16000, 0, .5, 0, session_id=1, sample_offset=offset)
                release.set()
                await asyncio.wait_for(bridge.stt_queue.join(), 1)
                await bridge.stt_queue.put(None)
                await asyncio.wait_for(worker, 1)
                self.assertEqual(contexts, ["", ""])
                self.assertEqual(bridge.metrics["windows_evicted"], 1)
            finally:
                release.set()
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)

    async def test_transient_failures_retry_three_times_with_backoff(self):
        errors = [http_error(code)
                  for code in (429, 500, 501, 599)]
        errors += [TimeoutError("mock timeout"), aiohttp.ClientConnectionError("mock connection reset")]
        for error in errors:
            with self.subTest(error=repr(error)):
                await self.enqueue()
                with mock.patch.object(bridge, "gemini_stt_request",
                        side_effect=[error, error, error, bridge.STTResult("Recovered speech")]) as request, \
                     mock.patch.object(bridge.asyncio, "sleep", new_callable=mock.AsyncMock) as sleep:
                    await self.consume()
                self.assertEqual(request.call_count, 4)
                self.assertEqual([call.args[0] for call in sleep.await_args_list], [1, 2, 4])
                self.assertEqual(bridge.record_transcript_line.call_args.args[0], "Recovered speech")

    async def test_permanent_client_error_is_not_retried(self):
        await self.enqueue()
        error = http_error(400)
        with mock.patch.object(bridge, "gemini_stt_request", side_effect=error) as request:
            await self.consume()
        self.assertEqual(request.call_count, 1)
        self.assertEqual(bridge.metrics["windows_retried"], 0)
        bridge.record_transcript_line.assert_not_called()

    async def test_async_error_response_is_released_before_retry_backoff(self):
        await self.enqueue()
        rejected, recovered = http_context(429), http_context()
        session = mock.Mock(closed=False)
        session.post.side_effect = [rejected, recovered]

        async def check_backoff(delay):
            rejected.__aexit__.assert_awaited_once()

        with mock.patch.object(bridge, "http_session", session), \
             mock.patch.object(bridge.asyncio, "sleep", side_effect=check_backoff) as sleep:
            await self.consume()
        self.assertEqual(session.post.call_count, 2)
        sleep.assert_awaited_once_with(1.0)
        recovered.__aexit__.assert_awaited_once()
        self.assertEqual(bridge.record_transcript_line.call_args.args[0], "Recovered speech")

    async def test_generation_payload_sets_configured_thinking_without_temperature(self):
        for level in ("low", "high"):
            with self.subTest(level=level):
                session = mock.Mock(closed=False)
                session.post.return_value = http_context()
                with mock.patch.object(bridge, "http_session", session), \
                     mock.patch.object(bridge, "GEMINI_THINKING_LEVEL", level):
                    result = await bridge.gemini_stt_request(SPEECH, 16000, 0, 0.5, 0, "")
                self.assertEqual(result.outcome, "speech")
                config = session.post.call_args.kwargs["json"]["generationConfig"]
                self.assertEqual(config["thinkingConfig"]["thinkingLevel"], level)
                self.assertNotIn("temperature", config)

    async def test_async_permanent_response_is_released_when_window_is_dropped(self):
        await self.enqueue()
        rejected = http_context(400)
        session = mock.Mock(closed=False)
        session.post.return_value = rejected
        with mock.patch.object(bridge, "http_session", session):
            await self.consume()
        self.assertEqual(session.post.call_count, 1)
        rejected.__aexit__.assert_awaited_once()
        bridge.record_transcript_line.assert_not_called()

    async def test_rejected_provider_diagnostics_exclude_transcript_and_credentials(self):
        secret_text = "private conversation sentinel"
        context = http_context(text=secret_text)
        response = context.__aenter__.return_value
        response.json.return_value = {"candidates": [{"finishReason": "MAX_TOKENS",
            "content": {"parts": [{"text": secret_text}]}}],
            "usageMetadata": {"candidatesTokenCount": 10}}
        session = mock.Mock(closed=False)
        session.post.return_value = context
        with mock.patch.object(bridge, "http_session", session), mock.patch.object(bridge.logger, "warning") as warning:
            with self.assertRaises(bridge.GeminiResponseError):
                await bridge.gemini_stt_request(SPEECH, 16000, 0, 0.5, 0, "")
        formatted = warning.call_args.args[0] % warning.call_args.args[1:]
        self.assertIn("MAX_TOKENS", formatted)
        self.assertNotIn(secret_text, formatted)
        self.assertNotIn("offline-test-key", formatted)
        context.__aexit__.assert_awaited_once()

    async def test_retry_exhaustion_finishes_queue_item_without_recording_text(self):
        await self.enqueue()
        with mock.patch.object(bridge, "gemini_stt_request", side_effect=TimeoutError("mock")) as request, \
             mock.patch.object(bridge.asyncio, "sleep", new_callable=mock.AsyncMock):
            await self.consume()
        self.assertEqual(request.call_count, 4)
        bridge.record_transcript_line.assert_not_called()

    async def test_context_is_sequential_and_resets_after_empty_session_flush(self):
        await self.enqueue()
        await self.enqueue()
        await bridge.flush_remaining_buffer(reset_context=True)
        await self.enqueue()
        with mock.patch.object(bridge, "gemini_stt_request",
                side_effect=[bridge.STTResult(text) for text in ("alpha beta", "alpha beta gamma", "alpha beta")]) as request:
            await self.consume()
        self.assertEqual([call.args[-1] for call in request.call_args_list], ["", "alpha beta", ""])
        self.assertEqual([call.args[0] for call in bridge.record_transcript_line.call_args_list],
                         ["alpha beta", "gamma", "alpha beta"])

    async def test_retries_and_no_speech_keep_the_previous_speech_context(self):
        for _ in range(3):
            await self.enqueue()
        results = [bridge.STTResult("alpha beta"), aiohttp.ClientConnectionError("mock"),
                   bridge.STTResult("", "no_speech"), bridge.STTResult("alpha beta gamma")]
        with mock.patch.object(bridge, "gemini_stt_request", side_effect=results) as request, \
             mock.patch.object(bridge.asyncio, "sleep", new_callable=mock.AsyncMock):
            await self.consume()
        self.assertEqual([call.args[-1] for call in request.call_args_list],
                         ["", "alpha beta", "alpha beta", "alpha beta"])
        self.assertEqual([call.args[0] for call in bridge.record_transcript_line.call_args_list],
                         ["alpha beta", "gamma"])
        self.assertEqual(bridge.metrics["windows_no_speech"], 1)
        self.assertEqual(bridge.metrics["windows_retried"], 1)

    async def test_provider_rejection_is_visible_and_is_not_retried_as_silence(self):
        for kind in ("blocked", "truncated", "invalid_response"):
            with self.subTest(kind=kind):
                await self.enqueue()
                with mock.patch.object(bridge, "gemini_stt_request",
                        side_effect=bridge.GeminiResponseError(kind)) as request:
                    await self.consume()
                self.assertEqual(request.await_count, 1)
                self.assertIsNotNone(bridge.metrics["last_error"])
        bridge.record_transcript_line.assert_not_called()
        self.assertIsNone(bridge.metrics["last_stt_success_at"])
        self.assertIsNone(bridge.metrics["last_transcript_at"])

    async def test_valid_no_speech_does_not_claim_a_transcript_was_emitted(self):
        await self.enqueue()
        with mock.patch.object(bridge, "gemini_stt_request", return_value=bridge.STTResult("", "no_speech")):
            await self.consume()
        self.assertEqual(bridge.metrics["windows_processed"], 1)
        self.assertEqual(bridge.metrics["windows_no_speech"], 1)
        self.assertEqual(bridge.metrics["transcripts_emitted"], 0)
        self.assertIsNotNone(bridge.metrics["last_stt_success_at"])
        self.assertIsNone(bridge.metrics["last_transcript_at"])
        self.assertIsNone(bridge.metrics["last_error"])
        bridge.record_transcript_line.assert_not_called()

    async def test_subsecond_tail_is_queued_and_context_does_not_cross_session(self):
        await self.enqueue()
        bridge.ingest_raw_pcm(SPEECH)
        await bridge.flush_remaining_buffer(reset_context=True)
        self.assertEqual(bridge.audio_pcm_buffer, bytearray())
        self.assertEqual(bridge.audio_pre_tail, bytearray())
        await self.enqueue()
        with mock.patch.object(bridge, "gemini_stt_request",
                side_effect=[bridge.STTResult(text) for text in ("first segment", "tail segment", "new session")]) as request:
            await self.consume()
        self.assertEqual(len(request.call_args_list[1].args[0]), len(SPEECH))
        self.assertEqual(request.call_args_list[1].args[3], 0.5)
        self.assertEqual(request.call_args_list[2].args[-1], "")

    async def test_silence_is_not_sent_to_gemini(self):
        await bridge.enqueue_stt_window(b"\0" * 16000, 16000, 0, 0.5, 0)
        with mock.patch.object(bridge, "gemini_stt_request") as request:
            await self.consume()
        request.assert_not_called()

    async def test_slicer_flushes_subsecond_audio_only_after_eight_seconds_idle(self):
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock):
            bridge.ingest_raw_pcm(SPEECH)
            slicer = asyncio.create_task(bridge.window_slicer_worker())
            try:
                now[0] = 108.0
                await asyncio.sleep(0.25)
                self.assertTrue(bridge.stt_queue.empty())
                now[0] = 108.01
                for _ in range(300):
                    if not bridge.stt_queue.empty():
                        break
                    await asyncio.sleep(0.001)
                window = bridge.stt_queue.get_nowait()
                self.assertEqual(window.pcm, SPEECH)
                self.assertEqual(window.core_sec, 0.5)
                self.assertTrue(window.reset_context_after)
                bridge.stt_queue.task_done()
                self.assertEqual(bridge.buffered_audio_bytes(), 0)
            finally:
                slicer.cancel()
                await asyncio.gather(slicer, return_exceptions=True)

    async def test_new_session_does_not_join_previously_sliced_audio(self):
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock):
            bridge.ingest_raw_pcm(SPEECH)
            old = bridge.current_session
            now[0] = 109.0
            slicer = asyncio.create_task(bridge.window_slicer_worker())
            try:
                for _ in range(300):
                    if old.closed:
                        break
                    await asyncio.sleep(0.001)
                self.assertTrue(old.closed)
                self.assertEqual(bytes(old.pcm), b"")
                new_audio = b"\x00\x20" * 8000
                bridge.ingest_raw_pcm(new_audio)
                self.assertEqual(len(bridge.audio_sessions), 1)
                self.assertIsNot(old, bridge.current_session)
                self.assertEqual(bytes(bridge.current_session.pcm), new_audio)
                window = await asyncio.wait_for(bridge.stt_queue.get(), 1)
                bridge.stt_queue.task_done()
                self.assertEqual(window.pcm, SPEECH)
                self.assertEqual(window.session_id, old.session_id)
                self.assertNotEqual(window.session_id, bridge.current_session.session_id)
                self.assertEqual(bytes(bridge.current_session.pcm), new_audio)
            finally:
                slicer.cancel()
                await asyncio.gather(slicer, return_exceptions=True)

    async def test_ingress_limit_counts_all_pending_sessions(self):
        now = [100.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        with mock.patch.object(bridge, "time", clock), mock.patch.object(bridge, "MAX_BUFFER_BYTES", 8):
            bridge.ingest_raw_pcm(b"\x01\x01" * 2)
            now[0] = 109.0
            bridge.ingest_raw_pcm(b"\x02\x02" * 2)
            self.assertEqual(len(bridge.audio_sessions), 2)
            bridge.ingest_raw_pcm(b"\x03\x03")
            self.assertEqual(bridge.buffered_audio_bytes(), 8)
            self.assertEqual(bridge.audio_ring_bytes, 10)
            self.assertEqual(bytes(bridge.audio_sessions[0].pcm), b"\x01\x01")
            self.assertEqual(bridge.audio_sessions[0].consumed_samples, 1)
            self.assertEqual(bridge.metrics["buffer_bytes_evicted"], 2)

    async def test_audio_offsets_and_receipt_times_survive_eviction_and_overlap(self):
        now = [100.0]
        wall = [1000.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        clock.time.side_effect = lambda: wall[0]
        bridge.stt_queue = asyncio.Queue(maxsize=1)
        await self.enqueue()
        with mock.patch.object(bridge, "time", clock), \
             mock.patch.object(bridge, "find_dynamic_cut_offset", return_value=5 * 32000):
            bridge.ingest_raw_pcm(SPEECH * 14)  # Seven seconds, first received at epoch 1000.
            session = bridge.current_session
            waiting = asyncio.create_task(bridge._slice_one(session))
            await asyncio.sleep(0)
            self.assertTrue(waiting.done())
            self.assertEqual(session.consumed_samples, 80000)
            now[0], wall[0] = 105.0, 1005.0
            bridge.ingest_raw_pcm(SPEECH * 10)
            await asyncio.wait_for(waiting, 1)
            first = bridge.stt_queue.get_nowait()
            bridge.stt_queue.task_done()
            await bridge._slice_one(session)
            second = bridge.stt_queue.get_nowait()
            bridge.stt_queue.task_done()
        self.assertEqual((first.audio_start_at, first.received_at), (993.0, 1000.0))
        self.assertEqual((second.audio_start_at, second.received_at), (998.0, 1005.0))
        self.assertEqual((first.pre_sec, second.pre_sec), (0, 1))
        self.assertEqual((first.sample_offset, second.sample_offset), (0, 80000))
        self.assertEqual(session.consumed_samples, 160000)

    async def test_bursty_backfill_clamps_audio_time_without_losing_sample_offsets(self):
        now, wall = [100.0], [1000.0]
        clock = mock.Mock(wraps=bridge.time)
        clock.monotonic.side_effect = lambda: now[0]
        clock.time.side_effect = lambda: wall[0]
        with mock.patch.object(bridge, "time", clock), \
             mock.patch.object(bridge, "find_dynamic_cut_offset", return_value=5 * 32000):
            bridge.ingest_raw_pcm(SPEECH * 14)
            session = bridge.current_session
            await bridge._slice_one(session)
            now[0], wall[0] = 100.2, 1000.2
            bridge.ingest_raw_pcm(SPEECH * 14)
            await bridge._slice_one(session)
            await bridge._slice_one(session, final=True)
        windows = [bridge.stt_queue.get_nowait() for _ in range(3)]
        for window in windows:
            bridge.stt_queue.task_done()
            self.assertLessEqual(window.audio_start_at + window.core_sec + window.post_sec,
                                 window.received_at + 1e-9)
            self.assertEqual(window.session_id, session.session_id)
        self.assertEqual([window.sample_offset for window in windows], [0, 80000, 160000])
        for window, expected in zip(windows, (993.0, 994.2, 996.2)):
            self.assertAlmostEqual(window.audio_start_at, expected)
        self.assertEqual(session.consumed_samples, 224000)

    async def test_shutdown_drains_an_inflight_request_beyond_fifteen_seconds(self):
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_request(*args):
            entered.set()
            await release.wait()
            return bridge.STTResult("Accepted audio survives shutdown")

        async def background():
            await asyncio.Event().wait()

        await self.enqueue()
        app = {"slicer_task": asyncio.create_task(background()),
               "research_task": asyncio.create_task(background()),
               "stt_worker_task": asyncio.create_task(bridge.stt_consumer_worker())}
        cleanup = None
        with mock.patch.object(bridge, "gemini_stt_request", side_effect=slow_request):
            try:
                await asyncio.wait_for(entered.wait(), 1)
                cleanup = asyncio.create_task(bridge.on_cleanup(app))
                for _ in range(4):
                    await asyncio.sleep(0)
                loop = asyncio.get_running_loop()
                original_time, debug = loop.time, loop.get_debug()
                loop.set_debug(False)
                await asyncio.sleep(0)
                try:
                    with mock.patch.object(loop, "time", lambda: original_time() + 20):
                        for _ in range(6):
                            await asyncio.sleep(0)
                        release.set()
                        await cleanup
                finally:
                    loop.set_debug(debug)
                bridge.record_transcript_line.assert_called_once()
                self.assertEqual(bridge.record_transcript_line.call_args.args[0], "Accepted audio survives shutdown")
                self.assertFalse(app["stt_worker_task"].cancelled())
                await bridge.stt_queue.join()
            finally:
                release.set()
                tasks = [*app.values(), *([cleanup] if cleanup else [])]
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


class RouteTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.patches = [mock.patch.object(bridge, "stt_queue", asyncio.Queue(maxsize=24)),
                        mock.patch.object(bridge, "GEMINI_API_KEY", ""),
                        mock.patch.object(bridge, "flush_daily_paragraph")]
        for patch in self.patches:
            patch.start()
        self.addCleanup(lambda: [patch.stop() for patch in reversed(self.patches)])
        bridge.audio_pcm_buffer.clear()
        bridge.audio_pre_tail.clear()
        bridge.audio_sessions.clear()
        bridge.current_session = None
        bridge._next_session_id = 0
        bridge.stopping = False

    async def client_for(self, app):
        client = TestClient(TestServer(app))
        await client.start_server()
        self.addAsyncCleanup(client.close)
        return client

    async def test_omi_setup_response_remains_compatible_without_gemini_key(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        response = await client.get("/omi")
        self.assertEqual(response.status, 200)
        data = await response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data, {"status": "ok", "is_setup_completed": True, "message": ""})

    async def test_overloaded_ingress_returns_200_and_retains_ring_audio(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        with mock.patch.object(bridge, "MAX_BUFFER_BYTES", 4):
            for raw in (b"\x01\x01" * 2, b"\x02\x02" * 2):
                response = await client.post("/omi", data=raw)
                self.assertEqual(response.status, 200)
        self.assertEqual(bytes(bridge.current_session.pcm), b"\x02\x02" * 2)
        self.assertEqual(bridge.audio_ring[-2][2] + bridge.audio_ring[-1][2],
                         b"\x01\x01" * 2 + b"\x02\x02" * 2)

    async def test_packet_larger_than_pending_budget_is_accepted_and_retained(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        raw = b"\x01\x02" + bytes(bridge.MAX_BUFFER_BYTES)
        response = await client.post("/omi", data=raw)
        self.assertEqual(response.status, 200)
        self.assertEqual(bytes(bridge.current_session.pcm), raw[-bridge.MAX_BUFFER_BYTES:])
        self.assertEqual(bridge.audio_ring[-1][2], raw)
        self.assertEqual(bridge.current_session.consumed_samples, 1)

    async def test_http_packet_size_limit_is_independent_of_pending_budget(self):
        with mock.patch.object(bridge, "RING_BUFFER_SECONDS", 1):
            client = await self.client_for(bridge.create_app(start_workers=False))
            raw = bytes(bridge.BYTES_PER_SECOND)
            response = await client.post("/omi", data=raw)
            self.assertEqual(response.status, 200)
            response = await client.post("/omi", data=raw + b"\0\0")
            self.assertEqual(response.status, 413)

    async def test_health_is_separate_and_existing_monitor_routes_remain_available(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for path in ("/health", "/omi/unexpected"):
            with self.subTest(path=path):
                response = await client.get(path)
                self.assertEqual(response.status, 404)
        for path in ("/listen", "/audio/last"):
            response = await client.get(path)
            self.assertEqual(response.status, 200)
        async with client.ws_connect("/listen-ws"):
            pass

    async def test_local_health_app_exposes_telemetry_separately(self):
        client = await self.client_for(bridge.create_health_app())
        response = await client.get("/health")
        self.assertEqual(response.status, 503)
        data = await response.json()
        self.assertIn("queue_depth", data)
        self.assertEqual(data["thinking_level"], bridge.GEMINI_THINKING_LEVEL)
        self.assertFalse(data["gemini_key_configured"])
        response = await client.get("/omi")
        self.assertEqual(response.status, 404)

    async def test_health_returns_success_only_when_workers_are_healthy(self):
        client = await self.client_for(bridge.create_health_app())
        alive = mock.Mock()
        alive.done.return_value = False
        with mock.patch.object(bridge, "GEMINI_API_KEY", "offline-key"), \
             mock.patch.object(bridge, "background_tasks", {"worker": alive}), \
             mock.patch.dict(bridge.metrics, {"last_error": None}):
            response = await client.get("/health")
            self.assertEqual(response.status, 200)
            self.assertEqual((await response.json())["status"], "ok")
            alive.done.return_value = True
            response = await client.get("/health")
            self.assertEqual(response.status, 503)
            self.assertEqual((await response.json())["status"], "degraded")

    async def test_pcm_ingress_validates_sample_rate_and_frame_alignment(self):
        client = await self.client_for(bridge.create_app(start_workers=False))
        for rate in ("0", "bad", "44100"):
            with self.subTest(rate=rate):
                response = await client.post("/omi?sample_rate=" + rate, data=SPEECH,
                                             headers={"Content-Type": "application/octet-stream"})
                self.assertEqual(response.status, 400)
        response = await client.post("/omi", data=b"\x01", headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status, 400)
        response = await client.post("/omi", data=SPEECH, headers={"Content-Type": "application/octet-stream"})
        self.assertEqual(response.status, 200)


class TimestampTests(unittest.TestCase):
    def test_audio_time_controls_midnight_notes_and_timing_metadata_is_retained(self):
        source = datetime(2026, 10, 24, 23, 59, 59).timestamp()
        processed = datetime(2026, 10, 25, 0, 2, 0).timestamp()
        transcript_stream, event_stream = io.StringIO(), io.StringIO()
        transcript, events = logging.Logger("test_transcript"), logging.Logger("test_events")
        handlers = [logging.StreamHandler(transcript_stream), logging.StreamHandler(event_stream)]
        for target, handler in zip((transcript, events), handlers):
            handler.setFormatter(logging.Formatter("%(message)s"))
            target.setLevel(logging.INFO)
            target.addHandler(handler)
        try:
            with tempfile.TemporaryDirectory() as directory:
                broadcast, webhook = mock.Mock(), mock.Mock()
                with mock.patch.multiple(bridge, transcript_logger=transcript, event_logger=events,
                        daily_notes=bridge.DailyNotes(directory), WAKE_WORD="trigger", HERMES_RESEARCH_WEBHOOK="",
                        broadcast=broadcast, schedule_webhook=webhook):
                    bridge.record_transcript_line("trigger spoken words", audio_start_at=source,
                                                   received_at=source + 0.5, processed_at=processed,
                                                   session_id=42, sample_offset=80000)
                self.assertIn("[2026-10-24 23:59:59] trigger spoken words", transcript_stream.getvalue())
                self.assertTrue(Path(directory, "2026-10-24.md").exists())
                self.assertFalse(Path(directory, "2026-10-25.md").exists())
                event = json.loads(event_stream.getvalue())
                self.assertEqual(event["audio_start_at"], source)
                self.assertEqual(event["received_at"], source + 0.5)
                self.assertEqual(event["processed_at"], processed)
                self.assertEqual(event["time_basis"], "estimated_from_receipt")
                self.assertEqual((event["session_id"], event["sample_offset"]), (42, 80000))
                self.assertEqual(json.loads(broadcast.call_args.args[0])["audio_start_at"], source)
                self.assertEqual(json.loads(webhook.call_args.args[1])["processed_at"], processed)
        finally:
            for handler in handlers:
                handler.close()
            transcript_stream.close()
            event_stream.close()



class BridgeTests(unittest.IsolatedAsyncioTestCase):
    bridge = bridge

    async def asyncSetUp(self):
        b = self.bridge
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        for name, value in (("GEMINI_API_KEY", "offline-key"), ("LOG_DIR", temp.name),
                            ("daily_notes", None), ("http_session", None)):
            active = patch.object(b, name, value)
            active.start()
            self.addCleanup(active.stop)
        b.stt_queue = None
        b.audio_pcm_buffer = bytearray()
        b.audio_pre_tail = bytearray()
        b.audio_ring = deque()
        b.audio_ring_bytes = 0
        b.audio_sessions = deque()
        b.current_session = None
        b._next_session_id = 0
        b.stopping = False
        b.background_tasks.clear()
        b.webhook_tasks.clear()
        b.live_listeners.clear()
        b.daily_notes = None
        b.rolling_research_buffer.clear()
        for key in ("windows_processed", "windows_retried", "windows_dropped", "windows_skipped", "windows_evicted", "buffer_bytes_evicted"):
            b.metrics[key] = 0
        for key in ("last_error", "last_packet_at", "last_stt_success_at"):
            b.metrics[key] = None
        b.metrics["stt_in_flight"] = False
        self.app = None
        self.calls = []
        self.lines = []
        async def transcribe(*args):
            self.calls.append(args)
            return b.STTResult("sample transcript")
        self.api = patch.object(b, "gemini_stt_request", side_effect=transcribe).start()
        patch.object(b, "record_transcript_line", side_effect=lambda text, **timing: self.lines.append(text)).start()
        self.addCleanup(patch.stopall)
        self.pcm = struct.pack("<h", 500) * 16000

    async def asyncTearDown(self):
        if self.app is not None:
            await self.bridge.on_shutdown(self.app)
            await self.bridge.on_cleanup(self.app)

    async def start(self):
        self.app = {}
        await self.bridge.on_startup(self.app)
        await asyncio.sleep(.01)

    async def finish_window(self):
        await asyncio.wait_for(self.bridge.stt_queue.join(), 2)

    async def test_queue_is_created_on_server_loop(self):
        self.assertIsNone(self.bridge.stt_queue)
        await self.start()
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.lines, ["sample transcript"])
        self.assertTrue(self.bridge.build_health_dict()["workers_alive"])

    async def test_permanent_error_is_not_retried(self):
        await self.start()
        self.api.side_effect = http_error(401)
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.api.call_count, 1)
        self.assertEqual(self.bridge.metrics["windows_dropped"], 1)
        self.assertEqual(self.bridge.build_health_dict()["status"], "degraded")

    async def test_silence_is_retained_but_not_sent_to_gemini(self):
        await self.start()
        silent = bytes(16000)
        self.bridge.ingest_raw_pcm(silent)
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        self.assertEqual(b"".join(chunk[2] for chunk in self.bridge.audio_ring), silent)
        self.api.assert_not_called()
        self.assertEqual(self.bridge.metrics["windows_skipped"], 1)

    async def test_short_tail_and_empty_boundary_reset_context(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:2])
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(len(self.calls[0][0]), 2)
        self.assertEqual(self.calls[1][-1], "")
        await self.bridge.flush_remaining_buffer()  # No samples, but still a session boundary.
        await self.bridge.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await self.finish_window()
        self.assertEqual(self.calls[2][-1], "")

    async def test_new_packet_closes_previous_idle_session(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:2])
        self.bridge.current_session.last_packet_at -= 9
        self.bridge.ingest_raw_pcm(self.pcm[:4])
        await self.bridge.flush_remaining_buffer()
        await self.finish_window()
        self.assertEqual([len(c[0]) for c in self.calls], [2, 4])
        self.assertEqual([c[-1] for c in self.calls], ["", ""])

    async def test_real_slicer_flushes_idle_subsecond_tail(self):
        await self.start()
        self.bridge.ingest_raw_pcm(self.pcm[:160])
        self.bridge.current_session.last_packet_at -= 9
        await asyncio.sleep(.25)
        await self.finish_window()
        self.assertEqual(len(self.calls[0][0]), 160)
        self.assertFalse(self.bridge.audio_sessions)

    async def test_ring_exactly_trims_partial_oldest_packet(self):
        b = self.bridge
        chunk = self.pcm * 7
        for _ in range(43):
            b.audio_pcm_buffer.clear()
            b.ingest_raw_pcm(chunk)
        self.assertEqual(b.audio_ring_bytes, 9_600_000)
        response = await b.handle_audio_last(make_mocked_request("GET", "/audio/last?seconds=300"))
        with wave.open(io.BytesIO(response.body)) as wav:
            self.assertEqual(wav.getnframes(), 300 * 16000)
            self.assertEqual(wav.readframes(300 * 16000), (chunk * 43)[-9_600_000:])

    async def test_audio_export_and_invalid_parameters(self):
        self.bridge.ingest_raw_pcm(self.pcm)
        response = await self.bridge.handle_audio_last(make_mocked_request("GET", "/audio/last?seconds=1"))
        with wave.open(io.BytesIO(response.body)) as wav:
            self.assertEqual((wav.getnchannels(), wav.getsampwidth(), wav.getframerate()), (1, 2, 16000))
            self.assertEqual(wav.readframes(16000), self.pcm)
        for value in ("bad", "0", "301"):
            with self.assertRaises(web.HTTPBadRequest):
                await self.bridge.handle_audio_last(make_mocked_request("GET", f"/audio/last?seconds={value}"))

    async def test_invalid_samples_rejected_but_oversized_pcm_keeps_fresh_suffix(self):
        b = self.bridge
        with self.assertRaises(web.HTTPBadRequest):
            b.ingest_raw_pcm(self.pcm, 8000)
        with self.assertRaises(web.HTTPBadRequest):
            b.ingest_raw_pcm(b"x")
        raw = b"\x01\x02" + bytes(b.MAX_BUFFER_BYTES)
        b.ingest_raw_pcm(raw)
        self.assertEqual(bytes(b.audio_pcm_buffer), raw[-b.MAX_BUFFER_BYTES:])
        self.assertEqual(b.audio_ring_bytes, len(raw))
        self.assertEqual(b.metrics["buffer_bytes_evicted"], 2)
        self.assertEqual(b.current_session.consumed_samples, 1)

    async def test_dead_worker_is_not_healthy(self):
        await self.start()
        worker = self.app["slicer_task"]
        worker.cancel()
        await asyncio.gather(worker, return_exceptions=True)
        response = await self.bridge.handle_health(None)
        self.assertEqual(response.status, 503)
        self.assertEqual(json.loads(response.body)["status"], "degraded")
        self.assertFalse(json.loads(response.body)["workers_alive"])

    async def test_shutdown_drains_full_queue_and_short_tail(self):
        await self.start()
        b = self.bridge
        for _ in range(24):
            await b.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        b.ingest_raw_pcm(self.pcm[:2])
        await b.on_shutdown(self.app)
        with self.assertRaises(web.HTTPServiceUnavailable):
            b.ingest_raw_pcm(self.pcm)
        await asyncio.wait_for(b.on_cleanup(self.app), 3)
        self.app = None
        self.assertEqual(self.api.call_count, 25)
        self.assertEqual(b.metrics["windows_dropped"], 0)
        self.assertTrue(b.stt_queue.empty())

    async def test_shared_http_session_stays_open_until_accepted_audio_drains(self):
        await self.start()
        b = self.bridge
        session = b.http_session
        self.assertIsNotNone(session)
        self.assertFalse(session.closed)
        entered, release = asyncio.Event(), asyncio.Event()

        async def slow_transcript(*args):
            entered.set()
            await release.wait()
            return b.STTResult("drained transcript")

        self.api.side_effect = slow_transcript
        await b.enqueue_stt_window(self.pcm, 16000, 0, 1, 0)
        await asyncio.wait_for(entered.wait(), 1)
        cleanup = asyncio.create_task(b.on_cleanup(self.app))
        try:
            for _ in range(4):
                await asyncio.sleep(0)
            self.assertFalse(session.closed)
            release.set()
            await asyncio.wait_for(cleanup, 2)
            self.app = None
            self.assertTrue(session.closed)
            self.assertEqual(self.lines, ["drained transcript"])
        finally:
            release.set()
            if not cleanup.done():
                cleanup.cancel()
                await asyncio.gather(cleanup, return_exceptions=True)

    async def test_http_ingress_to_worker_and_playable_export(self):
        b = self.bridge
        await self.start()
        app = web.Application()
        app.router.add_post("/omi", b.handle_omi)
        app.router.add_get("/audio/last", b.handle_audio_last)
        async with TestClient(TestServer(app)) as client:
            response = await client.post("/omi?sample_rate=16000", data=self.pcm)
            self.assertEqual(response.status, 200)
            await b.flush_remaining_buffer()
            await self.finish_window()
            response = await client.get("/audio/last?seconds=1")
            with wave.open(io.BytesIO(await response.read())) as wav:
                self.assertEqual(wav.readframes(16000), self.pcm)
        self.assertEqual(self.lines, ["sample transcript"])


class NotesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.notes_class = bridge.DailyNotes

    def test_pause_word_limit_midnight_and_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            notes = self.notes_class(directory)
            entries = [("2026-10-24 23:59:40", "first"),
                       ("2026-10-24 23:59:48", "second"),
                       ("2026-10-24 23:59:57", "after pause"),
                       ("2026-10-25 00:00:02", "word " * 120),
                       ("2026-10-25 00:00:07", "next paragraph")]
            for stamp, text in entries:
                notes.add(stamp, text)
            first = Path(directory, "2026-10-24.md").read_text()
            second = Path(directory, "2026-10-25.md").read_text()
            self.assertIn("first second", first)
            self.assertEqual(first.count("**["), 2)
            self.assertEqual(second.count("**["), 2)
            source = Path(directory, "source.log")
            source.write_text("".join(f"[{stamp}] {' '.join(text.split())}\n" for stamp, text in entries))
            rebuilt = self.notes_class(directory)
            rebuilt.rebuild(source)
            self.assertEqual(first, Path(directory, "2026-10-24.md").read_text())
            self.assertEqual(second, Path(directory, "2026-10-25.md").read_text())
            rebuilt.add("2026-10-25 00:00:12", "continued after restart")
            self.assertIn("next paragraph continued after restart", Path(directory, "2026-10-25.md").read_text())


if __name__ == "__main__":
    unittest.main()
