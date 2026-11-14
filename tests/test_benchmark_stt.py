"""Synthetic benchmark lifecycle, measurement semantics and redaction without models/network."""
import asyncio
import argparse
from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import wave

import bridge

spec = importlib.util.spec_from_file_location("omi_benchmark", Path(__file__).resolve().parents[1] / "scripts/benchmark_stt.py")
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class Clock:
    def __init__(self):
        self.value = 100.0

    def now(self):
        return self.value

    async def sleep(self, delay):
        self.value += max(0, delay)
        await asyncio.sleep(0)


class FakeBackend:
    def __init__(self, callbacks, *, lazy=False, text="Test phrase.", early_final=False,
                 flush_result=True, run_failure=False, gap=False):
        self.callbacks = callbacks
        self.lazy, self.text, self.early_final = lazy, text, early_final
        self.flush_result, self.run_failure, self.gap = flush_result, run_failure, gap
        self.connected = self.ready = False
        self.stopped = asyncio.Event()
        self.closed = False
        self.feed_sizes = []
        self.flush_timeout = None
        self.resolved_model = "test-resolved-model"

    async def run(self):
        if self.run_failure:
            raise RuntimeError("provider-url?key=private-secret")
        self.ready = self.lazy
        self.connected = not self.lazy
        if self.connected:
            self.callbacks["on_connection"](True, None)
        await self.stopped.wait()

    def feed(self, raw, received_at, received_mono):
        if not (self.ready or self.connected):
            raise RuntimeError("Fed before ready")
        self.feed_sizes.append(len(raw))
        self.connected = True
        if len(self.feed_sizes) == 1:
            if self.text:
                self.callbacks["on_partial"]({"text": "Test phrase"})
            if self.early_final and self.text:
                self.callbacks["on_final"](self.text)
            if self.gap:
                self.callbacks["on_gap"]("failed", .1, detail="private-secret")
        self.callbacks["on_latency"]("local_decode_sec", .02)

    async def flush_now(self, timeout=5):
        self.flush_timeout = timeout
        if not self.early_final and self.text:
            self.callbacks["on_final"](self.text)
        return self.flush_result

    async def close(self):
        self.closed = True
        self.stopped.set()
        return True


class AccuracyTests(unittest.TestCase):
    def test_wer_measures_edits_and_normalizes_known_numeric_formats(self):
        result = benchmark.word_error_rate("one two three", "one three")
        self.assertAlmostEqual(result["wer"], 1 / 3)
        self.assertEqual(result["word_errors"], 1)
        result = benchmark.word_error_rate("API status two hundred, seven requests in one point five seconds.",
                                           "A P I status 200: 7 requests in 1.5 seconds!")
        self.assertEqual(result["wer"], 0)
        self.assertEqual(benchmark.word_error_rate("", "unexpected")["wer"], None)
        self.assertEqual(benchmark.word_error_rate("", "unexpected")["word_errors"], 1)

    def test_backend_selection_is_local_by_default_and_rejects_unknown_names(self):
        self.assertEqual(benchmark.parse_backends("whisper, deepgram,whisper"), ["whisper", "deepgram"])
        with self.assertRaises(argparse.ArgumentTypeError):
            benchmark.parse_backends("provider-with-private-key")

    def test_synthesis_only_uses_known_phrases_and_returns_fixed_silence_control(self):
        with tempfile.TemporaryDirectory() as directory:
            def synthesize(command, **kwargs):
                if command[0] == "/usr/bin/afconvert":
                    with wave.open(command[-1], "wb") as audio:
                        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                        audio.writeframes(b"\x00\x10" * 1600)
                return subprocess.CompletedProcess(command, 0)
            with mock.patch.object(benchmark.subprocess, "run", side_effect=synthesize) as run:
                clips = benchmark.synthetic_clips(directory)
            self.assertEqual(len(clips), 4)
            self.assertEqual([clip.expected for clip in clips[:3]], [phrase for _, phrase in benchmark.PHRASES])
            self.assertEqual([call.args[0][-1] for call in run.call_args_list if call.args[0][0] == "/usr/bin/say"],
                             [phrase for _, phrase in benchmark.PHRASES])
            self.assertEqual(clips[-1].pcm, bytes(benchmark.BPS * 3))
            self.assertEqual(clips[-1].expected, "")
            self.assertEqual(clips[0].clip_bytes, 3200)
            self.assertEqual(len(clips[0].pcm), 3200 + benchmark.BPS * 2)


class LifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def run_trial(self, clip=None, **settings):
        clock = Clock()
        instances = []
        def factory(callbacks):
            backend = FakeBackend(callbacks, **settings)
            instances.append(backend)
            return backend
        clip = clip or benchmark.Clip("speech", "Test phrase.", b"\x00\x10" * 3200, 6400)
        result = await benchmark.trial("deepgram" if settings.get("lazy") else "whisper", clip, factory,
                                       sleep=clock.sleep, monotonic=clock.now)
        return result, instances[0]

    async def test_eager_and_lazy_backends_pace_audio_flush_and_close(self):
        for lazy in (False, True):
            with self.subTest(lazy=lazy):
                result, backend = await self.run_trial(lazy=lazy)
                self.assertEqual(result["status"], "passed")
                self.assertEqual(result["wer"], 0)
                self.assertAlmostEqual(result["sent_audio_sec"], .2)
                self.assertEqual(backend.feed_sizes, [3200, 3200])
                self.assertEqual(backend.flush_timeout, 30)
                self.assertTrue(backend.closed)
                self.assertEqual(result["provider_resolved_model"], "test-resolved-model")
                self.assertAlmostEqual(result["first_partial_from_first_feed_sec"], 0)
                self.assertAlmostEqual(result["first_correct_prefix_from_first_feed_sec"], 0)
                self.assertAlmostEqual(result["last_final_relative_to_clip_end_sec"], 0)

    async def test_final_before_clip_end_remains_negative(self):
        result, _ = await self.run_trial(early_final=True)
        self.assertAlmostEqual(result["last_final_relative_to_clip_end_sec"], -.2)

    async def test_twenty_ms_frames_preserve_realtime_pacing_and_short_tail(self):
        clock = Clock()
        backend = FakeBackend({})
        def factory(callbacks):
            backend.callbacks = callbacks
            return backend
        clip = benchmark.Clip("speech", "Test phrase.", b"\x00\x10" * 2000, 4000)
        result = await benchmark.trial("deepgram", clip, factory, chunk_ms=20,
                                       sleep=clock.sleep, monotonic=clock.now)
        self.assertEqual(result["status"], "passed")
        self.assertEqual(backend.feed_sizes, [640] * 6 + [160])
        self.assertAlmostEqual(result["sent_audio_sec"], .125)
        self.assertAlmostEqual(result["final_callbacks_from_first_feed_sec"][0], .125)
        self.assertEqual(result["feed_chunk_ms"], 20)
        self.assertTrue(backend.closed)

    async def test_silence_has_no_wer_and_rejects_any_partial_or_final_text(self):
        clip = benchmark.Clip("silence", "", bytes(3200), 3200)
        result, _ = await self.run_trial(clip, text="")
        self.assertEqual(result["status"], "passed")
        self.assertIsNone(result["wer"])
        self.assertIsNone(result["first_partial_from_first_feed_sec"])
        result, _ = await self.run_trial(clip)
        self.assertEqual(result["failure"], "SilenceProducedText")

    async def test_provider_failures_flush_failures_and_gaps_are_redacted_and_close(self):
        for settings, expected in (({"run_failure": True}, "BackendFailure"),
                                    ({"flush_result": False}, "FlushIncomplete"), ({"gap": True}, "UntranscribedAudioGap")):
            with self.subTest(settings=settings):
                result, backend = await self.run_trial(**settings)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["failure"], expected)
                self.assertTrue(backend.closed)
                self.assertNotIn("private-secret", json.dumps(result))
                self.assertNotIn("provider-url", json.dumps(result))

    async def test_cancellation_closes_owned_backend(self):
        instances, fed = [], asyncio.Event()
        async def wait_forever(delay):
            if fed.is_set():
                await asyncio.Event().wait()
            await asyncio.sleep(0)
        def factory(callbacks):
            backend = FakeBackend(callbacks)
            feed = backend.feed
            def signal(*args):
                feed(*args)
                fed.set()
            backend.feed = signal
            instances.append(backend)
            return backend
        clip = benchmark.Clip("speech", "Test phrase.", b"\x00\x10" * 1600, 3200)
        task = asyncio.create_task(benchmark.trial("whisper", clip, factory, sleep=wait_forever))
        await asyncio.wait_for(fed.wait(), 1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(instances[0].closed)
        self.assertTrue(instances[0].stopped.is_set())

    async def test_benchmark_fresh_instances_and_never_mutates_live_state_or_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory, "existing.log")
            sentinel.write_text("existing history")
            original_state, original_logs = bridge.state, bridge.LOG_DIR
            original_metrics = dict(bridge.state.metrics)
            logger = logging.getLogger("bridge")
            original_disabled = logger.disabled
            instances = []
            def factory(callbacks):
                backend = FakeBackend(callbacks)
                instances.append(backend)
                return backend
            clock = Clock()
            clips = [benchmark.Clip("speech", "Test phrase.", b"\x00\x10" * 1600, 3200)]
            with mock.patch.object(bridge, "configure") as configure, mock.patch.object(bridge, "MODEL_LABEL", "test-requested"):
                result = await benchmark.benchmark(clips, ["whisper", "deepgram"], environ={"DEEPGRAM_API_KEY": "private-secret"},
                                                   factory=factory, repeats=2, sleep=clock.sleep, monotonic=clock.now)
            self.assertEqual(result["status"], "passed")
            self.assertEqual(len(instances), 4)
            self.assertTrue(all(instance.closed for instance in instances))
            self.assertEqual([call.args[0]["OMI_STT_BACKEND"] for call in configure.call_args_list],
                             ["whisper", "whisper", "deepgram", "deepgram"])
            self.assertTrue(all(call.args[0]["OMI_HOTWORDS"] == "" for call in configure.call_args_list))
            self.assertEqual(sentinel.read_text(), "existing history")
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])
            self.assertIs(bridge.state, original_state)
            self.assertEqual(bridge.state.metrics, original_metrics)
            self.assertEqual(bridge.LOG_DIR, original_logs)
            self.assertEqual(logger.disabled, original_disabled)


class CliTests(unittest.TestCase):
    def test_default_cli_uses_whisper_and_report_redacts_nested_secrets(self):
        report = {"status": "passed", "trials": [], "summary": {"resolved_model": "private-secret"}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "result.json")
            with mock.patch.object(bridge, "require_python315"), \
                 mock.patch.object(benchmark, "load_env", return_value={"DEEPGRAM_API_KEY": "private-secret"}), \
                 mock.patch.object(benchmark, "synthetic_clips", return_value=[]), \
                 mock.patch.object(benchmark, "benchmark", new=mock.AsyncMock(return_value=report)) as run, \
                 redirect_stdout(io.StringIO()) as output:
                code = benchmark.main(["--report", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(run.call_args.args[1], ["whisper"])
            self.assertNotIn("private-secret", path.read_text())
            self.assertNotIn("private-secret", output.getvalue())
            self.assertEqual(json.loads(path.read_text())["summary"]["resolved_model"], "[redacted]")

    def test_synthesis_errors_never_echo_subprocess_stderr_or_tracebacks(self):
        with mock.patch.object(bridge, "require_python315"), \
             mock.patch.object(benchmark, "load_env", return_value={}), \
             mock.patch.object(benchmark, "synthetic_clips", side_effect=subprocess.CalledProcessError(
                 1, ["say"], stderr="private-secret")), \
             redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()) as error:
            code = benchmark.main([])
        self.assertEqual(code, 1)
        self.assertNotIn("private-secret", error.getvalue())
        self.assertNotIn("Traceback", error.getvalue())
