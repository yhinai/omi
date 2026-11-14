"""Explicit recording boundaries and signal guards; no live audio or signals."""
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import signal
import stat
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import wave

from common import RATE, BPS
from diagnostics import AudioCapture, MAX_CAPTURE_BYTES
from scripts import capture_wearable as cli


class Clock:
    def __init__(self):
        self.now = 100.0
        self.handles = []

    def __call__(self):
        return self.now

    def schedule(self, seconds, callback):
        handle = SimpleNamespace(due=self.now+seconds, callback=callback, cancelled=False)
        handle.cancel = lambda: setattr(handle, "cancelled", True)
        self.handles.append(handle)
        return handle

    def advance(self, seconds):
        self.now += seconds
        for handle in self.handles[:]:
            if handle.due <= self.now:
                self.handles.remove(handle)
                if not handle.cancelled:
                    handle.callback()


class CaptureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name) / "private"
        self.clock = Clock()
        self.capture = AudioCapture(self.directory, clock=self.clock, schedule=self.clock.schedule)
        self.addCleanup(self.capture.close)

    def read_wav(self):
        with wave.open(self.capture.snapshot()["path"], "rb") as audio:
            self.assertEqual((audio.getnchannels(), audio.getsampwidth(), audio.getframerate()), (1, 2, RATE))
            return audio.readframes(audio.getnframes())

    def test_construction_snapshot_and_feed_do_not_record_by_default(self):
        self.capture.feed(b"\x00\x10"*1600, 0, 1)
        snapshot = self.capture.snapshot()
        self.assertEqual(snapshot["state"], "idle")
        self.assertEqual(snapshot["samples"], 0)
        self.assertIsNone(snapshot["path"])
        self.assertFalse(self.directory.exists())
        self.assertFalse(self.clock.handles)

    def test_explicit_capture_writes_secure_valid_wav_and_releases_pcm(self):
        self.assertTrue(self.capture.start())
        raw = b"\x00\x10\x00\xf0"*800
        self.capture.feed(raw, 1600, 4)
        self.capture.finish()
        result = self.capture.snapshot()
        self.assertEqual(self.read_wav(), raw)
        self.assertEqual(result["samples"], 1600)
        self.assertEqual(result["bytes"], len(raw))
        self.assertEqual(result["audio_sec"], .1)
        self.assertEqual(result["state"], "completed")
        self.assertEqual(stat.S_IMODE(self.directory.stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(Path(result["path"]).stat().st_mode), 0o600)
        self.assertFalse(self.capture._pcm)
        self.assertNotIn(raw.decode("latin-1"), json.dumps(result))

    def test_audio_limit_caps_huge_packet_at_ten_seconds(self):
        self.capture.start()
        self.capture.feed(bytes(MAX_CAPTURE_BYTES+BPS), 0, 1)
        result = self.capture.snapshot()
        self.assertEqual(result["reason"], "audio_limit")
        self.assertEqual(result["samples"], RATE*10)
        self.assertEqual(len(self.read_wav()), MAX_CAPTURE_BYTES)
        self.assertEqual(Path(result["path"]).stat().st_size, MAX_CAPTURE_BYTES+44)

    def test_wall_deadline_stops_even_when_no_more_packets_arrive(self):
        self.capture.start()
        raw = b"\x00\x10"*1600
        self.capture.feed(raw, 0, 1)
        self.clock.advance(10)
        result = self.capture.snapshot()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["reason"], "wall_timeout")
        self.capture.feed(bytes(BPS), 1600, 1)
        self.assertEqual(self.read_wav(), raw)

    def test_late_packet_is_not_recorded_and_empty_capture_creates_no_file(self):
        self.capture.start()
        self.capture.feed(bytes(BPS), 0, 1, now=self.clock()+10)
        result = self.capture.snapshot()
        self.assertEqual(result["state"], "completed")
        self.assertEqual(result["samples"], 0)
        self.assertIsNone(result["path"])
        self.assertFalse(list(self.directory.iterdir()))

    def test_concurrent_start_is_rejected_and_shutdown_discards_unfinished_audio(self):
        self.capture.start()
        identity = self.capture.snapshot()["capture_id"]
        self.assertFalse(self.capture.start())
        self.assertEqual(self.capture.snapshot()["capture_id"], identity)
        self.capture.feed(bytes(BPS), 0, 1)
        self.capture.close()
        self.clock.advance(20)
        result = self.capture.snapshot()
        self.assertEqual(result["state"], "cancelled")
        self.assertIsNone(result["path"])
        self.assertFalse(self.capture._pcm)
        self.assertFalse(list(self.directory.iterdir()))

    def test_source_change_or_gap_does_not_join_unrelated_audio(self):
        for offset, session, reason in ((1600, 2, "source_changed"), (3200, 1, "source_gap")):
            with self.subTest(reason=reason):
                self.capture.start()
                raw = b"\x00\x10"*1600
                self.capture.feed(raw, 0, 1)
                self.capture.feed(bytes(BPS), offset, session)
                self.assertEqual(self.capture.snapshot()["reason"], reason)
                self.assertEqual(self.read_wav(), raw)

    def test_existing_file_is_never_overwritten(self):
        with mock.patch("diagnostics.uuid.uuid4", return_value=SimpleNamespace(hex="fixed")):
            self.capture.start()
            existing = self.directory / "omi-audio-fixed.wav"
            existing.write_bytes(b"existing private recording")
            self.capture.feed(bytes(BPS), 0, 1)
            self.capture.finish()
        self.assertEqual(existing.read_bytes(), b"existing private recording")
        self.assertEqual(self.capture.snapshot()["state"], "error")
        self.assertEqual(self.capture.snapshot()["error_type"], "FileExistsError")
        self.assertFalse(self.capture._pcm)

    def test_symlink_destination_is_rejected_without_changing_target(self):
        target = self.directory.parent / "target"
        target.mkdir(mode=0o755)
        target.chmod(0o755)
        self.directory.symlink_to(target, target_is_directory=True)
        self.assertFalse(self.capture.start())
        self.assertEqual(self.capture.snapshot()["state"], "error")
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o755)

    def test_write_failure_leaves_no_partial_file_or_sensitive_error_text(self):
        self.capture.start()
        self.capture.feed(bytes(BPS), 0, 1)
        with mock.patch("diagnostics.wave.open", side_effect=OSError("private sample details")):
            self.capture.finish()
        result = self.capture.snapshot()
        self.assertEqual(result["error_type"], "OSError")
        self.assertNotIn("private sample details", json.dumps(result))
        self.assertFalse(list(self.directory.iterdir()))
        self.assertFalse(self.capture._pcm)


def health(state="idle", capture_id=None, pid=1234, *, phone=True, packets=100, packet_age=0, **updates):
    return {"pid": pid, "direct_ws_connected": phone, "direct_ws_packets": packets,
        "last_packet_age_sec": packet_age, "audio_capture": {"supported": True, "protocol_version": 1,
        "trigger": "SIGUSR1", "state": state, "capture_id": capture_id, **updates}}


class CaptureClientTests(unittest.TestCase):
    def record(self, states, **overrides):
        self.signal = mock.Mock()
        self.identity = mock.Mock(return_value=(501, "synthetic birth", "synthetic bridge command"))
        self.announce = mock.Mock()
        self.clock = Clock()
        self.get_health = mock.Mock(side_effect=states)
        options = dict(get_health=self.get_health, process_identity=self.identity,
            send_signal=self.signal, clock=self.clock, sleep=self.clock.advance, announce=self.announce)
        options.update(overrides)
        return cli.record_capture(**options)

    def test_explicit_client_checks_identity_and_capability_before_one_signal(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "omi-audio-test.wav")
            result = self.record([health(), health(packets=101), health("recording", "new"),
                health("completed", "new", path=path, samples=1600, received_packets=1, reason="wall_timeout")])
        self.signal.assert_called_once_with(1234, signal.SIGUSR1)
        self.assertEqual(self.identity.call_count, 2)
        self.assertEqual(result["path"], path)
        self.assertEqual(result["samples"], 1600)
        self.assertIn("10 seconds", self.announce.call_args.args[0])

    def test_old_unsupported_bridge_is_never_signalled(self):
        for old in ({"pid": 1234}, health(protocol_version=0), health(protocol_version=True), health(trigger="unknown")):
            with self.subTest(old=old), self.assertRaises(cli.CaptureError):
                self.record([old])
            self.signal.assert_not_called()
            self.identity.assert_not_called()

    def test_active_capture_or_changed_pid_or_process_identity_is_never_signalled(self):
        for states in ([health("recording", "existing")], [health(), health(pid=5678)],
                       [health(), health("recording", "existing")]):
            with self.subTest(states=states), self.assertRaises(cli.CaptureError):
                self.record(states)
            self.signal.assert_not_called()
        changed = mock.Mock(side_effect=[(501, "old", "command"), (501, "new", "command")])
        with self.assertRaises(cli.CaptureError):
            self.record([health(), health(packets=101)], process_identity=changed)
        self.signal.assert_not_called()

    def test_client_timeout_is_bounded_and_never_retriggers_recording(self):
        count = 100
        def no_completion(**kwargs):
            nonlocal count
            count += 1
            return health("idle", packets=count)
        with self.assertRaisesRegex(cli.CaptureError, "15 seconds"):
            self.record([], get_health=no_completion)
        self.assertLessEqual(self.clock(), 115.001)
        self.signal.assert_called_once()

    def test_bridge_restart_during_capture_cannot_report_another_process_capture(self):
        with self.assertRaisesRegex(cli.CaptureError, "restarted"):
            self.record([health(), health(packets=101), health("completed", "new", pid=5678, samples=100)])
        self.signal.assert_called_once()

    def test_disconnected_stale_or_stationary_phone_cannot_start_recording(self):
        for fields in ({"phone": False}, {"packet_age": 2}, {}, {"packets": None}):
            with self.subTest(fields=fields):
                count = 100
                def not_ready(**kwargs):
                    nonlocal count
                    count += 1
                    options = {"packets": 100 if not fields else count, **fields}
                    return health(**options)
                with self.assertRaisesRegex(cli.CaptureError, "Phone audio did not advance"):
                    self.record([], get_health=not_ready)
                self.signal.assert_not_called()
                self.announce.assert_not_called()
                self.assertLessEqual(self.clock(), 103.001)

    def test_capture_waits_for_actual_packet_progress_then_rechecks_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "audio.wav")
            result = self.record([health(phone=False), health(phone=False), health(packets=101),
                health("completed", "new", samples=1600, path=path)])
        self.assertEqual(result["samples"], 1600)
        self.assertAlmostEqual(self.clock(), 100.1)
        self.identity.assert_has_calls([mock.call(1234), mock.call(1234)])
        self.signal.assert_called_once()

    def test_empty_completed_capture_is_a_clear_failure(self):
        with self.assertRaisesRegex(cli.CaptureError, "received no audio"):
            self.record([health(), health(packets=101), health("completed", "new", samples=0, path=None)])
        self.signal.assert_called_once()

    def test_no_record_flag_and_help_never_probe_or_signal(self):
        for args, expected in (([], 2), (["--help"], 0)):
            with self.subTest(args=args), mock.patch.object(cli, "record_capture") as record, \
                 redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as stopped:
                    cli.main(args)
                self.assertEqual(stopped.exception.code, expected)
                record.assert_not_called()

    def test_user_controls_recording_start_in_terminal(self):
        calls = []
        with mock.patch.object(cli.sys.stdin, "isatty", return_value=True), \
             mock.patch("builtins.input", side_effect=lambda prompt: calls.append("enter")), \
             mock.patch.object(cli.time, "sleep", side_effect=lambda seconds: calls.append(("countdown", seconds))), \
             mock.patch.object(cli, "record_capture", side_effect=lambda: calls.append("record") or {"samples": 160000}), \
             redirect_stdout(io.StringIO()):
            cli.main(["--record", "--wait-for-enter"])
        self.assertEqual(calls, ["enter", ("countdown", 3), "record"])

    def test_nonterminal_wait_cannot_start_recording(self):
        with mock.patch.object(cli.sys.stdin, "isatty", return_value=False), \
             mock.patch.object(cli, "record_capture") as record, redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                cli.main(["--record", "--wait-for-enter"])
            record.assert_not_called()

    def test_process_guard_checks_exact_command_owner_and_birth_time(self):
        expected = Path("/example/project with spaces/bridge.py")
        line = "501 Sat Nov 14 12:00:00 2026 /example/python3.15 " + str(expected)
        run = mock.Mock(return_value=SimpleNamespace(stdout=line, returncode=0))
        result = cli.inspect_process(1234, expected_script=expected, runner=run, uid=501)
        self.assertEqual(result, (501, "Sat Nov 14 12:00:00 2026", "/example/python3.15 " + str(expected)))
        for bad in (line.replace("501 ", "502 ", 1), line+" --check-config",
                    line.replace("python3.15", "curl"), line.replace("python3.15", "python3-malicious")):
            run.return_value.stdout = bad
            with self.subTest(bad=bad), self.assertRaises(cli.CaptureError):
                cli.inspect_process(1234, expected_script=expected, runner=run, uid=501)


if __name__ == "__main__":
    unittest.main()
