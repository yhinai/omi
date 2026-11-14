"""Backend switch transactions in temporary projects; no live service or credentials."""
import fcntl
import importlib.util
import io
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

spec = importlib.util.spec_from_file_location("omi_set_backend", Path(__file__).resolve().parents[1] / "scripts/set_backend.py")
switcher = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = switcher
spec.loader.exec_module(switcher)


class SwitchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.path = self.root / ".env"
        self.original = b"# private fixture\r\nDEEPGRAM_API_KEY='fake-secret#only' # keep\r\nOMI_STT_BACKEND=whisper # keep mode comment\r\n"
        self.path.write_bytes(self.original)
        self.path.chmod(0o600)
        self.service = "whisper"
        self.phone = False
        self.provider = True
        self.health_status = "ok"
        self.now = 0
        self.probe = mock.patch.object(switcher, "probe").start()
        self.addCleanup(mock.patch.stopall)
        self.live = mock.patch.object(switcher, "live_backend", side_effect=lambda root: self.service).start()
        self.health = mock.patch.object(switcher, "health_snapshot", side_effect=self.snapshot).start()
        mock.patch.object(switcher, "time", SimpleNamespace(monotonic=lambda: self.now, sleep=self.advance)).start()
        self.restart = mock.patch.object(switcher, "restart", side_effect=self.start).start()

    def advance(self, seconds):
        self.now += seconds

    def snapshot(self, root, **kwargs):
        return {"stt_backend": self.service, "status": self.health_status,
                "direct_ws_connected": self.phone, "stream_connected": self.provider,
                "provider_connection_state": "connected" if self.provider else "idle"} if self.service else None

    def start(self, root, env, descriptor):
        self.service = switcher.configured_backend(self.path.read_bytes())
        self.provider, self.health_status = True, "ok"

    def switch(self, backend, env=None):
        return switcher.switch_backend(backend, self.root, {} if env is None else env)

    def test_success_preserves_secret_comments_crlf_and_mode(self):
        self.assertIn("Switched to deepgram", self.switch("deepgram"))
        self.assertEqual(self.path.read_bytes(), self.original.replace(b"BACKEND=whisper", b"BACKEND=deepgram"))
        self.assertEqual(stat.S_IMODE(self.path.stat().st_mode), 0o600)
        self.assertEqual(self.service, "deepgram")
        self.assertEqual(self.probe.call_args_list, [mock.call(self.root, "deepgram", {}), mock.call(self.root, "whisper", {})])
        self.assertFalse(list(self.root.glob(".env.switch-*")))

    def test_only_last_duplicate_assignment_changes_with_its_quote_style(self):
        data = b"OMI_STT_BACKEND=old-value\nSECRET=untouched\n export OMI_STT_BACKEND = 'Whisper'  # last wins\n"
        self.path.write_bytes(data)
        self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), data.replace(b"'Whisper'", b"'deepgram'"))

    def test_configuration_failure_does_not_edit_or_restart(self):
        self.probe.side_effect = switcher.SwitchError("configuration failed")
        with self.assertRaises(switcher.SwitchError):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.restart.assert_not_called()

    def test_unavailable_whisper_recovery_is_detected_before_cloud_switch(self):
        def check(root, backend, env):
            if backend == "whisper":
                raise switcher.SwitchError("whisper assets unavailable")
        self.probe.side_effect = check
        with self.assertRaisesRegex(switcher.SwitchError, "whisper assets unavailable"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.restart.assert_not_called()

    def test_failed_restart_restores_original_bytes_and_healthy_service(self):
        def fail_once(root, env, descriptor):
            if self.restart.call_count == 1:
                self.service = None
                raise switcher.SwitchError("failed")
            self.start(root, env, descriptor)
        self.restart.side_effect = fail_once
        with self.assertRaisesRegex(switcher.SwitchError, "whisper was restored and is healthy"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original)
        self.assertEqual(self.service, "whisper")
        self.assertEqual(self.restart.call_count, 2)

    def test_rollback_preserves_other_environment_updates_during_restart(self):
        def fail_once(root, env, descriptor):
            if self.restart.call_count == 1:
                self.path.write_bytes(self.path.read_bytes().replace(b"fake-secret#only", b"updated-secret"))
                self.service = None
                raise switcher.SwitchError("failed")
            self.start(root, env, descriptor)
        self.restart.side_effect = fail_once
        with self.assertRaisesRegex(switcher.SwitchError, "restored and is healthy"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original.replace(b"fake-secret#only", b"updated-secret"))

    def test_failed_rollback_never_claims_success(self):
        self.restart.side_effect = switcher.SwitchError("private failure")
        self.service = None
        with self.assertRaisesRegex(switcher.SwitchError, "automatic recovery was not confirmed"):
            self.switch("deepgram")

    def test_preflight_concurrent_edit_is_preserved_without_restart(self):
        def edit(root, backend, env):
            if backend == "deepgram":
                self.path.write_bytes(self.original + b"OTHER=new-value\n")
        self.probe.side_effect = edit
        with self.assertRaisesRegex(switcher.SwitchError, "changed concurrently"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original + b"OTHER=new-value\n")
        self.restart.assert_not_called()

    def test_concurrent_backend_change_is_never_overwritten_by_rollback(self):
        changed = self.original.replace(b"BACKEND=whisper", b"BACKEND='whisper'") + b"OTHER=external\n"
        def change(root, env, descriptor):
            self.path.write_bytes(changed)
            raise switcher.SwitchError("failed")
        self.restart.side_effect = change
        with self.assertRaisesRegex(switcher.SwitchError, "automatic recovery was not confirmed"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), changed)
        self.restart.assert_called_once()

    def test_conflicting_process_override_refuses_and_matching_override_is_removed(self):
        with self.assertRaisesRegex(switcher.SwitchError, "conflicting"):
            self.switch("deepgram", {"OMI_STT_BACKEND": "whisper"})
        self.assertEqual(self.path.read_bytes(), self.original)
        self.switch("deepgram", {"OMI_STT_BACKEND": "deepgram", "UNRELATED": "preserved"})
        self.assertNotIn("OMI_STT_BACKEND", self.restart.call_args.args[1])
        self.assertEqual(self.restart.call_args.args[1]["UNRELATED"], "preserved")

    def test_noop_requires_matching_live_health(self):
        self.assertIn("already configured and healthy", self.switch("whisper"))
        self.restart.assert_not_called()
        self.service = None
        self.assertIn("Switched", self.switch("whisper"))
        self.restart.assert_called_once()

    def test_symlinked_environment_and_contended_launcher_are_rejected(self):
        outside = self.root / "outside"
        outside.write_bytes(self.original)
        self.path.unlink()
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(switcher.SwitchError, "symlink"):
            self.switch("deepgram")
        self.assertEqual(outside.read_bytes(), self.original)
        self.restart.assert_not_called()
        self.path.unlink()
        self.path.write_bytes(self.original)
        with switcher.launcher_lock(self.root):
            with self.assertRaisesRegex(switcher.SwitchError, "Another launcher"):
                self.switch("deepgram")

    def test_default_whisper_assignment_is_removed_on_exact_rollback(self):
        self.original = b"DEEPGRAM_API_KEY=fake-secret"
        self.path.write_bytes(self.original)
        self.restart.side_effect = switcher.SwitchError("failed before stopping")
        with self.assertRaisesRegex(switcher.SwitchError, "restored and is healthy"):
            self.switch("deepgram")
        self.assertEqual(self.path.read_bytes(), self.original)

    def test_idle_cloud_activation_is_explicitly_unverified_until_audio(self):
        def idle(root, env, descriptor):
            self.service, self.provider = "deepgram", False
        self.restart.side_effect = idle
        message = self.switch("deepgram")
        self.assertIn("idle intake is ready", message)
        self.assertIn("Provider connection unverified until audio arrives", message)
        self.assertNotIn("provider connection are confirmed", message)
        self.assertGreaterEqual(self.now, 1)
        self.assertLess(self.now, 10)

    def test_active_phone_with_persistent_provider_failure_rolls_back(self):
        self.phone = True
        def fail_provider(root, env, descriptor):
            if self.restart.call_count == 1:
                self.service, self.provider, self.health_status = "deepgram", False, "degraded"
            else:
                self.start(root, env, descriptor)
        self.restart.side_effect = fail_provider
        with self.assertRaisesRegex(switcher.SwitchError, "whisper was restored and is healthy"):
            self.switch("deepgram")
        self.assertGreaterEqual(self.now, 10)
        self.assertEqual(self.service, "whisper")
        self.assertEqual(self.path.read_bytes(), self.original)


class SettlementTests(unittest.TestCase):
    def setUp(self):
        self.now = 0
        self.clock = SimpleNamespace(monotonic=lambda: self.now, sleep=self.advance)
        self.patch = mock.patch.object(switcher, "time", self.clock)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def advance(self, seconds):
        self.now += seconds

    def health(self, *, phone=False, connected=False, status="ok"):
        return {"stt_backend": "deepgram", "status": status,
                "direct_ws_connected": phone, "stream_connected": connected,
                "provider_connection_state": "connected" if connected else "idle"}

    def test_phone_arriving_during_idle_window_requires_provider_handshake(self):
        def health(root, **kwargs):
            return self.health(phone=self.now >= .5, connected=self.now >= 2)
        with mock.patch.object(switcher, "health_snapshot", side_effect=health):
            self.assertTrue(switcher.settle_backend(Path("/fixture"), "deepgram"))
        self.assertGreaterEqual(self.now, 2.25)

    def test_previously_active_phone_cannot_fall_back_to_idle_success(self):
        with mock.patch.object(switcher, "health_snapshot", return_value=self.health()):
            with self.assertRaises(switcher.SwitchError):
                switcher.settle_backend(Path("/fixture"), "deepgram", require_provider=True)
        self.assertEqual(self.now, 10)

    def test_short_degradation_can_recover_but_single_connected_sample_is_insufficient(self):
        def health(root, **kwargs):
            if self.now < .5:
                return self.health(phone=True, status="degraded")
            return self.health(phone=True, connected=True)
        with mock.patch.object(switcher, "health_snapshot", side_effect=health):
            self.assertTrue(switcher.settle_backend(Path("/fixture"), "deepgram"))
        self.assertEqual(self.now, .75)

    def test_degraded_http_response_is_read_without_treating_it_as_healthy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "logs/omi_bridge.pid").write_text("17")
            body = {"pid": 17, "stt_backend": "deepgram", "status": "degraded", "direct_ws_connected": True}
            def fail(*args, **kwargs):
                raise switcher.urllib.error.HTTPError("http://127.0.0.1/health", 503, "unavailable", {},
                                                      io.BytesIO(json.dumps(body).encode()))
            opener = mock.Mock()
            opener.open.side_effect = fail
            with mock.patch.object(switcher.urllib.request, "build_opener", return_value=opener):
                self.assertEqual(switcher.health_snapshot(root), body)
                self.assertIsNone(switcher.live_backend(root))


class ProcessContractTests(unittest.TestCase):
    def test_probe_and_launcher_hide_output_and_pass_lock_not_backend_override(self):
        root = Path("/fixture")
        with mock.patch.object(switcher.subprocess, "run", return_value=mock.Mock(returncode=0)) as run:
            switcher.probe(root, "deepgram", {"DEEPGRAM_API_KEY": "fake-secret"})
            self.assertEqual(run.call_args.kwargs["env"]["OMI_STT_BACKEND"], "deepgram")
            self.assertEqual(run.call_args.kwargs["stdout"], subprocess.DEVNULL)
            switcher.restart(root, {"OMI_STT_BACKEND": "deepgram"}, 17)
            self.assertEqual(run.call_args.args[0], ["/fixture/run.sh", "restart"])
            self.assertEqual(run.call_args.kwargs["pass_fds"], (17,))
            self.assertEqual(run.call_args.kwargs["env"]["OMI_LAUNCHER_LOCK_FD"], "17")
            self.assertNotIn("OMI_STT_BACKEND", run.call_args.kwargs["env"])

    def test_inherited_lock_is_validated_against_the_actual_launcher_file(self):
        source = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
        helper = source[source.index("# The switcher may hand off"):source.index("process_birth()")]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            lock = root / "logs/omi_bridge.launcher.lock"
            with lock.open("w") as held, (root / "other").open("w") as other:
                fcntl.flock(held.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                for descriptor, valid in ((held.fileno(), True), (other.fileno(), False)):
                    result = subprocess.run(["/bin/bash", "-c", helper + '\necho acquired'],
                        env={"DIR": str(root), "CLI_PYTHON": sys.executable, "OMI_LAUNCHER_LOCK_FD": str(descriptor)},
                        pass_fds=(descriptor,), capture_output=True, text=True)
                    self.assertEqual(result.returncode == 0, valid, result.stderr)
                    self.assertEqual("acquired" in result.stdout, valid)
                # Ordinary launcher commands still contend with a switcher's lock.
                result = subprocess.run(["/bin/bash", "-c", helper + '\necho acquired'],
                    env={"DIR": str(root), "CLI_PYTHON": sys.executable}, capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("acquired", result.stdout)

    def test_inherited_descriptor_cannot_inject_shell_commands(self):
        source = (Path(__file__).resolve().parents[1] / "run.sh").read_text()
        helper = source[source.index("# The switcher may hand off"):source.index("process_birth()")]
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(["/bin/bash", "-c", helper],
                env={"DIR": directory, "CLI_PYTHON": sys.executable, "OMI_LAUNCHER_LOCK_FD": "3; echo injected"},
                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("injected", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
