"""Check safe command dispatch and truthful user-facing connection stages."""
import unittest
from unittest import mock
import cli
from scripts import doctor


class CliTests(unittest.TestCase):
    def test_healthy_idle_service_is_distinct_from_audio_flow(self):
        health = dict(status="ok", stt_backend="deepgram", backend_ready=True, direct_ws_connected=False,
                      provider_connection_state="idle", last_packet_age_sec=400, pending_saves=0, output_errors=0)
        with mock.patch.object(doctor, "live_health", return_value=(200, health)):
            report = cli.status_report()
        self.assertEqual((report["service"], report["phone"], report["audio"], report["saving"]),
                         ("healthy", "disconnected", "idle", "ready"))

    def test_fresh_audio_and_save_failure_are_separate_states_and_secrets_are_omitted(self):
        health = dict(status="degraded", stt_backend="deepgram", backend_ready=False, direct_ws_connected=True,
                      provider_connection_state="retrying", last_packet_age_sec=0, pending_saves=1, output_errors=2,
                      last_error="private transcript and credentials", last_output_error="private save error")
        with mock.patch.object(doctor, "live_health", return_value=(503, health)):
            report = cli.status_report()
        self.assertEqual((report["service"], report["audio"], report["recognizer"], report["saving"]),
                         ("degraded", "flowing", "retrying", "error"))
        self.assertNotIn("private", str(report))

    def test_recovered_save_is_ready_even_with_historical_error_count(self):
        health = dict(status="ok", stt_backend="deepgram", backend_ready=True, pending_saves=0,
                      output_errors=3, last_output_error=None, provider_connection_state="reconnecting")
        with mock.patch.object(doctor, "live_health", return_value=(200, health)):
            report = cli.status_report()
        self.assertEqual(report["saving"], "ready")
        self.assertEqual(report["recognizer"], "reconnecting")

    def test_unavailable_or_invalid_health_never_appears_running(self):
        for value in [OSError(), (200, {"stt_backend": "unknown"}), (200, None)]:
            with self.subTest(value=value), mock.patch.object(doctor, "live_health") as read:
                if isinstance(value, Exception): read.side_effect = value
                else: read.return_value = value
                self.assertEqual(cli.status_report()["service"], "unavailable")

    def dispatch(self, argv):
        with mock.patch.object(cli.os, "chdir"), mock.patch.object(cli.os, "execv") as execute:
            cli.main(argv)
        return execute.call_args.args

    def test_only_explicit_backend_selection_reaches_switcher(self):
        executable, arguments = self.dispatch(["backend", "deepgram"])
        self.assertEqual(arguments, [cli.sys.executable, str(cli.ROOT / "scripts/set_backend.py"), "deepgram"])
        self.assertEqual(self.dispatch(["restart"])[1], [str(cli.ROOT / "run.sh"), "restart"])

    def test_forwarded_arguments_are_literal_and_check_is_read_only(self):
        phrase = "Blue lantern seven; $(touch never)"
        self.assertEqual(self.dispatch(["verify", "--phrase", phrase])[1][-2:], ["--phrase", phrase])
        self.assertEqual(self.dispatch(["check", "--json"])[1][-2:], ["--live", "--json"])
        self.assertEqual(self.dispatch(["dev", "smoke", "--backend", "deepgram"])[1][-2:], ["--backend", "deepgram"])

    def test_service_commands_reject_unexpected_flags(self):
        with mock.patch.object(cli.os, "execv") as execute, mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.main(["restart", "--set-whisper"])
        execute.assert_not_called()
