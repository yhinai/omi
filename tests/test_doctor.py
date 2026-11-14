"""Read-only preflight checks, including missing dependencies and secret redaction."""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
import urllib.error

spec = importlib.util.spec_from_file_location("omi_doctor", Path(__file__).resolve().parents[1] / "scripts/doctor.py")
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


class DoctorTests(unittest.TestCase):
    def fixture(self, directory):
        root = Path(directory)
        for name in (".venv/bin/python3.15", ".venv-whisper/bin/python", "run.sh", "tui.sh"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/bash\n")
            path.chmod(0o700)
        (root / "requirements-whisper.txt").write_text("mlx==0.32.4\nnumpy==2.5.4\n")
        model = root / "models"
        model.mkdir()
        for name in ("config.json", "weights.safetensors", "silero_vad.onnx"):
            (model / name).write_text("test")
        return root, {"OMI_STREAM_TOKEN": "private-fixture-token", "OMI_WHISPER_MODEL": str(model),
                      "OMI_WHISPER_VAD": str(model / "silero_vad.onnx")}

    def fake_probe(self, python, packages):
        return {"implementation": "cpython", "version": [3, 15 if python.name == "python3.15" else 14, 2],
                "releaselevel": "final", "gil_disabled": False, "platform": "darwin", "machine": "arm64",
                "packages": {name: pinned or "3.14.5" for name, pinned in packages.items()} if isinstance(packages, dict)
                            else {name: "3.14.5" for name in packages}}

    def run_inspect(self, root, env, **kwargs):
        with mock.patch.object(doctor, "probe", side_effect=self.fake_probe), \
             mock.patch.object(doctor.shutil, "which", return_value="/usr/sbin/lsof"):
            return doctor.inspect(root, environ=env, **kwargs)

    def check(self, report, name):
        return next(c for c in report["checks"] if c["check"] == name)

    def cloud_fixture(self, directory):
        root = Path(directory)
        for name in (".venv/bin/python3.15", "run.sh", "tui.sh"):
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("#!/bin/bash\n")
            path.chmod(0o700)
        return root, {"OMI_STT_BACKEND": "deepgram", "DEEPGRAM_API_KEY": "private-cloud-fixture-key",
                      "OMI_STREAM_TOKEN": "private-fixture-token"}

    def test_complete_preflight_is_read_only_and_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            (root / ".env").write_text('OMI_STREAM_TOKEN="file-secret"\nOMI_IDLE_FLUSH_SEC=4 # comment\n')
            before = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            report = self.run_inspect(root, env)
            after = {path.relative_to(root): path.read_bytes() for path in root.rglob("*") if path.is_file()}
            self.assertEqual(report["status"], "passed")
            self.assertEqual(before, after)
            self.assertNotIn(env["OMI_STREAM_TOKEN"], json.dumps(report))
            self.assertNotIn("file-secret", json.dumps(report))
            self.assertTrue(all(c["status"] == "pass" for c in report["checks"]))

    def test_dotenv_is_not_executed_and_process_values_win(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sentinel = root / "should-not-exist"
            (root / ".env").write_text(f"export OMI_STREAM_TOKEN='$(touch {sentinel})'\nOMI_IDLE_FLUSH_SEC=6\n")
            values = doctor.environment(root, {"OMI_IDLE_FLUSH_SEC": "4"})
            self.assertEqual(values["OMI_IDLE_FLUSH_SEC"], "4")
            self.assertTrue(values["OMI_STREAM_TOKEN"].startswith("$(touch "))
            self.assertFalse(sentinel.exists())

    def test_missing_dependency_and_wrong_runtime_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            def broken(python, packages):
                info = self.fake_probe(python, packages)
                if python.name == "python3.15":
                    info["packages"]["aiohttp"] = None
                else:
                    info["machine"] = "x86_64"
                return info
            with mock.patch.object(doctor, "probe", side_effect=broken), \
                 mock.patch.object(doctor.shutil, "which", return_value="lsof"):
                report = doctor.inspect(root, environ=env)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(self.check(report, "bridge_package_aiohttp")["status"], "fail")
            self.assertEqual(self.check(report, "worker_runtime")["status"], "fail")

    def test_missing_assets_bad_timings_and_anonymous_tokens_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            (root / "models/weights.safetensors").unlink()
            env.update(OMI_STREAM_TOKEN="", OMI_IDLE_FLUSH_SEC="nan")
            report = self.run_inspect(root, env)
            for name in ("model_weights", "audio_timings"):
                self.assertEqual(self.check(report, name)["status"], "fail")
            for name in ("stream_token", "caption_token"):
                self.assertEqual(self.check(report, name)["status"], "warn")

    def test_caption_limit_accepts_boundaries_and_rejects_invalid_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            for value in ("1", "16", "1024", "0", "1025", "3.5", "nan", ""):
                with self.subTest(value=value):
                    env["OMI_MAX_CAPTION_CLIENTS"] = value
                    report = self.run_inspect(root, env)
                    expected = "pass" if value in ("1", "16", "1024") else "fail"
                    self.assertEqual(self.check(report, "caption_client_limit")["status"], expected)

    def test_recognition_language_must_be_publishable_including_dotenv_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            for language, allowlist, expected in (("en", "en", "pass"), (" AR ", "en, ar", "pass"),
                                                  ("auto", "en,ar", "pass"), ("", "en", "pass"),
                                                  ("ar", "en", "fail"), ("auto", " , ", "fail")):
                with self.subTest(language=language, allowlist=allowlist):
                    (root / ".env").write_text(f"OMI_LANGUAGE='{language}'\nOMI_LANGUAGES='{allowlist}'\n")
                    report = self.run_inspect(root, env)
                    self.assertEqual(self.check(report, "recognition_languages")["status"], expected)
            env.update(OMI_LANGUAGE="en", OMI_LANGUAGES="en")
            report = self.run_inspect(root, env)
            self.assertEqual(self.check(report, "recognition_languages")["status"], "pass")

    def test_cloud_mode_needs_no_local_worker_or_assets_and_never_uses_key(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.cloud_fixture(directory)
            with mock.patch.object(doctor, "probe", side_effect=self.fake_probe) as probe, \
                 mock.patch.object(doctor.shutil, "which", return_value="lsof"), \
                 mock.patch.object(doctor, "live_health") as health:
                report = doctor.inspect(root, environ=env)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["backend"], "deepgram")
            health.assert_not_called()
            probe.assert_called_once_with(root / ".venv/bin/python3.15", {"aiohttp": None})
            names = {check["check"] for check in report["checks"]}
            self.assertFalse(any(name.startswith("worker_") for name in names))
            self.assertTrue(names.isdisjoint({"model_config", "model_weights", "speech_detector"}))
            self.assertNotIn(env["DEEPGRAM_API_KEY"], json.dumps(report))
            self.assertIn("Audio leaves this Mac", self.check(report, "deepgram_data_flow")["detail"])
            self.assertIn("mip_opt_out=true", self.check(report, "deepgram_data_flow")["detail"])
            self.assertIn("redact_usage=true", self.check(report, "deepgram_data_flow")["detail"])

    def test_cloud_key_and_configuration_fail_without_echoing_values(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.cloud_fixture(directory)
            cases = [("DEEPGRAM_API_KEY", ""), ("OMI_DEEPGRAM_MODEL", "unsupported-private-model"),
                     ("OMI_DEEPGRAM_ENDPOINTING_MS", "0"), ("OMI_DEEPGRAM_ENDPOINTING_MS", "5001"),
                     ("OMI_DEEPGRAM_ENDPOINTING_MS", "3.5"), ("OMI_DEEPGRAM_DIARIZE", "sometimes"),
                     ("OMI_LANGUAGE", "auto"), ("OMI_LANGUAGE", "ar"), ("OMI_LANGUAGES", "en,ar"),
                     ("OMI_LANGUAGES", "")]
            for key, value in cases:
                with self.subTest(key=key, value=value):
                    report = self.run_inspect(root, {**env, key: value})
                    self.assertEqual(report["status"], "failed")
                    self.assertNotIn(env["DEEPGRAM_API_KEY"], json.dumps(report))
                    self.assertNotIn("unsupported-private-model", json.dumps(report))
            for endpointing, diarize, language in (("1", "TRUE", "EN"), ("5000", "False", "en-US")):
                with self.subTest(endpointing=endpointing):
                    report = self.run_inspect(root, {**env, "OMI_DEEPGRAM_ENDPOINTING_MS": endpointing,
                                                   "OMI_DEEPGRAM_DIARIZE": diarize, "OMI_LANGUAGE": language})
                    self.assertEqual(report["status"], "passed")

    def test_unknown_backend_fails_and_live_backend_must_match_selected_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.cloud_fixture(directory)
            report = self.run_inspect(root, {**env, "OMI_STT_BACKEND": env["DEEPGRAM_API_KEY"]})
            self.assertEqual(self.check(report, "stt_backend")["status"], "fail")
            self.assertNotIn(env["DEEPGRAM_API_KEY"], json.dumps(report))
            for backend, expected in (("whisper", "fail"), ("deepgram", "pass")):
                with self.subTest(backend=backend), mock.patch.object(doctor, "live_health", return_value=(
                    200, {"status": "ok", "stt_backend": backend, "direct_ws_connected": True})):
                    report = self.run_inspect(root, env, live=True)
                    self.assertEqual(self.check(report, "live_health")["status"], expected)

    def test_failure_output_never_includes_probe_stderr_or_malformed_env(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            error = subprocess.CalledProcessError(1, ["python"], output="sensitive transcript", stderr=env["OMI_STREAM_TOKEN"])
            with mock.patch.object(doctor, "probe", side_effect=error):
                report = doctor.inspect(root, environ=env)
            self.assertNotIn("sensitive transcript", json.dumps(report))
            self.assertNotIn(env["OMI_STREAM_TOKEN"], json.dumps(report))
            (root / ".env").write_text('OMI_STREAM_TOKEN="unclosed-secret')
            report = doctor.inspect(root, environ=env)
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("unclosed-secret", json.dumps(report))

    def test_live_health_is_opt_in_and_never_echoes_health_body(self):
        with tempfile.TemporaryDirectory() as directory:
            root, env = self.fixture(directory)
            with mock.patch.object(doctor, "live_health") as health:
                self.run_inspect(root, env)
                health.assert_not_called()
                health.return_value = (503, {"status": "degraded", "last_error": "private recognized words"})
                report = self.run_inspect(root, env, live=True)
                self.assertEqual(self.check(report, "live_health")["status"], "fail")
                self.assertNotIn("private recognized words", json.dumps(report))
                health.return_value = (200, {"status": "ok", "stt_backend": "whisper", "direct_ws_connected": False})
                report = self.run_inspect(root, env, live=True)
                self.assertEqual(self.check(report, "live_health")["status"], "pass")
                self.assertEqual(self.check(report, "live_phone")["status"], "warn")

    def test_live_health_disables_proxies_and_limits_body_size(self):
        opener = mock.Mock()
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.code = 200
        response.read.return_value = b"x" * 65537
        opener.open.return_value = response
        with mock.patch.object(doctor.urllib.request, "ProxyHandler") as proxy, \
             mock.patch.object(doctor.urllib.request, "build_opener", return_value=opener):
            with self.assertRaises(ValueError):
                doctor.live_health()
            proxy.assert_called_once_with({})
            opener.open.assert_called_once_with("http://127.0.0.1:8788/health", timeout=3)
            response.read.assert_called_once_with(65537)

    def test_probe_reads_metadata_without_importing_packages(self):
        result = doctor.probe(sys.executable, ["omi-doctor-missing-dependency"])
        self.assertIsNone(result["packages"]["omi-doctor-missing-dependency"])
        self.assertEqual(result["implementation"], "cpython")

    def test_json_cli_failure_is_parseable_and_nonzero(self):
        report = {"status": "failed", "checks": [{"check": "bridge_runtime", "status": "fail", "detail": "Missing"}]}
        with mock.patch.object(doctor, "inspect", return_value=report) as inspect, \
             redirect_stdout(io.StringIO()) as output:
            code = doctor.main(["--json", "--live"])
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(output.getvalue()), report)
        inspect.assert_called_once_with(live=True)


class UnknownConfigurationTests(unittest.TestCase):
    def test_doctor_warns_only_when_dotenv_has_unknown_omi_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in (".venv/bin/python3.15", ".venv-whisper/bin/python", "run.sh", "tui.sh"):
                p = root / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text("#!/bin/bash\n")
                p.chmod(0o700)
            (root / "requirements-whisper.txt").write_text("mlx==0.32.4\n")
            model = root / "models"
            model.mkdir()
            for name in ("config.json", "weights.safetensors", "silero_vad.onnx"):
                (model / name).write_text("ok")
            (root / ".env").write_text("OMI_STREAM_TOKEN=secret-val\nOMI_LIVE_VAD_MODE=legacy-secret\n")

            fake_probe = lambda py, pkgs: {
                "implementation": "cpython",
                "version": [3, 15 if py.name == "python3.15" else 14, 2],
                "releaselevel": "final",
                "gil_disabled": False,
                "platform": "darwin",
                "machine": "arm64",
                "packages": {k: v or "3.14.5" for k, v in pkgs.items()},
            }
            with mock.patch.object(doctor, "probe", side_effect=fake_probe), \
                 mock.patch.object(doctor.shutil, "which", return_value="/usr/sbin/lsof"):
                report = doctor.inspect(root, environ={
                    "OMI_WHISPER_MODEL": str(model),
                    "OMI_WHISPER_VAD": str(model / "silero_vad.onnx"),
                })
            self.assertEqual(report["status"], "passed")
            warn_check = next(c for c in report["checks"] if c["check"] == "unused_dotenv_keys")
            self.assertEqual(warn_check["status"], "warn")
            self.assertIn("OMI_LIVE_VAD_MODE", warn_check["detail"])
            self.assertNotIn("legacy-secret", json.dumps(report))


    def test_selected_pipecat_checks_only_its_own_optional_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ('.venv/bin/python3.15','.venv-pipecat/bin/python','run.sh','tui.sh'):
                p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#!/bin/bash\n');p.chmod(0o700)
            env={'OMI_STT_BACKEND':'deepgram','DEEPGRAM_API_KEY':'synthetic',
                 'OMI_DEEPGRAM_PIPELINE':'pipecat','OMI_PIPECAT_PYTHON':str(root/'.venv-pipecat/bin/python')}
            def probe(python,packages):
                return {'implementation':'cpython','version':[3,15 if python.name=='python3.15' else 14,2],
                        'releaselevel':'final','gil_disabled':False,'platform':'darwin','machine':'arm64',
                        'packages':{name:version or '3.14.5' for name,version in packages.items()}}
            with mock.patch.object(doctor,'probe',side_effect=probe),mock.patch.object(doctor.shutil,'which',return_value='lsof'):
                report=doctor.inspect(root,environ=env)
            self.assertEqual(report['status'],'passed')
            names={r['check'] for r in report['checks']}
            self.assertIn('pipecat_package_pipecat-ai',names)
            self.assertFalse(any(n.startswith('worker_') for n in names))
            with mock.patch.object(doctor,'probe',side_effect=probe),mock.patch.object(doctor.shutil,'which',return_value='lsof'), \
                 mock.patch.object(doctor,'live_health',return_value=(200,{'status':'ok','stt_backend':'deepgram'})):
                live=doctor.inspect(root,environ=env,live=True)
            self.assertEqual(next(c['status'] for c in live['checks'] if c['check']=='live_health'),'fail')
