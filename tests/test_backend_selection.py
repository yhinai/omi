"""Backend selection never silently moves an existing recording to the cloud."""
import json
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import bridge
from test_bridge import live_state


class BackendSelectionTests(unittest.TestCase):
    def setUp(self):
        names = ("STT_BACKEND", "MODEL_LABEL", "DEEPGRAM_CONFIG", "WHISPER_ENV", "WHISPER_PYTHON",
                 "WHISPER_MODEL", "WHISPER_VAD", "HOST", "STREAM_TOKEN", "CAPTION_TOKEN",
                 "MAX_CAPTION_CLIENTS", "MAX_AUDIO_AGE_SEC", "IDLE_FLUSH_SEC")
        patcher = mock.patch.multiple(bridge, **{name: getattr(bridge, name) for name in names})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_a_key_alone_does_not_change_the_default_backend(self):
        with mock.patch.object(bridge.os.path, "isfile", return_value=True), \
             mock.patch.object(bridge.os, "access", return_value=True):
            bridge.configure({"DEEPGRAM_API_KEY": "test-key"})
        self.assertEqual(bridge.STT_BACKEND, "whisper")
        self.assertIsNone(bridge.DEEPGRAM_CONFIG)

    def test_cloud_mode_skips_all_local_worker_and_model_file_checks(self):
        with mock.patch.object(bridge.os.path, "isfile", side_effect=AssertionError("local assets accessed")), \
             mock.patch.object(bridge.os, "access", side_effect=AssertionError("local executable accessed")):
            bridge.configure({"OMI_STT_BACKEND": "deepgram", "DEEPGRAM_API_KEY": "test-key"})
        self.assertEqual(bridge.STT_BACKEND, "deepgram")
        self.assertEqual(bridge.MODEL_LABEL, "nova-3")

    def test_factory_passes_explicit_settings_and_preserves_callback_dict(self):
        bridge.configure({"OMI_STT_BACKEND": "deepgram", "DEEPGRAM_API_KEY": "test-key"})
        factory = mock.Mock()
        callbacks = {name: mock.Mock() for name in
                     ("on_partial", "on_final", "on_gap", "on_latency", "on_connection")}
        with mock.patch.dict(sys.modules, {"deepgram_stt": SimpleNamespace(DeepgramSTT=factory)}):
            self.assertIs(bridge.create_transcription_backend(callbacks), factory.return_value)
        self.assertEqual(factory.call_args.kwargs["model"], "nova-3")
        self.assertEqual(factory.call_args.kwargs["language"], "en")
        self.assertEqual(factory.call_args.kwargs["endpointing_ms"], 300)
        self.assertNotIn("max_age", callbacks)

    def test_unknown_backend_is_rejected_without_echoing_configuration(self):
        with self.assertRaisesRegex(ValueError, "whisper or deepgram") as caught:
            bridge.configure({"OMI_STT_BACKEND": "secret-value"})
        self.assertNotIn("secret-value", str(caught.exception))

    def test_idle_cloud_readiness_is_distinct_from_provider_connectivity(self):
        runtime = live_state()
        runtime.live_stt.ready = True
        runtime.live_stt.connection_state = "idle"
        alive = mock.Mock()
        alive.done.return_value = False
        runtime.background_tasks["test"] = alive
        with mock.patch.object(bridge, "state", runtime), mock.patch.object(bridge, "STT_BACKEND", "deepgram"):
            health = bridge.build_health_dict()
        self.assertEqual(health["status"], "ok")
        self.assertTrue(health["backend_ready"])
        self.assertFalse(health["stream_connected"])
        self.assertEqual(health["provider_connection_state"], "idle")

    def test_cloud_diagnostics_preserve_truthful_connection_and_save_stages(self):
        import tui
        runtime = live_state()
        runtime.live_stt.ready = False
        runtime.live_stt.connection_state = "reconnecting"
        runtime.live_stt.resolved_model = "general-nova-3"
        with mock.patch.object(bridge, "state", runtime), mock.patch.object(bridge, "STT_BACKEND", "deepgram"), \
             mock.patch.object(bridge, "workers_alive", return_value=True):
            bridge.record_latency("provider_connect_sec", .25)
            bridge.record_latency("final_to_persist_sec", .01)
            status = bridge.caption_status()
            self.assertEqual(bridge.build_health_dict()["provider_resolved_model"], "general-nova-3")
        viewer = tui.Viewer()
        viewer.add(status)
        self.assertEqual((viewer.backend, viewer.provider_state, viewer.resolved_model),
                         ("deepgram", "reconnecting", "general-nova-3"))
        self.assertEqual((viewer.connect_p50, viewer.persist_p50), (.25, .01))
        self.assertIn("degraded", viewer.toast)
        viewer.add({"type": "status", "backend_ready": True, "provider_connection_state": "idle",
                    "latency_provider_connect_p50_sec": None, "latency_final_to_persist_p50_sec": float("inf")})
        self.assertFalse(viewer.degraded_notified)
        self.assertIsNone(viewer.connect_p50)
        self.assertIsNone(viewer.persist_p50)

    def test_provider_provenance_survives_journal_normalization(self):
        runtime = live_state()
        metadata = {"provider": "deepgram", "connection_epoch": "epoch", "resolved_model": "model-version"}
        with mock.patch.object(bridge, "state", runtime), \
             mock.patch.object(bridge, "STT_BACKEND", "deepgram"), \
             mock.patch.object(bridge, "MODEL_LABEL", "nova-3"), \
             mock.patch.object(bridge, "event_logger") as journal:
            self.assertTrue(bridge.record_live_final("Synthetic test.", utterance_id="test", recognition=metadata))
        event = json.loads(journal.info.call_args.args[0])
        self.assertEqual(event["recognition"], metadata)
        self.assertEqual(event["stt_backend"], "deepgram")
        self.assertEqual(event["model"], "nova-3")

    def test_capture_capability_requires_installed_signal_handler(self):
        runtime = live_state()
        with mock.patch.object(bridge, "state", runtime):
            self.assertFalse(bridge.build_health_dict()["audio_capture"]["supported"])
            runtime.capture_signal_ready = True
            self.assertTrue(bridge.build_health_dict()["audio_capture"]["supported"])
            self.assertEqual(runtime.audio_capture.snapshot()["state"], "idle")

    def test_capture_hook_receives_original_source_positions_without_starting_it(self):
        runtime = live_state()
        runtime.live_stt.total_input_samples = 32000
        with mock.patch.object(bridge, "state", runtime), \
             mock.patch.object(runtime.audio_capture, "feed") as capture:
            bridge.ingest_raw_pcm(b"\x01\x00" * 160)
        self.assertEqual(capture.call_args.kwargs["sample_offset"], 32000)
        self.assertEqual(capture.call_args.kwargs["session_id"], runtime.live_stt.session_id)
        self.assertEqual(runtime.audio_capture.snapshot()["state"], "idle")

    def test_pipecat_is_opt_in_and_can_be_selected_without_loading_framework_in_bridge(self):
        with mock.patch.object(bridge.os.path, 'isfile', return_value=True), \
             mock.patch.object(bridge.os, 'access', return_value=True):
            bridge.configure({'OMI_STT_BACKEND':'deepgram','DEEPGRAM_API_KEY':'synthetic',
                              'OMI_DEEPGRAM_PIPELINE':'pipecat','OMI_PIPECAT_PYTHON':'/optional/python'})
        callbacks={key:lambda *args,**kwargs:None for key in
                   ['on_partial','on_final','on_gap','on_latency','on_connection']}
        adapter=bridge.create_transcription_backend(callbacks)
        self.assertEqual(adapter.pipeline,'pipecat')
        self.assertNotIn('pipecat.services.deepgram.stt',sys.modules)
        runtime=live_state()
        runtime.live_stt=adapter
        with mock.patch.object(bridge,'state',runtime):
            self.assertEqual(bridge.caption_status()['stt_pipeline'],'pipecat')
            self.assertEqual(bridge.build_health_dict()['stt_pipeline'],'pipecat')
        import tui
        viewer=tui.Viewer()
        viewer.add({'type':'status','stt_pipeline':'pipecat'})
        self.assertEqual(viewer.orchestration,'pipecat')
