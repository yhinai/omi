"""Live-stage verification against fake endpoints and temporary synthetic journals only."""
import asyncio
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock
import sys
import tempfile

import aiohttp

spec = importlib.util.spec_from_file_location("omi_verify_live", Path(__file__).resolve().parents[1] / "scripts/verify_live.py")
verify_live = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = verify_live
spec.loader.exec_module(verify_live)

PRIVATE = "unrelated-private-conversation"
TOKEN = "private-caption-token"
PHRASE = "Blue lantern seven"


def final_event(**changes):
    return {"type": "transcript", "is_final": True, "text": PHRASE + ".", "event_id": "synthetic-test-event",
            "stream_id": "synthetic-stream", "session_id": 1, "utterance_id": "synthetic-turn",
            "timestamp": "2026-11-14 12:00:00", "sample_offset": 16000, "sample_end_offset": 32000,
            "recognition": {"provider": "deepgram", "requested_model": "nova-3", "language": "en-US", "resolved_model": "nova-3"},
            **changes}


def health(packets=5, **changes):
    return {"stt_backend": "deepgram", "status": "ok", "pid": 10, "backend_ready": True, "workers_alive": True,
            "stream_connected": True, "direct_ws_connected": True, "direct_ws_packets": packets,
            "last_error": PRIVATE, "direct_ws_remote": "private-address", **changes}


def initial_messages():
    return [{"type": "replay", "phase": "start", "count": 0}, {"type": "replay", "phase": "end"},
            {"type": "status", "stt_backend": "deepgram", "stream_id": "synthetic-stream"}]


class Content:
    def __init__(self, body):
        self.body, self.offset = body, 0

    async def read(self, count):
        chunk = self.body[self.offset:self.offset + count]
        self.offset += len(chunk)
        return chunk


class Response:
    def __init__(self, data, status=200, raw=False):
        self.status = status
        self.content = Content(data if raw else json.dumps(data).encode())

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass


class Socket:
    def __init__(self, messages):
        self.messages = list(messages)
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        self.closed = True

    async def receive(self):
        if not self.messages:
            return SimpleNamespace(type=aiohttp.WSMsgType.CLOSE, data=None)
        event = self.messages.pop(0)
        return SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=event if isinstance(event, str) else json.dumps(event))


class Session:
    def __init__(self, messages, *, snapshots=None, public_status=401):
        self.socket = Socket(messages)
        self.snapshots = list(snapshots or [health(), health(7)])
        self.public_status = public_status
        self.calls = []
        self.options = None

    def factory(self, **options):
        self.options = options
        return self

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    def get(self, url, **kwargs):
        self.calls.append(("get", url, kwargs))
        if url == verify_live.HEALTH_URL:
            return Response(self.snapshots.pop(0))
        if url == verify_live.PUBLIC_URL:
            return Response({}, self.public_status)
        raise AssertionError("Unexpected network destination")

    def ws_connect(self, url, **kwargs):
        self.calls.append(("ws", url, kwargs))
        if url != verify_live.CAPTION_URL:
            raise AssertionError("Unexpected caption destination")
        return self.socket


async def no_sleep(seconds):
    await asyncio.sleep(0)


class VerificationTests(unittest.IsolatedAsyncioTestCase):
    def fixture(self, directory, event=None):
        root = Path(directory)
        logs = root / "logs"
        logs.mkdir()
        event = event or final_event()
        unrelated = final_event(text=PRIVATE, event_id="unrelated")
        (logs / "omi_transcript_events.jsonl").write_text(json.dumps(unrelated) + "\n" + json.dumps(event) + "\n")
        (logs / "2026-11-14.md").write_text("# Synthetic transcript\n\n" + PRIVATE + "\n\n**[12:00:00]** " + event["text"] + "\n")
        return root

    async def test_phrase_checks_new_final_storage_provenance_and_viewer_without_private_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            messages = initial_messages() + [{"type": "partial", "text": PRIVATE}, final_event(text=PRIVATE, event_id="unrelated"), final_event()]
            session = Session(messages)
            ready = mock.Mock()
            report = await verify_live.verify(phrase=PHRASE, root=root, environ={"OMI_STREAM_TOKEN": TOKEN},
                                              session_factory=session.factory, sleep=no_sleep, on_ready=ready)
            ready.assert_called_once_with()
            self.assertEqual(report["status"], "partial")  # Actual terminal and public endpoint are unobserved.
            for name in ("controlPlane", "backendReady", "phoneConnected", "streamPacketProgress", "captionSubscription",
                         "captionStatus", "newEventsOnly", "providerConnected", "resolvedModel", "providerFinal", "provenance",
                         "journal", "markdown", "viewerConsumption"):
                self.assertEqual(report["stages"][name]["status"], "observed", name)
            self.assertEqual(report["stages"]["terminalRenderer"]["status"], "not_tested")
            self.assertEqual(report["stages"]["streamPacketProgress"]["packets"], 2)
            for private in (PRIVATE, TOKEN, PHRASE, "private-address"):
                self.assertNotIn(private, json.dumps(report))
            self.assertFalse(session.options["trust_env"])
            ws_call = next(call for call in session.calls if call[0] == "ws")
            self.assertEqual(ws_call[1], "ws://127.0.0.1:8787/listen?replay=0")
            self.assertEqual(ws_call[2]["headers"], {"Authorization": "Bearer " + TOKEN})
            self.assertEqual(ws_call[2]["max_msg_size"], verify_live.MAX_MESSAGE)
            self.assertTrue(session.socket.closed)
            self.assertTrue(all("127.0.0.1" in call[1] for call in session.calls))

    async def test_default_status_does_not_claim_asr_or_inspect_saved_history(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session(initial_messages(), snapshots=[health(), health(5, stream_connected=False)])
            with mock.patch.object(verify_live, "journal_observation", side_effect=AssertionError("Unexpected file read")), \
                 mock.patch.object(verify_live, "markdown_observation", side_effect=AssertionError("Unexpected file read")):
                report = await verify_live.verify(root=directory, environ={}, session_factory=session.factory, sleep=no_sleep)
            self.assertEqual(report["stages"]["backendReady"]["status"], "observed")
            for stage in ("providerFinal", "journal", "markdown", "viewerConsumption", "streamPacketProgress", "providerConnected", "resolvedModel"):
                self.assertEqual(report["stages"][stage]["status"], "not_tested")

    async def test_only_explicit_public_probe_uses_external_url_and_never_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            for status, expected in ((401, "observed"), (200, "failed"), (302, "failed")):
                with self.subTest(status=status):
                    session = Session(initial_messages(), public_status=status)
                    report = await verify_live.verify(root=directory, environ={"OMI_CAPTION_TOKEN": TOKEN}, public=True,
                                                      session_factory=session.factory, sleep=no_sleep)
                    self.assertEqual(report["stages"]["publicCaptionAuth"]["status"], expected)
                    public = next(call for call in session.calls if call[1] == verify_live.PUBLIC_URL)
                    self.assertEqual(public[2], {"headers": {}, "allow_redirects": False})

    async def test_old_replay_and_unmatched_text_cannot_satisfy_phrase(self):
        with tempfile.TemporaryDirectory() as directory:
            root = self.fixture(directory)
            for messages in ([{"type": "replay", "phase": "start", "count": 1}, final_event()],
                             [final_event(), *initial_messages(), final_event(text=PRIVATE, event_id="different")]):
                with self.subTest(messages=len(messages)):
                    report = await verify_live.verify(phrase=PHRASE, root=root, environ={},
                                                      session_factory=Session(messages).factory, sleep=no_sleep)
                    self.assertEqual(report["stages"]["providerFinal"]["status"], "failed")
                    self.assertEqual(report["stages"]["journal"]["status"], "not_tested")
                    self.assertNotIn(PRIVATE, json.dumps(report))

    async def test_bad_provider_provenance_and_mismatched_journal_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            saved = final_event(recognition={"provider": "whisper"})
            root = self.fixture(directory, saved)
            messages = [*initial_messages(), final_event()]
            report = await verify_live.verify(phrase=PHRASE, root=root, environ={},
                                              session_factory=Session(messages).factory, sleep=no_sleep)
            self.assertEqual(report["stages"]["journal"]["status"], "failed")
            self.assertEqual(report["stages"]["provenance"]["status"], "observed")
            messages = [*initial_messages(), saved]
            report = await verify_live.verify(phrase=PHRASE, root=root, environ={},
                                              session_factory=Session(messages).factory, sleep=no_sleep)
            self.assertEqual(report["stages"]["provenance"]["status"], "failed")

    async def test_message_count_and_size_are_bounded_and_errors_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            session = Session([*initial_messages(), {"type": "partial", "text": PRIVATE}] * 3)
            with mock.patch.object(verify_live, "MAX_MESSAGES", 4):
                report = await verify_live.verify(phrase=PHRASE, root=directory, environ={},
                                                  session_factory=session.factory, sleep=no_sleep)
            self.assertEqual(report["caption_messages_observed"], 4)
            self.assertEqual(report["stages"]["captionSubscription"]["status"], "failed")
            session = Session(["x" * 101])
            with mock.patch.object(verify_live, "MAX_MESSAGE", 100):
                report = await verify_live.verify(root=directory, environ={}, session_factory=session.factory, sleep=no_sleep)
            self.assertEqual(report["stages"]["captionSubscription"]["status"], "failed")
            self.assertNotIn(PRIVATE, json.dumps(report))

    async def test_health_body_bound_and_bad_configuration_never_network(self):
        session = SimpleNamespace(get=lambda *args, **kwargs: Response(b"x" * (verify_live.MAX_BODY + 1), raw=True))
        with self.assertRaises(ValueError):
            await verify_live.read_health(session)
        with tempfile.TemporaryDirectory() as directory:
            factory = mock.Mock(side_effect=AssertionError("No network expected"))
            for kwargs in ({"timeout": 61}, {"timeout": float("nan")}, {"phrase": "short"},
                           {"environ": {"OMI_CAPTION_TOKEN": "bad token"}}):
                arguments = {"root": directory, "environ": {}, "session_factory": factory, **kwargs}
                report = await verify_live.verify(**arguments)
                self.assertEqual(report["stages"]["configuration"]["status"], "failed")
            factory.assert_not_called()


class StorageAndCliTests(unittest.TestCase):
    def test_journal_decodes_only_matched_id_and_markdown_checks_only_its_day(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            event = final_event(text="Blue lantern 7.")
            path = root / "logs/omi_transcript_events.jsonl"
            path.write_text(json.dumps(final_event(event_id="unrelated", text=PRIVATE)) + "\n" + json.dumps(event) + "\n")
            (root / "logs/2026-11-14.md").write_text("Blue lantern seven.")
            loads = json.loads
            with mock.patch.object(verify_live.json, "loads", wraps=loads) as decode:
                status, _ = verify_live.journal_observation(root, event)
            self.assertEqual(status, "observed")
            self.assertEqual(decode.call_count, 1)
            self.assertNotIn(PRIVATE, decode.call_args.args[0].decode())
            self.assertEqual(verify_live.markdown_observation(root, event)[0], "observed")
            event["timestamp"] = "../../private"
            self.assertEqual(verify_live.markdown_observation(root, event)[0], "failed")

    def test_journal_bounds_report_unresolved_and_do_not_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "logs").mkdir()
            (root / "outside").write_text(json.dumps(final_event()) + "\n")
            (root / "logs/omi_transcript_events.jsonl").symlink_to(root / "outside")
            self.assertEqual(verify_live.journal_observation(root, final_event())[0], "not_tested")

    def test_json_cli_writes_only_safe_stage_report(self):
        report = {"status": "partial", "stages": {"terminalRenderer": {"status": "not_tested", "detail": "Unobserved"}}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory, "safe-report.json")
            with mock.patch.object(verify_live, "verify", new=mock.AsyncMock(return_value=report)) as verify, \
                 redirect_stdout(io.StringIO()) as output:
                code = verify_live.main(["--phrase", PHRASE, "--timeout", "60", "--json", "--report", str(path)])
            self.assertEqual(code, 0)
            self.assertEqual(json.loads(output.getvalue()), report)
            self.assertEqual(json.loads(path.read_text()), report)
            verify.assert_awaited_once_with(phrase=PHRASE, timeout=60, public=False, on_ready=mock.ANY)
