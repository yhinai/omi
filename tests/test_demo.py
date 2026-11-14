"""Synthetic demo transport and lifecycle checks; no model or real viewer required."""
import asyncio
import importlib.util
import io
from pathlib import Path
import signal
import sys
import tempfile
import wave
import unittest
from unittest import mock

import aiohttp
from aiohttp.test_utils import TestClient, TestServer

spec = importlib.util.spec_from_file_location("omi_demo", Path(__file__).resolve().parents[1] / "scripts/demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class DemoTransportTests(unittest.IsolatedAsyncioTestCase):
    async def test_authenticated_fixtures_are_labeled_and_replace_provisional_text(self):
        import tui
        async with TestClient(TestServer(demo.create_demo_app("demo-token", pace=0))) as client:
            response = await client.get("/listen")
            self.assertEqual(response.status, 401)
            response = await client.get("/listen", headers={"Authorization": "Bearer wrong"})
            self.assertEqual(response.status, 401)
            self.assertEqual((await client.get("/v1/listen")).status, 404)
            viewer = tui.Viewer()
            events = []
            async with client.ws_connect("/listen?replay=100", headers={"Authorization": "Bearer demo-token"}) as ws:
                for _ in range(11):
                    event = await ws.receive_json(timeout=1)
                    self.assertTrue(event["synthetic"])
                    events.append(event)
                    viewer.add(event)
            self.assertEqual([event["phase"] for event in events[:2]], ["start", "end"])
            self.assertEqual([event["text"] for event in events if event["type"] == "partial"],
                             ["The bridge is getting ready", "The bridge is ready to demonstrate"])
            text = "\n".join(caption.text for caption in viewer.captions)
            self.assertIn("SYNTHETIC DEMO", text)
            self.assertIn("no microphone, speech recognition, or live history", text)
            self.assertIn("The bridge is ready to demo.", text)
            self.assertNotIn("getting ready", text)
            self.assertNotIn("ready to demonstrate", text)
            self.assertIn("👨‍👩‍👧", text)
            self.assertIn("می\u200cروم", text)
            self.assertEqual(viewer.draft, "")
            self.assertEqual(viewer.toast, "Synthetic gap notice · no real audio was lost.")
            self.assertEqual(viewer.model, "SYNTHETIC DEMO — no ASR model")
            self.assertIsNone(viewer.p50)
            self.assertLess(len(viewer.captions), 10)

    async def test_resume_skips_previously_finalized_fixtures(self):
        async with TestClient(TestServer(demo.create_demo_app("demo-token", pace=0))) as client:
            async with client.ws_connect("/listen?after=demo-unicode",
                    headers={"Authorization": "Bearer demo-token"}) as ws:
                start = await ws.receive_json(timeout=1)
                self.assertFalse(start["reset"])
                self.assertEqual((await ws.receive_json(timeout=1))["phase"], "end")
                self.assertEqual((await ws.receive_json(timeout=1))["event_id"], "demo-gap")
                self.assertEqual((await ws.receive_json(timeout=1))["event_id"], "demo-complete")


class FakeViewer:
    def __init__(self, *, running=False):
        self.returncode = None if running else 0
        self.stopped = asyncio.Event()
        self.signals = []
        self.killed = False

    async def wait(self):
        if self.returncode is None:
            await self.stopped.wait()
        return self.returncode

    def send_signal(self, value):
        self.signals.append(value)
        self.returncode = 130
        self.stopped.set()

    def kill(self):
        self.killed = True
        self.returncode = -9
        self.stopped.set()


class DemoLifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def assert_server_closed(self, url):
        async with aiohttp.ClientSession() as client:
            with self.assertRaises(aiohttp.ClientConnectionError):
                await client.get(url)

    async def test_child_uses_private_environment_and_temporary_exports_then_closes_server(self):
        observed = {}
        async def spawn(*command, **kwargs):
            observed.update(command=command, **kwargs)
            url = command[command.index("--url") + 1]
            token = kwargs["env"]["OMI_CAPTION_TOKEN"]
            async with aiohttp.ClientSession() as client:
                async with client.ws_connect(url, headers={"Authorization": "Bearer " + token}) as ws:
                    self.assertEqual((await ws.receive_json(timeout=1))["phase"], "start")
            return FakeViewer()

        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(demo, "PROJECT_DIR", Path(directory)), \
             mock.patch.dict(demo.os.environ, {"OMI_STREAM_TOKEN": "live-secret", "OMI_TUI_MOUSE_LOG": directory + "/mouse.log"}), \
             mock.patch.object(demo.asyncio, "create_subprocess_exec", side_effect=spawn):
            live = Path(directory, "logs")
            live.mkdir()
            sentinel = live / "keep.txt"
            sentinel.write_text("existing history")
            self.assertEqual(await demo.run_demo(), 0)
            self.assertEqual(list(live.iterdir()), [sentinel])
            self.assertEqual(sentinel.read_text(), "existing history")
            self.assertFalse(Path(directory, "mouse.log").exists())
        command = observed["command"]
        token = observed["env"]["OMI_CAPTION_TOKEN"]
        self.assertGreaterEqual(len(token), 32)
        self.assertFalse(any(token in str(argument) for argument in command))
        self.assertNotIn("OMI_STREAM_TOKEN", observed["env"])
        self.assertNotIn("OMI_TUI_MOUSE_LOG", observed["env"])
        self.assertEqual(command[:3], (sys.executable, "-B", str(Path(directory, "tui.py"))))
        self.assertFalse(Path(observed["cwd"]).exists())
        self.assertEqual(command[command.index("--export-dir") + 1], observed["cwd"])
        await self.assert_server_closed(command[command.index("--url") + 1])

    async def test_interruption_stops_and_reaps_child_then_closes_server(self):
        viewer = FakeViewer(running=True)
        launched = asyncio.Event()
        observed = {}
        async def spawn(*command, **kwargs):
            observed["url"] = command[command.index("--url") + 1]
            observed["directory"] = kwargs["cwd"]
            launched.set()
            return viewer

        with mock.patch.object(demo.asyncio, "create_subprocess_exec", side_effect=spawn):
            task = asyncio.create_task(demo.run_demo())
            await asyncio.wait_for(launched.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(viewer.signals, [signal.SIGINT])
        self.assertIsNotNone(viewer.returncode)
        self.assertFalse(Path(observed["directory"]).exists())
        await self.assert_server_closed(observed["url"])

    async def test_failed_child_start_cleans_up_listener(self):
        observed = {}
        async def fail(*command, **kwargs):
            observed["url"] = command[command.index("--url") + 1]
            raise OSError("private failure details")

        with mock.patch.object(demo.asyncio, "create_subprocess_exec", side_effect=fail):
            with self.assertRaises(OSError):
                await demo.run_demo()
        await self.assert_server_closed(observed["url"])

    async def test_interruption_during_launch_still_acquires_and_reaps_child(self):
        viewer = FakeViewer(running=True)
        launching, release = asyncio.Event(), asyncio.Event()
        async def spawn(*command, **kwargs):
            launching.set()
            await release.wait()
            return viewer

        with mock.patch.object(demo.asyncio, "create_subprocess_exec", side_effect=spawn):
            task = asyncio.create_task(demo.run_demo())
            await asyncio.wait_for(launching.wait(), 1)
            task.cancel()
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertEqual(viewer.signals, [signal.SIGINT])
        self.assertIsNotNone(viewer.returncode)

    async def test_unresponsive_child_is_killed_and_reaped_on_interruption(self):
        class StubbornViewer(FakeViewer):
            def send_signal(self, value):
                self.signals.append(value)

        viewer = StubbornViewer(running=True)
        launched = asyncio.Event()
        async def spawn(*command, **kwargs):
            launched.set()
            return viewer

        with mock.patch.object(demo.asyncio, "create_subprocess_exec", side_effect=spawn):
            task = asyncio.create_task(demo.run_demo())
            await asyncio.wait_for(launched.wait(), 1)
            await asyncio.sleep(0)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(viewer.killed)
        self.assertEqual(viewer.returncode, -9)

    def test_nonterminal_error_is_clear_and_redacted(self):
        stderr = io.StringIO()
        with mock.patch.object(demo.sys, "argv", ["demo.py"]), \
             mock.patch.object(demo.sys.stdin, "isatty", return_value=False), \
             mock.patch.object(demo.sys, "stderr", stderr):
            with self.assertRaises(SystemExit) as raised:
                demo.main()
        self.assertEqual(raised.exception.code, 2)
        self.assertIn("real terminal", stderr.getvalue())


class RecordedDemoTests(unittest.TestCase):
    def test_recorded_demo_cli_validation_and_wav_loader(self):
        for bad_argv in (["--wav", "x.wav"], ["--headless"]):
            with self.subTest(argv=bad_argv), mock.patch.object(demo.sys, "stderr", io.StringIO()):
                with self.assertRaises(SystemExit) as ctx:
                    demo.main(bad_argv)
                self.assertEqual(ctx.exception.code, 2)

        bundled, label = demo.load_recorded_demo_pcm(None)
        self.assertEqual(label, "demo.wav")
        self.assertGreater(len(bundled), 16000 * 2)
        with tempfile.TemporaryDirectory() as directory:
            good_wav = Path(directory, "good.wav")
            with wave.open(str(good_wav), "wb") as w:
                w.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes(b"\x00\x00" * 1600)
            pcm, label = demo.load_recorded_demo_pcm(good_wav)
            self.assertEqual(label, "good.wav")
            self.assertEqual(len(pcm), 1600 * 2 + 16000 * 2)

            bad_wav = Path(directory, "stereo.wav")
            with wave.open(str(bad_wav), "wb") as w:
                w.setparams((2, 2, 16000, 0, "NONE", "not compressed"))
                w.writeframes(b"\x00\x00" * 1600)
            with self.assertRaises(ValueError):
                demo.load_recorded_demo_pcm(bad_wav)


if __name__ == "__main__":
    unittest.main()
