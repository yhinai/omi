"""Optional Pipecat subprocess transport; reuse Omi's tested Deepgram source/save contract."""
import asyncio
import base64
from contextlib import suppress
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import aiohttp
from deepgram_stt import DeepgramSTT

PIPECAT_VERSION = "1.16.0"
SDK_VERSION = "7.19.0"
MAX_MESSAGE = 1024 * 1024


class PipelineSocket:
    """Bounded, acknowledged private IPC with an owned Pipecat pipeline process."""
    def __init__(self, python, key, settings, process_factory=asyncio.create_subprocess_exec):
        self.python, self.key, self.settings = python, key, settings
        self.process_factory = process_factory
        self.process = self.pump = self.diagnostics = None
        self.events = asyncio.Queue(maxsize=128)
        self.waiters = {}
        self.sequence = 0
        self.failure = None
        self.frame_counts = {}
        self.provider_connect_sec = None

    async def __aenter__(self):
        launch = asyncio.create_task(self.process_factory(
            str(self.python), "-B", str(Path(__file__).with_name("pipecat_worker.py")),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=MAX_MESSAGE,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "DEEPGRAM_API_KEY": self.key,
                 "OMI_PIPECAT_SETTINGS": json.dumps(self.settings)}))
        try:
            self.process = await asyncio.shield(launch)
            self.diagnostics = asyncio.create_task(self._drain_diagnostics())
            ready = json.loads(await asyncio.wait_for(self.process.stdout.readline(), 30))
            required = {"kind": "ready", "protocol": 1, "pipecat": PIPECAT_VERSION, "deepgram_sdk": SDK_VERSION}
            if not isinstance(ready, dict) or any(ready.get(k) != v for k,v in required.items()):
                raise ConnectionError("Pipecat pipeline did not become ready")
            seconds = ready.get("provider_connect_sec")
            if isinstance(seconds, (int, float)) and not isinstance(seconds, bool) and math.isfinite(seconds) and seconds >= 0:
                self.provider_connect_sec = seconds
            self.pump = asyncio.create_task(self._receive())
            return self
        except BaseException:
            if self.process is None:
                self.process = await launch
            await self.__aexit__(None, None, None)
            raise

    async def _drain_diagnostics(self):
        # Third-party diagnostic text can contain recognized speech or request details.
        while await self.process.stderr.read(4096):
            pass

    async def _receive(self):
        try:
            while line := await self.process.stdout.readline():
                message = json.loads(line)
                if message.get("kind") == "ack":
                    future = self.waiters.get(message.get("id"))
                    if future is not None and not future.done():
                        future.set_result(None)
                elif message.get("kind") == "event" and isinstance(message.get("event"), dict):
                    if message["event"].get("type") == "Metadata":
                        counts = message["event"].get("omi_pipeline", {})
                        if isinstance(counts, dict):
                            self.frame_counts = {k:v for k,v in counts.items() if k in {"audio", "interim", "final"}
                                                 and isinstance(v, int) and not isinstance(v, bool) and v >= 0}
                    await self.events.put(SimpleNamespace(type=aiohttp.WSMsgType.TEXT,
                                                         data=json.dumps(message["event"])))
                else:
                    raise ConnectionError("Invalid Pipecat pipeline response")
            raise ConnectionError("Pipecat pipeline exited")
        except asyncio.CancelledError:
            raise
        except Exception:
            self.failure = ConnectionError("Pipecat pipeline transport failed")
            for future in self.waiters.values():
                if not future.done():future.set_exception(self.failure)
            await self.events.put(SimpleNamespace(type=aiohttp.WSMsgType.CLOSED, data=None))

    async def _send(self, message):
        if self.failure:
            raise self.failure
        self.sequence += 1
        identity = self.sequence
        future = self.waiters[identity] = asyncio.get_running_loop().create_future()
        try:
            self.process.stdin.write((json.dumps({**message, "id": identity})+"\n").encode())
            await self.process.stdin.drain()
            await asyncio.wait_for(future, 5)
        finally:
            self.waiters.pop(identity, None)
            if not future.done():
                future.cancel()
            elif not future.cancelled():
                future.exception()  # Consume a simultaneous pipe-failure notification.

    async def send_bytes(self, raw):
        await self._send({"op": "audio", "pcm": base64.b64encode(raw).decode("ascii")})

    async def send_json(self, event):
        if event.get("type") not in {"CloseStream", "KeepAlive"}:
            raise ValueError("Unsupported Pipecat control")
        await self._send({"op": event["type"]})

    async def receive(self):
        return await self.events.get()

    async def __aexit__(self, *args):
        if self.process is not None and self.process.returncode is None:
            try:
                self.process.stdin.write(b'{"op":"stop"}\n')
                await self.process.stdin.drain()
                await asyncio.wait_for(self.process.wait(), 5)
            except (OSError, TimeoutError, ConnectionError):
                if self.process.returncode is None:
                    with suppress(ProcessLookupError):self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), 3)
                except TimeoutError:
                    with suppress(ProcessLookupError):self.process.kill()
                    await self.process.wait()
        for task in (self.pump, self.diagnostics):
            if task is not None:task.cancel()
        await asyncio.gather(*(t for t in (self.pump, self.diagnostics) if t is not None), return_exceptions=True)


class _Session:
    def __init__(self, python, key, settings, socket_factory):
        self.arguments = python, key, settings
        self.socket_factory = socket_factory
        self.socket = None

    async def __aenter__(self):return self
    async def __aexit__(self, *args):pass
    def ws_connect(self, *args, **kwargs):
        self.socket = self.socket_factory(*self.arguments)
        return self.socket


class PipecatDeepgramSTT(DeepgramSTT):
    pipeline = "pipecat"

    def __init__(self, *, python, api_key, on_final, socket_factory=PipelineSocket, **kwargs):
        def save(text, **fields):
            fields["recognition"] = {**fields["recognition"], "orchestration": "pipecat",
                                     "pipecat_version": PIPECAT_VERSION, "deepgram_sdk_version": SDK_VERSION}
            return on_final(text, **fields)
        settings = {k: kwargs[k] for k in ("model", "language", "endpointing_ms", "diarize", "keyterms") if k in kwargs}
        self._pipeline_session = _Session(python, api_key, settings, socket_factory)
        super().__init__(api_key=api_key, on_final=save,
            session_factory=lambda **unused: self._pipeline_session, **kwargs)

    @property
    def pipeline_frames(self):
        return getattr(self._pipeline_session.socket, "frame_counts", {})

    def _latency(self, name, seconds):
        if name == "provider_connect_sec":
            super()._latency("pipeline_start_sec", seconds)
            actual = getattr(self._pipeline_session.socket, "provider_connect_sec", None)
            if actual is not None:
                super()._latency("provider_connect_sec", actual)
        else:
            super()._latency(name, seconds)
