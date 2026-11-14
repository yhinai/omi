"""Private Pipecat/Deepgram worker. Stdout is a bounded JSON IPC protocol, never a log."""
import asyncio
import base64
from dataclasses import dataclass
import importlib.metadata
import json
import os
import stat
import sys
import time


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


async def serve(reader, settings, output=emit, client=None):
    # Pipecat and the provider SDK can log transcript text; keep all sinks disabled.
    from loguru import logger
    logger.remove()
    from deepgram.core.events import EventType
    from deepgram.listen.v1.types import ListenV1CloseStream, ListenV1KeepAlive
    from pipecat.frames.frames import DataFrame, EndFrame, InputAudioRawFrame, TranscriptionFrame, InterimTranscriptionFrame
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.worker import PipelineParams, PipelineWorker
    from pipecat.processors.frame_processor import FrameProcessor
    from pipecat.services.deepgram.stt import DeepgramSTTService
    from pipecat.workers.runner import WorkerRunner

    failure = asyncio.Event()
    started = asyncio.Event()
    sent = {}
    frame_counts = {"audio": 0, "interim": 0, "final": 0}
    connection_seconds = None

    @dataclass
    class ProviderFrame(DataFrame):
        payload: dict

    @dataclass
    class CommandFrame(DataFrame):
        operation: str
        identity: int

    class Service(DeepgramSTTService):
        closing = False

        def _build_connect_kwargs(self):
            options = super()._build_connect_kwargs()
            # SDK v7 omits this provider option from its typed signature.
            no_delay = options.pop("no_delay", None)
            if no_delay is not None:
                options["request_options"] = {"additional_query_parameters": {"no_delay": no_delay}}
            return options

        async def _connection_handler(self):
            nonlocal connection_seconds
            # One SDK connection per Omi epoch. Parent owns replay/retry; hidden
            # SDK reconnects would reset provider timestamps under the same clock.
            try:
                connecting_at = time.monotonic()
                async with self._client.listen.v1.connect(**self._build_connect_kwargs()) as connection:
                    connection_seconds = max(0, time.monotonic() - connecting_at)
                    self._connection = connection
                    self._connection_settled.set()
                    connection.on(EventType.MESSAGE, self._on_message)
                    connection.on(EventType.ERROR, self._on_error)
                    await connection.start_listening()
            except asyncio.CancelledError:
                raise
            except Exception:
                failure.set()
            finally:
                self._connection = None
                self._connection_settled.set()
                if not self.closing:failure.set()

        async def _on_error(self, error):
            failure.set()

        async def _on_message(self, message):
            try:
                await super()._on_message(message)
                payload = message.model_dump(mode="json", exclude_none=True)
                if payload.get("type") in {"Results", "Metadata", "SpeechStarted", "Error"}:
                    # Same downstream data queue as transcription frames: metadata
                    # cannot overtake a final already produced by this service.
                    await self.push_frame(ProviderFrame(payload))
            except Exception:
                failure.set()

        async def process_frame(self, frame, direction):
            if isinstance(frame, CommandFrame):
                await FrameProcessor.process_frame(self, frame, direction)
                if self._connection is None:
                    failure.set()
                    return
                try:
                    if frame.operation == "CloseStream":
                        self.closing = True
                        await self._connection.send_close_stream(ListenV1CloseStream(type="CloseStream"))
                    else:
                        await self._connection.send_keep_alive(ListenV1KeepAlive(type="KeepAlive"))
                    sent[frame.identity].set()
                except Exception:
                    failure.set()
                return
            await super().process_frame(frame, direction)
            if isinstance(frame, InputAudioRawFrame) and hasattr(frame, "omi_identity"):
                if self._connection is None:
                    failure.set()
                else:
                    frame_counts["audio"] += 1
                    sent[frame.omi_identity].set()

    class Sink(FrameProcessor):
        async def process_frame(self, frame, direction):
            await super().process_frame(frame, direction)
            if isinstance(frame, ProviderFrame):
                if frame.payload.get("type") == "Metadata":
                    frame.payload["omi_pipeline"] = dict(frame_counts)
                output({"kind": "event", "event": frame.payload})
            elif isinstance(frame, InterimTranscriptionFrame):
                frame_counts["interim"] += 1
            elif isinstance(frame, TranscriptionFrame):
                frame_counts["final"] += 1
            await self.push_frame(frame, direction)

    service = Service(api_key=os.environ.get("DEEPGRAM_API_KEY", ""), sample_rate=16000,
        channels=1, encoding="linear16", mip_opt_out=True, audio_passthrough=False,
        settings=DeepgramSTTService.Settings(model=settings.get("model", "nova-3"),
            language=settings.get("language", "en"), interim_results=True,
            endpointing=settings.get("endpointing_ms", 300), diarize=settings.get("diarize", False),
            punctuate=True, smart_format=True, keyterm=settings.get("keyterms") or None, extra={"no_delay": True, "redact_usage": True, "vad_events": True}))
    if client is not None:service._client = client
    worker = PipelineWorker(Pipeline([service, Sink()]),
        params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000,
                              enable_metrics=False, enable_usage_metrics=False),
        idle_timeout_secs=None, enable_turn_tracking=False, enable_rtvi=False)
    runner = WorkerRunner(handle_sigint=False)

    @worker.event_handler("on_pipeline_started")
    async def on_started(worker, frame):
        if service._connection is not None:started.set()
        else:failure.set()

    @worker.event_handler("on_pipeline_error")
    async def on_error(worker, frame):
        failure.set()

    await runner.add_workers(worker)
    running = asyncio.create_task(runner.run())
    broken = asyncio.create_task(failure.wait())
    async def wait_or_fail(awaitable, timeout):
        waiting = asyncio.ensure_future(awaitable)
        try:
            done, _ = await asyncio.wait((waiting, broken, running), timeout=timeout,
                                          return_when=asyncio.FIRST_COMPLETED)
            if waiting not in done or failure.is_set():
                raise ConnectionError("Pipecat pipeline failed or timed out")
            return waiting.result()
        finally:
            if not waiting.done():waiting.cancel()
            await asyncio.gather(waiting, return_exceptions=True)
    try:
        await wait_or_fail(started.wait(), 25)
        output({"kind": "ready", "protocol": 1, "pipecat": importlib.metadata.version("pipecat-ai"),
                "deepgram_sdk": importlib.metadata.version("deepgram-sdk"), "provider_connect_sec": connection_seconds})
        while True:
            line = await wait_or_fail(reader.readline(), 120)
            if not line:break
            request = json.loads(line)
            if request.get("op") == "stop":break
            identity = request.get("id")
            if not isinstance(identity, int) or isinstance(identity, bool):raise ValueError("Invalid sequence")
            sent[identity] = asyncio.Event()
            try:
                if request.get("op") == "audio":
                    raw = base64.b64decode(request.get("pcm", ""), validate=True)
                    if not raw or len(raw)>3200 or len(raw)%2:raise ValueError("Invalid PCM")
                    frame = InputAudioRawFrame(raw, 16000, 1)
                    frame.omi_identity = identity
                elif request.get("op") in {"CloseStream", "KeepAlive"}:
                    frame = CommandFrame(request["op"], identity)
                else:raise ValueError("Invalid command")
                await worker.queue_frame(frame)
                await wait_or_fail(sent[identity].wait(), 5)
                output({"kind": "ack", "id": identity})
            finally:
                sent.pop(identity, None)
    finally:
        service.closing = True
        broken.cancel()
        if not running.done():
            await worker.queue_frame(EndFrame())
            try:await asyncio.wait_for(asyncio.shield(running), 5)
            except TimeoutError:
                await worker.cancel()
                running.cancel()
        await asyncio.gather(running, broken, return_exceptions=True)
    return frame_counts


def main():
    if sys.version_info[:2] != (3, 14):
        raise SystemExit("Use the optional Python 3.14 Pipecat environment")
    if not stat.S_ISFIFO(os.fstat(sys.stdin.fileno()).st_mode):
        raise SystemExit("Pipecat worker requires a private bridge pipe")
    async def run():
        reader = asyncio.StreamReader(limit=65536)
        await asyncio.get_running_loop().connect_read_pipe(lambda: asyncio.StreamReaderProtocol(reader), sys.stdin)
        await serve(reader, json.loads(os.environ.get("OMI_PIPECAT_SETTINGS", "{}")))
    try:asyncio.run(run())
    except Exception:
        emit({"kind": "failed"})
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
