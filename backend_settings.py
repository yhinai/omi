"""Import-safe recognition contract and cloud configuration; no network or model imports."""
from dataclasses import dataclass, field
import os
import json
from pathlib import Path
from typing import Mapping, Protocol


class TranscriptionBackend(Protocol):
    """Source positions, bounded intake and immutable saves stay provider independent."""
    connected: bool
    total_input_samples: int
    session_id: int
    pending_bytes: int

    @property
    def pending_age(self) -> float | None: ...

    @property
    def speech_pending(self) -> bool: ...

    def feed(self, raw: bytes, received_at: float, received_mono: float) -> None: ...
    def new_session(self) -> None: ...
    def request_flush(self) -> str: ...
    async def flush_now(self, timeout: float = 5) -> bool: ...
    async def run(self) -> None: ...
    async def close(self) -> bool: ...


@dataclass(frozen=True)
class DeepgramConfig:
    api_key: str = field(repr=False)
    model: str = "nova-3"
    language: str = "en"
    endpointing_ms: int = 300
    diarize: bool = False
    pipeline: str = "direct"
    python: str = ""
    keyterms: tuple = ()

    @classmethod
    def from_env(cls, env: Mapping[str, str]):
        key = env.get("DEEPGRAM_API_KEY", "").strip()
        if not key or len(key) > 8192 or any(c.isspace() or not c.isprintable() for c in key):
            raise ValueError("A valid DEEPGRAM_API_KEY is required for Deepgram mode")
        model = env.get("OMI_DEEPGRAM_MODEL", "nova-3").strip().lower()
        if model != "nova-3":
            raise ValueError("This Deepgram adapter supports OMI_DEEPGRAM_MODEL=nova-3")
        language = env.get("OMI_LANGUAGE", "en").strip().lower()
        allowed = {part.strip().lower() for part in env.get("OMI_LANGUAGES", "en").split(",") if part.strip()}
        english = {"en", "en-us"}
        if language not in english or not allowed or not allowed <= english:
            raise ValueError("Deepgram mode currently requires English and an English-only language allowlist")
        try:
            endpointing = int(env.get("OMI_DEEPGRAM_ENDPOINTING_MS", "300"))
        except (ValueError, TypeError):
            raise ValueError("Deepgram endpointing must be an integer from 1 to 5000 milliseconds") from None
        if not 1 <= endpointing <= 5000:
            raise ValueError("Deepgram endpointing must be an integer from 1 to 5000 milliseconds")
        diarize = env.get("OMI_DEEPGRAM_DIARIZE", "false").strip().lower()
        if diarize not in {"true", "false"}:
            raise ValueError("OMI_DEEPGRAM_DIARIZE must be true or false")
        pipeline = env.get("OMI_DEEPGRAM_PIPELINE", "direct").strip().lower()
        if pipeline not in {"direct", "pipecat"}:
            raise ValueError("OMI_DEEPGRAM_PIPELINE must be direct or pipecat")
        python = os.path.expanduser(env.get("OMI_PIPECAT_PYTHON",
            str(Path(__file__).parent / ".venv-pipecat/bin/python")))
        try:
            terms=json.loads(env.get("OMI_DEEPGRAM_KEYTERMS", "[]"))
        except (TypeError, ValueError):raise ValueError("Keyterms must be a JSON array") from None
        if not isinstance(terms,list) or len(terms)>100 or any(not isinstance(t,str) or not t.strip() or len(t)>100 for t in terms) or sum(len(t) for t in terms)>500:
            raise ValueError("Use at most 100 nonempty keyterms totaling at most 500 characters")
        return cls(key, model, language, endpointing, diarize == "true", pipeline, python, tuple(terms))
