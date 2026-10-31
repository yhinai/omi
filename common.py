"""Shared PCM contract and validated local recognition policy (no model imports)."""
from dataclasses import dataclass
import math

RATE, BPS = 16000, 32000


@dataclass
class Packet:
    pcm: bytes
    received_at: float
    received_mono: float
    sample_offset: int
    session_id: int


@dataclass(frozen=True)
class WhisperPolicy:
    initial_seconds: float = .25
    update_seconds: float = .2
    pause_seconds: float = .5
    max_turn_seconds: float = 20
    start_probability: float = .1
    continue_probability: float = .3
    confirm_probability: float = .5

    def __post_init__(self):
        timings = (self.initial_seconds, self.update_seconds, self.pause_seconds, self.max_turn_seconds)
        if not all(math.isfinite(v) and v > 0 for v in timings):
            raise ValueError("Whisper timings must be finite and positive")
        if not self.initial_seconds <= self.max_turn_seconds <= 30:
            raise ValueError("Whisper maximum turn must fit its 30-second context")
        if not (0 < self.start_probability <= self.continue_probability <= self.confirm_probability < 1):
            raise ValueError("Whisper speech probabilities must be ordered between zero and one")


WHISPER_POLICY = WhisperPolicy()
