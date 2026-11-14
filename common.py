"""Shared PCM contract and validated local recognition policy (no model imports)."""
from dataclasses import dataclass
from collections import deque
import math
import struct

RATE, BPS = 16000, 32000
QUEUE_AUDIO_SECONDS = 10
QUEUE_MAX_BYTES = BPS * QUEUE_AUDIO_SECONDS
# Phone ingress sends 10 ms (320-byte) frames at 100 pkt/s; 2048 lets the
# 10-second byte budget govern normal 5-100 ms frames while bounding tiny floods.
QUEUE_MAX_PACKETS = 2048
REPLAY_MAX_BYTES = BPS * 45
PACKET_MAX_BYTES = 3200
SHUTDOWN_TIMEOUT = 20
WORKER_DIAGNOSTIC_BYTES = 8192


@dataclass
class Packet:
    pcm: bytes
    received_at: float
    received_mono: float
    sample_offset: int
    session_id: int
    audio_start_at: float | None = None

    def __post_init__(self):
        if self.audio_start_at is None:
            self.audio_start_at = self.received_at - len(self.pcm) / BPS


@dataclass(frozen=True, slots=True)
class _AudioQualityRecord:
    received_mono: float
    received_samples: int
    sampled_samples: int
    square_sum: int
    signed_sum: int
    peak: int
    clipped: int


class AudioQualityMeter:
    """Bounded PCM16LE observations, retaining aggregate numbers rather than audio.

    Call add with chronological monotonic packet receipt times, then snapshot
    with the current monotonic time. Measurements cover retained packets, not a
    continuous recording. Large packets are sampled uniformly across their
    length; their peaks and clipping rates are estimates over selected samples.
    """

    def __init__(self, window_seconds=5.0, max_records=512, max_samples_per_packet=RATE):
        if not math.isfinite(window_seconds) or window_seconds <= 0:
            raise ValueError("Audio quality window must be finite and positive")
        if not isinstance(max_records, int) or not 1 <= max_records <= 512:
            raise ValueError("Audio quality history must contain at most 512 records")
        if not isinstance(max_samples_per_packet, int) or not 1 <= max_samples_per_packet <= RATE:
            raise ValueError("Audio quality sampling must use at most one second per packet")
        self.window_seconds = window_seconds
        self.max_samples_per_packet = max_samples_per_packet
        self._records = deque(maxlen=max_records)
        self._last_evicted_at = None

    def _expire(self, now):
        while self._records and self._records[0].received_mono < now - self.window_seconds:
            self._records.popleft()

    def add(self, pcm, received_mono):
        if len(pcm) % 2:
            raise ValueError("Audio quality requires complete PCM16 samples")
        if not math.isfinite(received_mono):
            raise ValueError("Audio quality receipt time must be finite")
        self._expire(received_mono)
        count = len(pcm) // 2
        if not count:
            return
        if self._records and received_mono < self._records[-1].received_mono:
            raise ValueError("Audio quality packet times must be chronological")
        selected = min(count, self.max_samples_per_packet)
        if selected == count:
            values = (sample for sample, in struct.iter_unpack("<h", pcm))
        else:
            # Include both ends without copying or scanning the whole message.
            denominator = max(1, selected - 1)
            values = (struct.unpack_from("<h", pcm, index * (count - 1) // denominator * 2)[0]
                      for index in range(selected))
        square_sum = signed_sum = peak = clipped = 0
        for sample in values:
            square_sum += sample * sample
            signed_sum += sample
            peak = max(peak, abs(sample))
            clipped += sample in (-32768, 32767)
        if len(self._records) == self._records.maxlen:
            self._last_evicted_at = self._records[0].received_mono
        self._records.append(_AudioQualityRecord(received_mono, count, selected,
                            square_sum, signed_sum, peak, clipped))

    def snapshot(self, now):
        if not math.isfinite(now):
            raise ValueError("Audio quality snapshot time must be finite")
        self._expire(now)
        sampled = sum(record.sampled_samples for record in self._records)
        received = sum(record.received_samples for record in self._records)
        squares = sum(record.square_sum for record in self._records)
        peak = max((record.peak for record in self._records), default=0)
        return {
            "window_sec": self.window_seconds,
            "max_packet_records": self._records.maxlen,
            "max_samples_per_packet": self.max_samples_per_packet,
            "packets": len(self._records),
            "coverage_scope": "retained_packets",
            "coverage": "sampled" if sampled < received else "complete",
            "sampling": "uniform_across_packet_when_limited",
            "record_cap_limited": (self._last_evicted_at is not None
                                   and self._last_evicted_at >= now - self.window_seconds),
            "sampled_samples": sampled,
            "received_samples": received,
            "sampled_audio_sec": round(sampled / RATE, 6),
            "received_audio_sec": round(received / RATE, 6),
            "peak_dbfs": round(20 * math.log10(peak / 32768), 3) if peak else None,
            "rms_dbfs": round(10 * math.log10(squares / sampled / 32768**2), 3) if squares else None,
            "clipped_sample_pct": round(100 * sum(record.clipped for record in self._records) / sampled, 4) if sampled else None,
            "dc_offset_fs": round(sum(record.signed_sum for record in self._records) / sampled / 32768, 6) if sampled else None,
        }


@dataclass(frozen=True)
class WhisperPolicy:
    initial_seconds: float = .25
    update_seconds: float = .2
    pause_seconds: float = .5
    prefinal_quiet_seconds: float = .25
    # Continuous room sound (a TV, music) never yields half a second of silence, so
    # the cap is what decides how often a line can appear: 20 s reads as "paused".
    max_turn_seconds: float = 8
    # 0.1 opened a turn on almost any room noise; 57% of turns decoded to nothing.
    start_probability: float = .25
    continue_probability: float = .3
    confirm_probability: float = .35  # was .5: quiet or distant speech never reached it and was dropped
    confirm_frames: int = 2           # two 32 ms frames reject 1-frame transients while keeping short words

    def __post_init__(self):
        timings = (self.initial_seconds, self.update_seconds, self.pause_seconds, self.max_turn_seconds)
        if not all(math.isfinite(v) and v > 0 for v in timings):
            raise ValueError("Whisper timings must be finite and positive")
        if not math.isfinite(self.prefinal_quiet_seconds) or not 0 <= self.prefinal_quiet_seconds <= self.pause_seconds:
            raise ValueError("Whisper pre-final quiet window must be between zero and pause_seconds")
        if not isinstance(self.confirm_frames, int) or isinstance(self.confirm_frames, bool) or self.confirm_frames < 1:
            raise ValueError("Whisper confirm_frames must be a positive integer")
        if not self.initial_seconds <= self.max_turn_seconds <= 30:
            raise ValueError("Whisper maximum turn must fit its 30-second context")
        if not (0 < self.start_probability <= self.continue_probability <= self.confirm_probability < 1):
            raise ValueError("Whisper speech probabilities must be ordered between zero and one")


WHISPER_POLICY = WhisperPolicy()
