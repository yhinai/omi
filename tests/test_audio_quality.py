"""Bounded signal measurements without retaining or classifying private audio."""
import json
import math
import struct
import unittest

from common import AudioQualityMeter, RATE


class AudioQualityTests(unittest.TestCase):
    def test_empty_and_silence_have_json_finite_dbfs(self):
        meter = AudioQualityMeter()
        meter.add(b"", 1)
        empty = meter.snapshot(1)
        self.assertEqual(empty["packets"], 0)
        self.assertIsNone(empty["peak_dbfs"])
        self.assertIsNone(empty["rms_dbfs"])
        self.assertIsNone(empty["dc_offset_fs"])
        meter.add(bytes(RATE * 2), 2)
        silence = meter.snapshot(2)
        self.assertIsNone(silence["peak_dbfs"])
        self.assertIsNone(silence["rms_dbfs"])
        self.assertEqual(silence["clipped_sample_pct"], 0)
        self.assertEqual(silence["dc_offset_fs"], 0)
        self.assertEqual(silence["sampled_audio_sec"], 1)
        json.dumps(empty, allow_nan=False)
        json.dumps(silence, allow_nan=False)

    def test_signed_little_endian_levels_and_dc_offset(self):
        meter = AudioQualityMeter()
        meter.add(b"\x00\x40\x00\xc0", 1)  # +16384, -16384
        result = meter.snapshot(1)
        self.assertAlmostEqual(result["peak_dbfs"], 20 * math.log10(.5), places=3)
        self.assertEqual(result["rms_dbfs"], result["peak_dbfs"])
        self.assertEqual(result["dc_offset_fs"], 0)
        self.assertEqual(result["clipped_sample_pct"], 0)
        positive = AudioQualityMeter()
        positive.add(b"\x00\x40" * 4, 1)
        self.assertEqual(positive.snapshot(1)["dc_offset_fs"], .5)

    def test_both_pcm_rails_count_as_clipped_and_aggregation_is_sample_weighted(self):
        meter = AudioQualityMeter()
        meter.add(struct.pack("<hh", -32768, 32767), 1)
        meter.add(bytes(12), 2)
        result = meter.snapshot(2)
        self.assertEqual(result["peak_dbfs"], 0)
        self.assertEqual(result["clipped_sample_pct"], 25)
        self.assertAlmostEqual(result["rms_dbfs"],
                               10 * math.log10((32768**2 + 32767**2) / 8 / 32768**2), places=3)
        self.assertAlmostEqual(result["dc_offset_fs"], -1 / 8 / 32768, places=6)

    def test_window_expires_without_new_audio(self):
        meter = AudioQualityMeter()
        meter.add(b"\x00\x40", 10)
        self.assertEqual(meter.snapshot(15)["sampled_samples"], 1)
        result = meter.snapshot(15.001)
        self.assertEqual(result["sampled_samples"], 0)
        self.assertIsNone(result["rms_dbfs"])

    def test_record_cap_is_reported_and_old_samples_are_removed(self):
        meter = AudioQualityMeter(max_records=2)
        meter.add(b"\x00\x80", 1)
        meter.add(bytes(2), 2)
        meter.add(bytes(2), 3)
        result = meter.snapshot(3)
        self.assertEqual(result["packets"], 2)
        self.assertIsNone(result["peak_dbfs"])
        self.assertTrue(result["record_cap_limited"])
        self.assertFalse(meter.snapshot(6.001)["record_cap_limited"])

    def test_big_packets_sample_at_most_one_second_across_the_whole_packet(self):
        meter = AudioQualityMeter()
        raw = bytes(RATE * 4 - 2) + b"\xff\x7f"
        meter.add(raw, 1)
        result = meter.snapshot(1)
        self.assertEqual(result["sampled_samples"], RATE)
        self.assertEqual(result["received_samples"], RATE * 2)
        self.assertEqual(result["sampled_audio_sec"], 1)
        self.assertEqual(result["received_audio_sec"], 2)
        self.assertEqual(result["coverage"], "sampled")
        self.assertEqual(result["coverage_scope"], "retained_packets")
        self.assertIsNotNone(result["peak_dbfs"])
        self.assertGreater(result["clipped_sample_pct"], 0)
        self.assertFalse(result["record_cap_limited"])
        self.assertTrue(all(not isinstance(value, (bytes, bytearray, memoryview))
                            for value in result.values()))

    def test_invalid_samples_and_unbounded_settings_are_rejected(self):
        for settings in ({"window_seconds": float("inf")}, {"max_records": 513},
                         {"max_samples_per_packet": RATE + 1}):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                AudioQualityMeter(**settings)
        meter = AudioQualityMeter()
        with self.assertRaises(ValueError):
            meter.add(b"\x00", 1)
        with self.assertRaises(ValueError):
            meter.add(bytes(2), float("nan"))


if __name__ == "__main__":
    unittest.main()
