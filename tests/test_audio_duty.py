"""Audio duty accounting: received audio seconds against session wall clock."""
import time
import unittest

import bridge


class AudioDutyTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(bridge.state.metrics)
        bridge.state.metrics.update(session_started_at=None, session_ended_at=None,
                                    stream_audio_bytes=0)

    def tearDown(self):
        bridge.state.metrics.clear()
        bridge.state.metrics.update(self.saved)

    def test_without_a_session_ratio_is_unknown(self):
        self.assertEqual(bridge.stream_audio_duty(), (0.0, None, None))

    def test_live_session_reports_partial_duty(self):
        bridge.state.metrics["session_started_at"] = time.monotonic() - 10
        bridge.state.metrics["stream_audio_bytes"] = bridge.BYTES_PER_SECOND * 5
        audio, wall, duty = bridge.stream_audio_duty()
        self.assertAlmostEqual(audio, 5.0, places=3)
        self.assertIsNotNone(wall)
        self.assertIsNotNone(duty)
        self.assertGreaterEqual(wall, 10.0)
        self.assertLessEqual(duty, 0.5)

    def test_wall_clock_freezes_when_the_phone_disconnects(self):
        now = time.monotonic()
        bridge.state.metrics["session_started_at"] = now - 20
        bridge.state.metrics["session_ended_at"] = now - 10
        bridge.state.metrics["stream_audio_bytes"] = bridge.BYTES_PER_SECOND * 5
        audio, wall, duty = bridge.stream_audio_duty()
        self.assertAlmostEqual(audio, 5.0, places=3)
        self.assertIsNotNone(wall)
        self.assertIsNotNone(duty)
        self.assertAlmostEqual(wall, 10.0, places=3)
        self.assertAlmostEqual(duty, 0.5, places=3)

    def test_a_zero_length_window_has_no_ratio(self):
        bridge.state.metrics["session_started_at"] = time.monotonic() + 5
        self.assertIsNone(bridge.stream_audio_duty()[2])


if __name__ == "__main__":
    unittest.main()
