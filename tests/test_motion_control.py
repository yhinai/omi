import math
import unittest
from unittest import mock

from motion_control import ControlSession


def state(roll=0., pitch=0., at=10., generation='one'):
    r, p = math.radians(roll), math.radians(pitch)
    sample = dict(ax=-math.sin(p), ay=math.sin(r)*math.cos(p), az=math.cos(r)*math.cos(p),
                  gx=0., gy=0., gz=0., received_at=at)
    return dict(band=43, generation=generation, connected=True, enabled=True,
                sample=sample, published_at=at, events=[])


class ControlTests(unittest.TestCase):
    def calibrated(self):
        c = ControlSession()
        c.key('c', state(), 10)
        c.key('x', state(roll=20), 10)
        c.key('y', state(pitch=20), 10)
        return c

    def armed(self):
        c = self.calibrated()
        c.key(' ', state(), 10)
        self.assertTrue(c.armed)
        return c

    def test_motion_never_arms_and_incomplete_calibration_cannot_arm(self):
        c = ControlSession()
        c.key(' ', state(), 10)
        self.assertFalse(c.step(state(roll=45), 10, 100)['armed'])
        c.key('c', state(), 10)
        c.key('x', state(roll=20), 10)
        c.key(' ', state(), 10)
        self.assertFalse(c.armed)

    def test_calibrated_tilt_moves_and_neutral_stops_immediately(self):
        c = self.armed()
        right = c.step(state(roll=20), 10, 100)
        self.assertGreater(right['dx'], 0)
        self.assertEqual(right['dy'], 0)
        down = c.step(state(pitch=20, at=10.02), 10.02, 100.02)
        self.assertGreater(down['dy'], 0)
        stopped = c.step(state(at=10.04), 10.04, 100.04)
        self.assertEqual((stopped['dx'], stopped['dy']), (0, 0))

    def test_axis_learning_decouples_rotated_wrist_directions(self):
        c = ControlSession()
        c.key('c', state(), 10)
        c.key('x', state(roll=20, pitch=20), 10)
        c.key('y', state(roll=-20, pitch=20), 10)
        c.key(' ', state(), 10)
        right = c.step(state(roll=10, pitch=10), 10, 100)
        self.assertGreater(right['dx'], 0)
        self.assertAlmostEqual(right['dy'], 0)

    def test_overlapping_axes_are_rejected(self):
        c = ControlSession()
        c.key('c', state(), 10)
        c.key('x', state(roll=20), 10)
        c.key('y', state(roll=30), 10)
        self.assertNotIn('y', c.axes)
        c.key(' ', state(), 10)
        self.assertFalse(c.armed)

    def test_stale_sample_snapshot_reconnect_and_wrong_band_disarm(self):
        for change in ('sample', 'snapshot', 'connection', 'band', 'generation', 'nan', 'impact'):
            with self.subTest(change=change):
                c = self.armed()
                s = state(at=10.1)
                if change == 'sample': s['sample']['received_at'] = 9
                if change == 'snapshot': s['published_at'] = 9
                if change == 'connection': s['connected'] = False
                if change == 'band': s['band'] = 42
                if change == 'generation': s['generation'] = 'replacement'
                if change == 'nan': s['sample']['ax'] = float('nan')
                if change == 'impact': s['sample']['ax'] = 4
                result = c.step(s, 10.1, 100.1)
                self.assertFalse(result['armed'])
                self.assertFalse(c.step(state(at=10.12), 10.12, 100.12)['armed'])

    def test_mode_change_pauses_then_scroll_does_not_move_pointer(self):
        c = self.armed()
        c.key('m', state(), 10)
        self.assertFalse(c.armed)
        c.key(' ', state(), 10)
        result = c.step(state(pitch=30), 10, 100)
        self.assertEqual((result['dx'], result['dy']), (0, 0))
        self.assertGreater(result['scroll'], 0)

    def test_loop_stall_pauses_instead_of_jumping(self):
        c = self.armed()
        c.step(state(roll=30), 10, 100)
        result = c.step(state(roll=30, at=12), 12, 102)
        self.assertFalse(result['armed'])
        self.assertEqual(result['dx'], 0)

    def test_precision_reduces_output_and_speed_is_bounded(self):
        c = self.armed()
        normal = c.step(state(roll=40), 10, 100)['dx']
        c = self.armed()
        c.key('p', state(), 10)
        self.assertAlmostEqual(c.step(state(roll=40), 10, 100)['dx'], normal/5)
        for _ in range(100): c.key('+', state(), 10)
        for n in range(100):
            result = c.step(state(roll=89, at=10+n*.02), 10+n*.02, 100+n*.02)
            self.assertLessEqual(abs(result['dx']), 24)

    def test_click_requires_armed_fresh_and_has_cooldown(self):
        c = self.calibrated()
        self.assertIsNone(c.key('\r', state(), 10))
        c.key(' ', state(), 10)
        self.assertEqual(c.key('\r', state(), 10), 'click')
        self.assertIsNone(c.key('\r', state(at=10.1), 10.1))
        self.assertIsNone(c.key('\r', state(), 12))

    def test_double_tap_pauses_and_cancels_delayed_click(self):
        c = self.armed()
        c.key('t', state(), 10)
        s = state()
        s['events'] = [dict(event_id='tap', kind='tap', received_at=10)]
        self.assertFalse(c.step(s, 10, 100)['click'])
        s = state(at=10.2)
        s['events'] = [dict(event_id='double', kind='double-tap', received_at=10.2)]
        self.assertFalse(c.step(s, 10.2, 100.02)['armed'])
        self.assertIsNone(c.pending_tap)

    def test_default_taps_do_not_click_and_old_events_never_replay(self):
        c = self.armed()
        s = state()
        s['events'] = [dict(event_id='tap', kind='tap', received_at=10)]
        c.step(s, 10, 100)
        self.assertIsNone(c.pending_tap)
        c.key('t', state(), 10)
        c.step(s, 10, 100.02)
        self.assertIsNone(c.pending_tap)
        s['events'] = [dict(event_id='old', kind='tap', received_at=9)]
        c.step(s, 10, 100.04)
        self.assertIsNone(c.pending_tap)

    def test_single_tap_click_is_emitted_once_after_double_tap_window(self):
        c = self.armed()
        c.key('t', state(), 10)
        s = state()
        s['events'] = [dict(event_id='tap', kind='tap', received_at=10)]
        c.step(s, 10, 100)
        clicks = []
        for n in range(1,40):
            clicks.append(c.step(state(at=10+n*.02), 10+n*.02, 100+n*.02)['click'])
        self.assertEqual(sum(clicks), 1)

    def test_escape_pauses_without_reengaging(self):
        c = self.armed()
        c.key('\x1b', state(), 10)
        self.assertFalse(c.step(state(roll=30), 10, 100)['armed'])
