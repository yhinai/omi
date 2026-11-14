import asyncio
import json
import tempfile
import time
from pathlib import Path
import unittest
from unittest import mock

import motion


def sample(at=10, **values):
    return dict(received_at=at, ax=0.,ay=0.,az=1.,gx=0.,gy=0.,gz=0.,**values)


def state(s, generation='test'):
    return dict(connected=True,enabled=True,generation=generation,sample=s)


class ControllerTests(unittest.TestCase):
    def test_starts_disarmed_and_calibrates_neutral(self):
        c=motion.Controller();p=c.update(state(sample()),10)
        self.assertFalse(p['armed']);self.assertEqual((p['dx'],p['dy']),(0,0))

    def test_stale_or_disconnected_input_disarms(self):
        c=motion.Controller();c.update(state(sample()),10);c.armed_until=50
        self.assertFalse(c.update(state(sample()),11)['armed'])
        self.assertEqual(c.armed_until,0)

    def test_new_session_cannot_inherit_control_or_calibration(self):
        c=motion.Controller();c.update(state(sample()),10);c.armed_until=50
        p=c.update(state(sample(at=11),generation='replacement'),11)
        self.assertFalse(p['armed'])

    def test_tilt_is_bounded_and_dead_zone_ignores_small_tremor(self):
        c=motion.Controller();c.update(state(sample()),10)
        s=sample(at=10.1);s.update(ay=.01)
        self.assertEqual(c.update(state(s),10.1)['dx'],0)
        s.update(ay=1.,az=.01)
        self.assertLessEqual(c.update(state(s),10.1)['dx'],12)

    def test_wave_requires_alternating_rotation_and_has_cooldown(self):
        c=motion.Controller()
        values=[]
        for at,gz in [(10,150),(10.1,-150),(10.2,150),(10.3,-150),(10.4,150)]:
            s=sample(at=at);s['gz']=gz
            values.append(c.update(state(s),at)['dismiss'])
        self.assertEqual(values,[False,False,True,False,False])

    def test_motion_alone_never_arms(self):
        c=motion.Controller()
        for n in range(200):
            s=sample(at=10+n*.02);s['gz']=150 if n%2 else -150
            self.assertFalse(c.update(state(s),s['received_at'])['armed'])


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    async def test_control_lease_publishes_faster_without_raising_sensor_rate_or_changing_audio(self):
        with tempfile.TemporaryDirectory() as d:
            lease=Path(d)/'lease';output=Path(d)/'state.json'
            lease.write_text('control');telemetry=motion.MotionTelemetry();band=mock.Mock()
            with mock.patch.object(motion,'LEASE',lease),mock.patch.object(motion,'STATE',output):
                await telemetry.tick(band)
                band.configure.assert_called_once_with(motion=25,events=True)
                self.assertEqual(telemetry.rate,25)
                self.assertTrue(telemetry.controlling)
                lease.write_text('preview');telemetry.last_poll=0
                await telemetry.tick(band)
                band.configure.assert_called_with(motion=25,events=True)
                self.assertFalse(telemetry.controlling)

    async def test_lease_controls_only_motion_and_events(self):
        with tempfile.TemporaryDirectory() as d:
            lease=Path(d)/'lease';output=Path(d)/'state.json'
            band=mock.Mock();telemetry=motion.MotionTelemetry()
            with mock.patch.object(motion,'LEASE',lease),mock.patch.object(motion,'STATE',output):
                lease.touch();await telemetry.tick(band)
                band.configure.assert_called_once_with(motion=25,events=True)
                telemetry.on_motion(sample(at=time.time()))
                telemetry.publish();self.assertTrue(json.loads(output.read_text())['enabled'])
                lease.unlink();telemetry.last_poll=0;await telemetry.tick(band)
                band.configure.assert_called_with(motion=False,events=False)
                telemetry.close();self.assertFalse(json.loads(output.read_text())['connected'])
                self.assertEqual(output.stat().st_mode & 0o777,0o600)

    def test_event_buffer_is_bounded(self):
        t=motion.MotionTelemetry()
        for n in range(100):t.on_event({'kind':'tap','t':n})
        self.assertEqual(len(t.events),30)

    def test_optional_telemetry_disk_failure_does_not_interrupt_capture_cleanup(self):
        t=motion.MotionTelemetry()
        with mock.patch.object(motion,'atomic_json',side_effect=OSError('synthetic disk failure')):
            t.close()
        self.assertEqual(t.error,'OSError')

class PrecisionTests(unittest.TestCase):
    def test_precision_gain_reduces_small_corrections_fivefold(self):
        c=motion.Controller();c.update(state(sample()),10)
        s=sample(at=10.1);s.update(ay=.3,az=1)
        normal=c.update(state(s),10.1)['dx']
        c.gain=.2
        fine=c.update(state(s),10.1)['dx']
        self.assertAlmostEqual(fine,normal/5)

    def test_calibration_at_roll_boundary_avoids_full_speed_jump(self):
        c=motion.Controller()
        a=sample();a.update(ay=.01,az=-1)
        c.update(state(a),10)
        b=sample(at=10.1);b.update(ay=-.01,az=-1)
        self.assertEqual(c.update(state(b),10.1)['dx'],0)
