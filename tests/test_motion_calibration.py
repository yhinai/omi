import unittest
from unittest import mock
import motion_calibration as calibration


def sample(n, at):
    return dict(n=n,t=n*40,received_at=at,ax=0.,ay=0.,az=1.,gx=0.,gy=0.,gz=0.)


class CalibrationTests(unittest.TestCase):
    def test_waits_for_deliberate_start_and_fresh_input(self):
        c=calibration.Calibration()
        c.feed({},10);self.assertIsNone(c.active)
        c.start({},10);self.assertIsNone(c.active)

    def test_collects_each_sample_once_from_ring(self):
        c=calibration.Calibration()
        s=dict(generation='a',enabled=True,sample=sample(0,10),recent_samples=[sample(0,10)])
        c.start(s,10)
        c.feed(s,10.05);c.feed(s,10.1)
        self.assertEqual(len(c.active['samples']),1)

    def test_disconnection_discards_current_trial(self):
        c=calibration.Calibration()
        s=dict(generation='a',enabled=True,sample=sample(0,10))
        c.start(s,10);c.feed(dict(generation='b',sample=sample(1,10.1)),10.1)
        self.assertIsNone(c.active);self.assertEqual(c.index,0)

    def test_reconnect_between_trials_requires_fit_reconfirmation(self):
        c=calibration.Calibration();c.generation='a';c.index=2;c.trials=[{'old':'data'}]
        c.start(dict(generation='b',enabled=True,sample=sample(0,10)),10)
        self.assertEqual(c.index,2);self.assertEqual(c.trials,[{'old':'data'}]);self.assertIsNone(c.active)
        self.assertIn('Confirm the same wrist/fit',c.message)

    def test_completed_trial_has_noise_bias_rate_and_explicit_limits(self):
        neutral={'name':'neutral','seconds':8,'samples':[sample(n,10+n*.04) for n in range(201)],'ages':[.02]*201,'events':[]}
        r=calibration.summarize([neutral])
        self.assertEqual(r['neutral_noise_p95_deg'],0)
        self.assertEqual(r['trials']['neutral']['measured_hz'],25)
        self.assertEqual(r['gyro_bias_deg_sec']['gz'],0)
        self.assertFalse(r['complete'])
        self.assertIn('not a trained',r['limits'][0])

    def test_long_trial_collects_bounded_duration_and_advances(self):
        c=calibration.Calibration()
        s=dict(generation='a',enabled=True,sample=sample(0,10))
        c.start(s,10)
        for n in range(201):
            at=10+n*.04;s.update(sample=sample(n,at),recent_samples=[sample(n,at)])
            c.feed(s,at)
        self.assertIsNone(c.active);self.assertEqual(c.index,1);self.assertEqual(len(c.trials),1)

    def test_complete_labeled_session_saves_only_motion_and_reference(self):
        import tempfile,math,json
        from pathlib import Path
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(calibration,'ROOT',Path(directory)):
            (Path(directory)/'logs').mkdir()
            c=calibration.Calibration('left');at=100.;n=0
            for name,seconds,_ in calibration.STEPS:
                current=sample(n,at)
                s=dict(generation='a',enabled=True,sample=current,recent_samples=[current])
                c.start(s,at)
                for j in range(int(seconds*25)+1):
                    now=at+j*.04;v=sample(n+j,now)
                    if name.startswith('left'):v.update(ay=-.5,az=.866)
                    if name.startswith('right'):v.update(ay=.5,az=.866)
                    if name.startswith('up'):v.update(ax=.4,az=.916)
                    if name.startswith('down'):v.update(ax=-.4,az=.916)
                    s.update(sample=v,recent_samples=[v]);c.feed(s,now)
                n+=int(seconds*25)+1;at+=seconds+2
            self.assertIsNotNone(c.saved)
            r=json.loads(c.saved.read_text())
            self.assertFalse(r['audio_saved']);self.assertTrue(r['reference']['complete'])
            self.assertEqual(len(r['trials']),17)
            self.assertEqual(r['reference']['repeat_difference_deg']['left'],0)
            self.assertEqual(c.saved.stat().st_mode & 0o777,0o600)
