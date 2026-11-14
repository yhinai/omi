import unittest
import json

from scripts.live_gemini_motion import MotionWindow, make_raw_input, result_current


def state(generation='one'):
    samples=[dict(t=i*40,received_at=10+i*.04,ax=0.,ay=0.,az=1.,gx=0.,gy=0.,gz=0.) for i in range(30)]
    return dict(band=43,generation=generation,connected=True,enabled=True,published_at=11.2,
                sample=samples[-1],recent_samples=samples)


class WindowTests(unittest.TestCase):
    def test_fresh_contiguous_window_and_no_duplicate_samples(self):
        w=MotionWindow();s=state()
        self.assertTrue(w.ingest(s,11.2));self.assertIsNotNone(w.trial())
        count=len(w.samples);w.ingest(s,11.21)
        self.assertLessEqual(len(w.samples),count)  # The oldest point may age out.
        self.assertEqual(len({s['received_at'] for s in w.samples}),len(w.samples))

    def test_wrong_band_stale_and_disconnected_data_cannot_produce_window(self):
        for field,value in [('band',42),('connected',False),('enabled',False),('published_at',9.)]:
            with self.subTest(field=field):
                w=MotionWindow();w.ingest(state(),11.2)
                s=state();s[field]=value
                self.assertFalse(w.ingest(s,11.2));self.assertIsNone(w.trial())

    def test_reconnection_drops_previous_generation_samples(self):
        w=MotionWindow();w.ingest(state(),11.2)
        s=state('two');s['recent_samples']=s['recent_samples'][-2:]
        self.assertTrue(w.ingest(s,11.2));self.assertIsNone(w.trial())
        self.assertEqual(len(w.samples),2)

    def test_time_holes_are_not_compressed_into_gestures(self):
        w=MotionWindow();s=state();del s['recent_samples'][5:13]
        self.assertTrue(w.ingest(s,11.2));self.assertIsNone(w.trial())

    def test_invalid_numeric_samples_are_excluded(self):
        w=MotionWindow();s=state()
        for sample in s['recent_samples']:sample['gx']=float('nan')
        self.assertTrue(w.ingest(s,11.2));self.assertIsNone(w.trial())

    def test_raw_payload_preserves_every_sample_without_reference_or_derived_features(self):
        samples=state()['recent_samples']
        data=json.loads(make_raw_input(dict(samples=samples)))
        self.assertEqual(len(data['samples']),len(samples))
        for actual,expected in zip(data['samples'],samples):
            self.assertEqual(actual,[expected[k] for k in ('t','ax','ay','az','gx','gy','gz')])
        self.assertEqual(set(data),{'sensor','columns','samples'})
        self.assertNotIn('received_at',make_raw_input(dict(samples=samples)))

    def test_late_paused_replaced_or_disconnected_results_are_rejected(self):
        self.assertTrue(result_current('one',1,'one',1,12.,11.,True))
        for args in [('one',1,'two',1,12.,11.,True),('one',1,'one',2,12.,11.,True),
                     ('one',1,'one',1,14.,11.,True),('one',1,'one',1,12.,11.,False)]:
            self.assertFalse(result_current(*args))
