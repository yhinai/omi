"""Explicit, labeled wrist trials; motion only, no audio or Mac input."""
from collections import Counter
from datetime import datetime, timezone
import math
import statistics
import time

from config import ROOT, VEXO_BAND
from motion import atomic_json, attitude

STEPS = [
 ('neutral',8,'Hold your LEFT wrist in the comfortable position you would use to control the Mac. Keep still.'),
 ('left_1',6,'Tilt comfortably LEFT, then hold that tilted position. Keep your arm position consistent.'),
 ('right_1',6,'Return to neutral, then tilt comfortably RIGHT and hold.'),
 ('up_1',6,'Return to neutral, then tilt comfortably UP and hold.'),
 ('down_1',6,'Return to neutral, then tilt comfortably DOWN and hold.'),
 ('left_2',6,'Repeat your LEFT tilt and hold, using the same comfortable range.'),
 ('right_2',6,'Repeat your RIGHT tilt and hold.'),
 ('up_2',6,'Repeat your UP tilt and hold.'),
 ('down_2',6,'Repeat your DOWN tilt and hold.'),
 ('slow',10,'Move slowly LEFT to RIGHT and back three times, passing through neutral.'),
 ('fast',8,'Repeat LEFT to RIGHT and back three times, briskly but comfortably.'),
 ('scroll',10,'Make three slow UP/DOWN wrist movements, as if scrolling a page.'),
 ('wave',10,'Wave LEFT/RIGHT three separate times. Pause in neutral between waves.'),
 ('tap',8,'Tap the band gently three separate times. Pause between taps.'),
 ('double_tap',8,'Double-tap the band three separate times, pausing between pairs.'),
 ('ordinary',20,'Move naturally: type, use the trackpad, adjust your arm. Do not make deliberate control gestures.'),
 ('neutral_end',8,'Return to your ORIGINAL comfortable control position and keep still.'),
]


def percentile(values, fraction):
    values=sorted(values)
    return values[min(len(values)-1,round((len(values)-1)*fraction))] if values else None


def angle_delta(a,b):return (a-b+180)%360-180


def summarize_trial(trial, neutral):
    samples=trial['samples'];events=trial['events']
    summary={'samples':len(samples),'events':dict(Counter(e.get('kind') for e in events))}
    if len(samples)<2:
        summary['usable']=False
        return summary
    duration=(samples[-1]['t']-samples[0]['t'])/1000
    angles=[attitude(s) for s in samples]
    roll=[angle_delta(r,neutral[0]) for r,p in angles];pitch=[p-neutral[1] for r,p in angles]
    magnitude=[math.sqrt(sum(float(s[k])**2 for k in ('ax','ay','az'))) for s in samples]
    rotation=[math.sqrt(sum(float(s.get(k) or 0)**2 for k in ('gx','gy','gz'))) for s in samples]
    ids=[s.get('n') for s in samples]
    missing=sum(max(0,b-a-1) for a,b in zip(ids,ids[1:]) if isinstance(a,int) and isinstance(b,int) and b>=a)
    summary.update(sensor_duration_sec=duration,measured_hz=(len(samples)-1)/duration if duration>0 else None,
        missing_motion_samples=missing,usable=duration>=trial['seconds']*.7 and missing<=len(samples)*.1,
        roll_median_deg=statistics.median(roll[len(roll)//3:]),pitch_median_deg=statistics.median(pitch[len(pitch)//3:]),
        roll_range_deg=[min(roll),max(roll)],pitch_range_deg=[min(pitch),max(pitch)],
        rotation_speed_deg_sec={'p50':percentile(rotation,.5),'p95':percentile(rotation,.95),'max':max(rotation)},
        acceleration_g={'p50':percentile(magnitude,.5),'p95':percentile(magnitude,.95),'max':max(magnitude)},
        telemetry_age_sec={'p50':percentile(trial['ages'],.5),'p95':percentile(trial['ages'],.95)})
    return summary


def summarize(trials):
    first=next((t for t in trials if t['name']=='neutral'),None)
    if not first or len(first['samples'])<20:raise ValueError('Neutral trial has too little motion data; repeat calibration')
    poses=[attitude(s) for s in first['samples']]
    neutral=(math.degrees(math.atan2(sum(math.sin(math.radians(r)) for r,p in poses),sum(math.cos(math.radians(r)) for r,p in poses))),statistics.median(p for r,p in poses))
    roll_noise=[abs(angle_delta(r,neutral[0])) for r,p in poses];pitch_noise=[abs(p-neutral[1]) for r,p in poses]
    noise=max(percentile(roll_noise,.95),percentile(pitch_noise,.95))
    result={'neutral_roll_pitch_deg':neutral,'neutral_noise_p95_deg':noise,
        'suggested_dead_zone_deg':round(max(2,min(15,noise*3+1)),2),
        'gyro_bias_deg_sec':{k:statistics.mean(float(s.get(k) or 0) for s in first['samples']) for k in ('gx','gy','gz')},
        'trials':{t['name']:summarize_trial(t,neutral) for t in trials},'quality_notes':[]}
    if noise>5:result['quality_notes'].append('Neutral moved substantially; repeat the still trial before choosing a dead zone.')
    for direction in ('left','right','up','down'):
        a=result['trials'].get(direction+'_1',{});b=result['trials'].get(direction+'_2',{})
        if a.get('usable') and b.get('usable'):
            axy=(a['roll_median_deg'],a['pitch_median_deg']);bxy=(b['roll_median_deg'],b['pitch_median_deg'])
            result.setdefault('repeat_difference_deg',{})[direction]=math.hypot(axy[0]-bxy[0],axy[1]-bxy[1])
    end=result['trials'].get('neutral_end',{})
    if end.get('usable'):result['return_to_neutral_difference_deg']=math.hypot(end['roll_median_deg'],end['pitch_median_deg'])
    result['direction_vectors_deg']={d:[result['trials'][d+'_1'].get('roll_median_deg'),result['trials'][d+'_1'].get('pitch_median_deg')] for d in ('left','right','up','down') if d+'_1' in result['trials']}
    for a,b in [('left','right'),('up','down')]:
        x=result['trials'].get(a+'_1',{});y=result['trials'].get(b+'_1',{})
        if x.get('usable') and y.get('usable'):
            dot=x['roll_median_deg']*y['roll_median_deg']+x['pitch_median_deg']*y['pitch_median_deg']
            if dot>=0:result['quality_notes'].append(f'{a}/{b} were not clearly opposite; repeat those labels before mapping actions.')
    result['complete']=len(trials)==len(STEPS) and all(v.get('usable') for v in result['trials'].values())
    if len({t.get('generation') for t in trials})>1:result['quality_notes'].append('Reference spans source reconnects; wrist/fit was reconfirmed between trials. No trial combines disconnected data.')
    result['limits']=['Labeled single-session reference, not a trained or validated gesture classifier.',
        'Telemetry age is host callback-to-view age, not physical movement-to-screen latency.',
        'Wrist position/strap changes require recalibration; absolute hand/screen position is not observed.',
        'Ordinary activity is a small negative sample, not a false-activation-rate benchmark.']
    return result


class Calibration:
    def __init__(self, hand='left'):
        self.hand=hand;self.index=0;self.active=None;self.trials=[];self.generation=None;self.saved=None;self.message=''
        self.seen=set();self.seen_events=set()
        self.report=None

    def start(self,state,now):
        if self.saved or self.active is not None:return
        sample=state.get('sample')
        if not state.get('enabled') or not sample or not 0<=now-sample.get('received_at',0)<.3:
            self.message='Wait for fresh motion before starting a trial.';return
        if self.generation and state['generation']!=self.generation:
            self.active=None
            self.generation=state['generation'];self.message='Source reconnected. Completed trials kept. Confirm the same wrist/fit, then press Space to repeat this trial.';return
        self.generation=state['generation']
        name,seconds,instruction=STEPS[self.index]
        self.active={'name':name,'seconds':seconds,'generation':self.generation,'started_at':now,'samples':[],'events':[],'ages':[]}
        self.seen=set();self.seen_events=set();self.message='Recording motion trial'

    def feed(self,state,now):
        if not self.active:return
        if (state.get('generation')!=self.generation or not state.get('sample')
                or not 0<=now-state['sample'].get('received_at',0)<.5):
            self.active=None;self.message='Input changed/stalled: this trial was discarded. Press Space to repeat it.';return
        for sample in state.get('recent_samples',[state['sample']]):
            key=(sample.get('n'),sample.get('t'),sample.get('received_at'))
            if key not in self.seen and self.active['started_at']<=sample.get('received_at',0)<=now:
                self.seen.add(key);self.active['samples'].append(dict(sample));self.active['ages'].append(max(0,now-sample['received_at']))
        for event in state.get('events',[]):
            key=event['event_id']
            if key not in self.seen_events and self.active['started_at']<=event.get('received_at',0)<=now:
                self.seen_events.add(key);self.active['events'].append(dict(event))
        if now-self.active['started_at']>=self.active['seconds']:
            if len(self.active['samples'])<self.active['seconds']*10:
                self.active=None;self.message='Too few samples. Press Space to repeat this trial.';return
            self.trials.append(self.active);self.active=None;self.index+=1;self.message='Trial complete. Rest, then press Space when ready for the next one.'
            if self.index==len(STEPS):
                report={'schema_version':1,'band':VEXO_BAND,'hand':self.hand,'strap':'snug','created_at':datetime.now(timezone.utc).isoformat(),
                    'audio_saved':False,'generation':self.generation,'trials':self.trials,'reference':summarize(self.trials)}
                self.report=report
                self.saved=ROOT/'logs'/('motion-calibration-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json')
                atomic_json(self.saved,report);self.message='Calibration reference saved. It has not changed Mac controls.'

    def lines(self,now):
        if self.saved:
            r=self.report['reference']
            return ['REFERENCE SAVED: '+str(self.saved),
                f"Neutral noise p95: {r['neutral_noise_p95_deg']:.2f} deg; suggested dead zone: {r['suggested_dead_zone_deg']:.2f} deg",
                'Repeat differences (degrees): '+str({k:round(v,1) for k,v in r.get('repeat_difference_deg',{}).items()}),
                'Quality notes: '+('; '.join(r['quality_notes']) or 'No automatic flags; intentional-gesture accuracy still needs validation.'),
                self.message,'q: quit; open the motion preview again normally.']
        name,seconds,instruction=STEPS[self.index]
        instruction=instruction.replace('LEFT wrist',self.hand.upper()+' wrist')
        remaining=max(0,seconds-(now-self.active['started_at'])) if self.active else seconds
        return [f'CALIBRATION {self.index+1}/{len(STEPS)} · {self.hand} wrist · {name}',
            instruction,f'{remaining:.1f} seconds remaining' if self.active else f'Press SPACE when ready. Trial lasts {seconds}s.',
            self.message,'Motion only. q quits. r discards/repeats the current trial.']
