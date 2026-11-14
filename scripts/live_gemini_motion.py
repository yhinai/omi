"""Live Gemini motion preview. Sends motion to Google; never posts Mac input."""
import argparse
import asyncio
from collections import deque
import fcntl
import json
import math
import os
from pathlib import Path
import select
import sys
import termios
import time
import tty

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from config import VEXO_BAND
from motion import LEASE, STATE, atomic_json
from scripts.benchmark_gemini_motion import AXES, decide, key_from_file

STATUS = ROOT/'logs/gemini-live-status.json'
RAW_CRITERIA = {
    'still': 'Nearly constant acceleration including gravity and little angular velocity.',
    'tilting': 'Smooth change of gravity direction caused by a mostly one-way tilt.',
    'rotating': 'Mostly one-way rotation, particularly turning around gravity with little change in gravity direction.',
    'back_and_forth': 'Clear repeated, approximately regular direction reversals.',
    'shaking': 'Irregular or brisk oscillating motion with substantial acceleration or angular variation.',
    'impact': 'A brief isolated acceleration or angular impulse; physical contact is possible but not established.',
    'unknown': 'Insufficient, mixed, conflicting or otherwise unclassified motion.',
}
RAW_INSTRUCTIONS = ('Classify the physical movement in these raw six-axis IMU samples. '
    'Use only the supplied samples and units. There is no personal calibration, neutral pose, '
    'labeled example, or screen frame. Acceleration includes gravity. '
    'Do not infer an intended computer command, screen direction, or a finger gesture. '
    'Choose the best supported physical movement, or unknown when ambiguous.')


def make_raw_input(trial):
    return json.dumps(dict(sensor=dict(type='six-axis wrist IMU',
        acceleration_unit='g including gravity',gyro_unit='degrees per second',
        time_unit='milliseconds since band startup',frame='band sensor axes; mounting and neutral unspecified'),
        columns=['t_ms',*AXES],
        samples=[[s['t'],*[s[k] for k in AXES]] for s in trial['samples']]),
        separators=(',',':'),allow_nan=False)


class MotionWindow:
    def __init__(self):
        self.generation = None
        self.samples = deque(maxlen=80)
        self.seen = deque(maxlen=160)

    def ingest(self, state, now):
        if state.get('generation') != self.generation:
            self.generation = state.get('generation')
            self.samples.clear();self.seen.clear()
        try:
            fresh = (state.get('band') == VEXO_BAND and bool(self.generation)
                     and state.get('connected') is True and state.get('enabled') is True
                     and 0 <= now-state['published_at'] < .35
                     and 0 <= now-state['sample']['received_at'] < .3)
        except (KeyError,TypeError): fresh = False
        if not fresh:
            self.samples.clear();return False
        for sample in state.get('recent_samples', []):
            try:
                values = [sample[k] for k in (*AXES,'t','received_at')]
                if not all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in values):
                    continue
                if not 0 <= now-sample['received_at'] < 1.5: continue
                identity = sample['received_at']
                if identity in self.seen: continue
                if self.samples and (sample['t'] <= self.samples[-1]['t']
                                     or identity <= self.samples[-1]['received_at']): continue
                self.seen.append(identity);self.samples.append(dict(sample))
            except (KeyError,TypeError): continue
        while self.samples and now-self.samples[0]['received_at'] > 1.2:
            self.samples.popleft()
        return True

    def trial(self):
        samples = list(self.samples)
        if len(samples) < 15 or samples[-1]['received_at']-samples[0]['received_at'] < .8:
            return None
        # Reject windows with missing telemetry, rather than compressing a hole.
        if any(b['received_at']-a['received_at'] > .2 or b['t']-a['t'] > 200
               for a,b in zip(samples,samples[1:])): return None
        return dict(samples=samples)


def result_current(generation, request_epoch, current_generation, current_epoch, now, window_end, fresh):
    return (fresh and generation == current_generation and request_epoch == current_epoch
            and 0 <= now-window_end < 2.)


async def run(args):
    key=key_from_file(args.env_file)
    window=MotionWindow();rows=[];pending=None;request_epoch=0
    automatic=args.auto;message='Hold a pose for a second, then press Space to ask Gemini.'
    last_request=-math.inf;last_lease=last_paint=0
    result=None;requests=0;started=time.monotonic();fresh=False
    interactive=sys.stdin.isatty() and sys.stdout.isatty()
    terminal=termios.tcgetattr(sys.stdin) if interactive else None
    report=ROOT/'logs'/time.strftime(f'gemini-live-%Y%m%dT%H%M%SZ-{os.getpid()}.json',time.gmtime())
    try:
        if interactive:tty.setcbreak(sys.stdin.fileno());sys.stdout.write('\x1b[?25l')
        while time.monotonic()-started < args.seconds:
            now,tick=time.time(),time.monotonic()
            if tick-last_lease > 1:
                LEASE.write_text('control');os.chmod(LEASE,0o600);last_lease=tick
            try:state=json.loads(STATE.read_text())
            except (OSError,ValueError):state={}
            old_generation=window.generation
            fresh=window.ingest(state,now)
            if window.generation != old_generation:
                result=None
            pressed=None
            if interactive and select.select([sys.stdin],[],[],0)[0]:
                pressed=os.read(sys.stdin.fileno(),1).decode(errors='ignore').lower()
            if pressed in ('q','\x1b'):break
            if pressed == 'a':
                automatic=not automatic;request_epoch+=1;result=None
                message='Automatic mode ON; try stillness, slow tilts, turns and back-and-forth motion.' if automatic else 'Automatic mode paused. Space tests one window.'
            trial=window.trial() if fresh else None
            if pending is not None and pending.done():
                context,task=pending.context,pending
                pending=None
                try:
                    answer=task.result()
                except RuntimeError as error:
                    message=str(error);automatic=False
                    rows.append(dict(error=message));break
                accepted=result_current(context['generation'],context['epoch'],window.generation,
                    request_epoch,now,context['window_end'],fresh)
                row=dict(**answer,usable=accepted,window_ms=context['window_ms'],
                         sample_count=context['sample_count'],completed_at=now)
                rows.append(row)
                atomic_json(report,dict(model='gemini-decision-1',advisory_only=True,calibration_used=False,rows=rows))
                if accepted:
                    result=row
                    message='Prediction for the last motion window. Hold another pose; Space asks again.'
                else:
                    result=None;message='Result expired or the source changed; wait for fresh motion and try again.'
                if not interactive:print(json.dumps(row),flush=True)
                if requests>=args.max_requests:
                    automatic=False;message='Request limit reached. Q closes this preview.'
                    if not interactive:break
            want = pressed == ' ' or (automatic and trial is not None and tick-last_request >= 2)
            if want and pending is None and requests<args.max_requests:
                if trial is None:
                    message='Waiting for at least 0.8 seconds of uninterrupted, fresh motion.'
                else:
                    text=make_raw_input(trial)
                    pending=asyncio.create_task(asyncio.to_thread(decide,key,text,
                        criteria=RAW_CRITERIA,instructions=RAW_INSTRUCTIONS))
                    pending.context=dict(generation=window.generation,epoch=request_epoch,
                        window_end=trial['samples'][-1]['received_at'],
                        window_ms=round(trial['samples'][-1]['t']-trial['samples'][0]['t']),
                        sample_count=len(trial['samples']))
                    requests+=1;last_request=tick
                    message='Gemini is classifying the last motion window…'
            if not fresh:result=None
            if tick-last_paint > .1:
                status=dict(advisory_only=True,calibration_used=False,model='gemini-decision-1',fresh=fresh,
                    automatic=automatic,pending=pending is not None,requests=requests,
                    max_requests=args.max_requests,result=result,message=message,
                    updated_at=now,pid=os.getpid())
                atomic_json(STATUS,status)
                if interactive:
                    label=result['choice'].upper() if result else '—'
                    detail=(f"Confidence {result['confidence']:.0%} | API {result['elapsed_ms']:.0f} ms | "
                            f"{result['sample_count']} samples / {result['window_ms']} ms | "
                            f"Result age {now-result['completed_at']:.1f}s" if result else '')
                    sample=window.samples[-1] if window.samples and fresh else None
                    accel=('Accelerometer g: '+ '  '.join(f'{k} {sample[k]:+.3f}' for k in AXES[:3]) if sample else 'Accelerometer: waiting')
                    gyro=('Gyro deg/s:      '+ '  '.join(f'{k} {sample[k]:+.1f}' for k in AXES[3:]) if sample else 'Gyro: waiting')
                    display='\n'.join([
                        'VEXO 43 → GEMINI DECISION 1 · LIVE PREVIEW',
                        'Live wrist motion goes to Google. This preview does not control the Mac.',
                        '', 'SPACE: classify one pose   A: automatic on/off   Q / Escape: quit',
                        'Try stillness → slow tilt → turn → back-and-forth → shake; do each for 2–3 seconds.',
                        'RAW DATA ONLY: every received sample in each window. No saved calibration or examples.',
                        '', f'LAST WINDOW: {label}', detail, '', message,
                        '',accel,gyro,
                        f"Motion: {'fresh' if fresh else 'waiting/stale'} | Auto: {automatic} | "
                        f"Requests {requests}/{args.max_requests}",
                        f'Time remaining: {max(0,args.seconds-(tick-started)):.0f} sec',
                        '', 'Confidence is provisional. Unknown/low confidence is useful feedback.',
                        'Captions continue separately. No second Bluetooth client is opened.',
                    ])
                    sys.stdout.write('\x1b[H\x1b[2J'+display+'\n');sys.stdout.flush()
                last_paint=tick
            await asyncio.sleep(.03)
    finally:
        LEASE.unlink(missing_ok=True)
        try:
            atomic_json(report,dict(model='gemini-decision-1',advisory_only=True,calibration_used=False,rows=rows,requests=requests))
            atomic_json(STATUS,dict(stopped=True,advisory_only=True,calibration_used=False,requests=requests,report=str(report),
                                    updated_at=time.time(),pid=os.getpid()))
        finally:
            if terminal:
                termios.tcsetattr(sys.stdin,termios.TCSADRAIN,terminal);sys.stdout.write('\x1b[?25h\n')
        # A sent request may still complete; its result cannot reach any Mac input code.
        if pending is not None:
            try:await pending
            except Exception:pass
        print('Live preview stopped. Report:',report,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file',type=Path,required=True)
    parser.add_argument('--seconds',type=int,default=120)
    parser.add_argument('--max-requests',type=int,default=24)
    parser.add_argument('--auto',action='store_true',help='Start automatic bounded classification')
    args=parser.parse_args()
    if not 1<=args.seconds<=300 or not 1<=args.max_requests<=24:
        parser.error('Use 1–300 seconds and 1–24 requests')
    if not (sys.stdin.isatty() and sys.stdout.isatty()) and not args.auto:
        parser.error('Use an interactive terminal, or explicitly select --auto for a bounded check')
    with (ROOT/'logs/vexo-motion-view.lock').open('w') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Close the existing motion/control viewer first; captions can stay open.');return 1
        try:asyncio.run(run(args))
        except KeyboardInterrupt:pass
    return 0


if __name__=='__main__':raise SystemExit(main())
