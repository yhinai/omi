"""Live motion preview, guided calibration, or opt-in Mac controls (--control)."""
import argparse
import asyncio
import fcntl
import json
import math
import os
import select
import shutil
import textwrap
import sys
import termios
import time
import tty

from config import ROOT, VEXO_BAND
from motion import LEASE, STATE, Controller, atomic_json
from motion_calibration import Calibration, STEPS



def render(state, proposal, action, controller, mode):
    sample=state.get('sample') or {}
    events=state.get('events',[])[-5:]
    lines=[f'VEXO {VEXO_BAND} MOTION · DISPLAY ONLY',
           'r: guided reference   c: neutral   +/-: gain   p: precision   d: dead zone   m: mode   q: quit',
           'Tilt and wave proposals are shown for training; this viewer only displays data.',
           f"Connection: {'fresh' if proposal['fresh'] else 'waiting/stale — controls disarmed'} | Samples: {state.get('samples',0)}",
           f"Acceleration g: X {sample.get('ax',0):+.3f}  Y {sample.get('ay',0):+.3f}  Z {sample.get('az',0):+.3f}",
           f"Gyro deg/s:     X {sample.get('gx',0):+.1f}  Y {sample.get('gy',0):+.1f}  Z {sample.get('gz',0):+.1f}",
           f"Roll {proposal.get('roll',0):+.1f}°  Pitch {proposal.get('pitch',0):+.1f}° | Movement proposal {proposal['dx']:+.1f}, {proposal['dy']:+.1f}",
           f'Preview mode: {mode} | Gain {controller.gain:.2f}x | Dead zone {controller.dead_zone:.0f} degrees',
           f'Last action: {action}', 'Recent band events:']
    grid=[list('·'*25) for _ in range(5)]
    grid[2][12]='+'
    grid[max(0,min(4,2+round(proposal['dy']/6)))][max(0,min(24,12+round(proposal['dx']))) ]='●'
    lines[9:9]=['Tilt preview (+ = neutral):']+['  '+''.join(row) for row in grid]
    lines += [f"  {time.strftime('%H:%M:%S',time.localtime(e['received_at']))}  {e.get('kind')}  {e.get('face') or e.get('arg') or ''}" for e in reversed(events)]
    lines += ['Wave detection is a tunable heuristic. This view does not send Mac input.']
    return '\n'.join(lines)


async def main(args):
    controller=Controller()
    calibration=Calibration(args.hand) if args.calibrate else None
    base_gain=1.0; precision=False; mode="pointer"
    interactive=sys.stdin.isatty() and sys.stdout.isatty() and not args.json
    settings=termios.tcgetattr(sys.stdin) if interactive else None
    start=time.monotonic(); last_lease=0; action='none'; action_at=0; last_print=0; last_progress=0
    try:
        if interactive:tty.setcbreak(sys.stdin.fileno());sys.stdout.write('\x1b[?25l')
        while not args.seconds or time.monotonic()-start<args.seconds:
            now=time.time()
            if now-last_lease>1:
                LEASE.write_text('preview');os.chmod(LEASE,0o600);last_lease=now
            try:
                state=json.loads(STATE.read_text())
            except (OSError,ValueError):state={}
            proposal=controller.update(state,now)
            if interactive and select.select([sys.stdin],[],[],0)[0]:
                key=sys.stdin.read(1)
                if key in ('q','\x1b'):break
                if calibration:
                    if key==' ':calibration.start(state,now)
                    if key=='r':calibration.active=None;calibration.message='Press Space to repeat this trial.'
                elif key=='r':calibration=Calibration(args.hand)
                if key=='c' and proposal['fresh']:
                    controller.calibrate(state['sample']);controller.disarm();action='neutral calibrated'
                if key in ('+','='):base_gain=min(3.0,base_gain*1.25)
                if key=='-':base_gain=max(.1,base_gain/1.25)
                if key=='p':precision=not precision
                if key=='d':controller.dead_zone={3:6,6:10,10:3}.get(controller.dead_zone,6)
                if key=='m':mode='scroll' if mode=='pointer' else 'pointer'
                controller.gain=base_gain*(.2 if precision else 1)
            if calibration:
                calibration.feed(state,now)
                if now-last_progress>=1:
                    atomic_json(ROOT/'logs/vexo-calibration-progress.json',dict(
                        step=min(calibration.index+1,len(STEPS)),total=len(STEPS),
                        active=calibration.active is not None,completed_trials=len(calibration.trials),
                        saved=str(calibration.saved) if calibration.saved else None,
                        samples_in_trial=len(calibration.active['samples']) if calibration.active else 0,
                        message=calibration.message,updated_at=now))
                    last_progress=now
            if proposal.get('dismiss'):
                action='wave candidate (preview only)';action_at=now
            if calibration and calibration.saved:
                print('\x1b[H\x1b[2J'+'\n'.join(calibration.lines(now)),flush=True)
                break
            if now-last_print>.1:
                if interactive:
                    if calibration:
                        width,height=shutil.get_terminal_size((90,28))
                        lines=[]
                        for line in calibration.lines(now):lines.extend(textwrap.wrap(line,max(30,width-2)) or [''])
                        lines += ['',f"Live samples: {state.get('samples',0)} | {'fresh' if proposal['fresh'] else 'waiting for fresh input'}",
                            f"Roll {proposal.get('roll',0):+.1f} deg  Pitch {proposal.get('pitch',0):+.1f} deg",
                            'Built-in events: '+', '.join(str(e.get('kind')) for e in state.get('events',[])[-4:])]
                        display='\n'.join(lines[:max(8,height-2)])
                    else:display=render(state,proposal,action,controller,mode)
                    sys.stdout.write('\x1b[H\x1b[2J'+display+'\n');sys.stdout.flush()
                elif args.json:
                    print(json.dumps(dict(proposal=proposal,telemetry=state)),flush=True)
                elif now-last_print>1:print(render(state,proposal,action,controller,mode),flush=True)
                last_print=now
            await asyncio.sleep(.02)
    finally:
        controller.disarm()
        LEASE.unlink(missing_ok=True)
        if settings:termios.tcsetattr(sys.stdin,termios.TCSADRAIN,settings);sys.stdout.write('\x1b[?25h\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibrate',action='store_true',help='Guided labeled reference, with Space starting each trial')
    parser.add_argument('--control',action='store_true',help='Opt in to actual Mac pointer and scroll control')
    parser.add_argument('--hand',choices=('left','right'),default='left')
    parser.add_argument('--json',action='store_true',help='Stream telemetry/proposals as JSON; no controls')
    parser.add_argument('--seconds',type=float,default=0,help='Stop after this many seconds; 0 means until quit')
    args=parser.parse_args()
    if args.control and (args.calibrate or args.json or not sys.stdin.isatty() or not sys.stdout.isatty()):
        parser.error('--control needs an interactive terminal and cannot combine with --calibrate or --json')
    if args.calibrate and (not sys.stdin.isatty() or not sys.stdout.isatty() or args.json):
        parser.error("--calibrate needs an interactive terminal for deliberately starting each trial")
    if not math.isfinite(args.seconds) or args.seconds<0:parser.error('--seconds must be finite and nonnegative')
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/vexo-motion-view.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if args.control:
            from motion_control import main as control_main
        try:asyncio.run(control_main(args) if args.control else main(args))
        except KeyboardInterrupt:pass
