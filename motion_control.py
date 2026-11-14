"""Opt-in, explicitly engaged Vexo pointer and scroll controls."""
import asyncio
from collections import deque
import json
import math
import os
import select
import sys
import termios
import time
import tty

from config import ROOT, VEXO_BAND
from motion import LEASE, STATE, attitude, atomic_json

CONTROL_STATE = ROOT/'logs/vexo-control.json'


class ControlSession:
    """Decisions are separate from Quartz so replay tests never move the Mac."""
    def __init__(self):
        self.generation = None
        self.neutral = None
        self.axes = {}
        self.armed = False
        self.mode = 'pointer'
        self.gain = 1.0
        self.precision = False
        self.dead_zone = 6.0
        self.action = 'Hold wrist neutral and press C.'
        self.last_tick = None
        self.filtered = [0., 0.]
        self.scroll_remainder = 0.
        self.tap_click = False
        self.pending_tap = None
        self.seen_events = deque(maxlen=64)
        self.last_click = -math.inf

    def disarm(self, reason='Paused. Return to neutral, then press Space to resume.'):
        self.armed = False
        self.filtered = [0., 0.]
        self.scroll_remainder = 0.
        self.pending_tap = None
        self.action = reason

    @staticmethod
    def fresh(state, now):
        try:
            sample = state['sample']
            values = [sample[k] for k in ('ax','ay','az','gx','gy','gz','received_at')]
            if not all(isinstance(x, (int,float)) and not isinstance(x,bool) and math.isfinite(x) for x in values):
                return False
            magnitude = math.sqrt(sum(sample[k]**2 for k in ('ax','ay','az')))
            return (state.get('band') == VEXO_BAND and state.get('generation')
                    and state.get('connected') is True and state.get('enabled') is True
                    and 0 <= now-state['published_at'] < .35
                    and 0 <= now-sample['received_at'] < .3 and .65 < magnitude < 1.35)
        except (KeyError, TypeError, ValueError):
            return False

    def observe(self, state, now):
        if state.get('generation') != self.generation:
            self.generation = state.get('generation')
            self.neutral = None
            self.axes.clear()
            self.seen_events.clear()
            self.disarm('New connection. Hold wrist neutral and press C.')
        fresh = bool(self.fresh(state, now))
        if not fresh:
            self.disarm('Paused: motion stale, disconnected, or moving too abruptly.')
        return fresh

    def delta(self, sample):
        roll, pitch = attitude(sample)
        return ((roll-self.neutral[0]+180)%360-180, pitch-self.neutral[1])

    def key(self, key, state, now):
        if key in ('q','\x1b'):
            self.disarm('Stopped.')
            return 'quit' if key == 'q' else None
        fresh = self.observe(state, now)
        if key == 'm':
            self.mode = 'scroll' if self.mode == 'pointer' else 'pointer'
            self.disarm(f'{self.mode.title()} selected. Return to neutral and press Space.')
        elif key == 'p':
            self.precision = not self.precision
        elif key in ('+','='):
            self.gain = min(3., self.gain*1.25)
        elif key == '-':
            self.gain = max(.1, self.gain/1.25)
        elif key == 't':
            self.tap_click = not self.tap_click
            self.pending_tap = None
            self.action = f"Band tap clicking {'ON' if self.tap_click else 'OFF'}."
        elif key == 'c' and fresh:
            self.disarm()
            self.neutral = attitude(state['sample'])
            self.axes.clear()
            self.action = 'Neutral saved. Tilt comfortably RIGHT and press X.'
        elif key in ('x','y') and fresh and self.neutral is not None:
            self.disarm()
            vector = self.delta(state['sample'])
            length = math.hypot(*vector)
            if not 10 <= length <= 75:
                self.action = 'Use a comfortable 10–75 degree tilt, then try again.'
                return
            vector = tuple(v/length for v in vector)
            other = self.axes.get('y' if key == 'x' else 'x')
            if other and abs(vector[0]*other[0]+vector[1]*other[1]) > .9:
                self.action = 'Directions overlap too much. Try a different down/right wrist tilt.'
                return
            self.axes[key] = vector
            self.action = ('Right saved. Tilt comfortably DOWN and press Y.' if key == 'x'
                           else 'Down saved. Return to NEUTRAL and press Space to control the Mac.')
        elif key == ' ':
            if self.armed:
                self.disarm()
            elif fresh and self.neutral is not None and len(self.axes) == 2:
                # Recenter only at explicit engagement, never continuously while moving.
                self.neutral = attitude(state['sample'])
                self.armed = True
                self.filtered = [0., 0.]
                self.last_tick = None
                self.action = f'ACTIVE {self.mode}: tilt to move; neutral stops. Escape pauses.'
            else:
                self.action = 'Need fresh motion and C → X → Y calibration before Space.'
        elif key in ('\r','\n') and self.armed and fresh:
            if now-self.last_click >= .6:
                self.last_click = now
                return 'click'

    def step(self, state, now, tick):
        fresh = self.observe(state, now)
        dt = .02 if self.last_tick is None else tick-self.last_tick
        self.last_tick = tick
        if not 0 < dt <= .12:
            self.disarm('Paused: control loop was interrupted. Press Space to resume.')
        click = False
        for event in state.get('events', [])[-30:]:
            identity = event.get('event_id')
            if not identity or identity in self.seen_events:
                continue
            self.seen_events.append(identity)
            if not fresh or not 0 <= now-event.get('received_at',0) < .35:
                continue
            if event.get('kind') == 'double-tap':
                self.disarm('Double-tap: paused. Press Space to resume.')
            elif event.get('kind') == 'tap' and self.armed and self.tap_click:
                self.pending_tap = now + .55  # Double-tap can cancel the single click.
        if not self.armed or not fresh:
            return dict(armed=False, dx=0., dy=0., scroll=0, click=False)
        if self.pending_tap is not None and now >= self.pending_tap:
            self.pending_tap = None
            if now-self.last_click >= .6:
                self.last_click = now
                click = True
        roll, pitch = self.delta(state['sample'])
        x, y = self.axes['x'], self.axes['y']
        determinant = x[0]*y[1]-x[1]*y[0]
        tilt = ((roll*y[1]-pitch*y[0])/determinant, (pitch*x[0]-roll*x[1])/determinant)
        velocities = []
        for index, angle in enumerate(tilt):
            amount = max(0., abs(angle)-self.dead_zone)
            target = math.copysign(min(600., amount*16.*self.gain), angle)
            if self.precision: target *= .2
            # Immediate stop in neutral; short smoothing only while moving.
            self.filtered[index] = (0. if not amount else self.filtered[index]
                                     + (1-math.exp(-dt/.045))*(target-self.filtered[index]))
            velocities.append(self.filtered[index])
        dx, dy = (v*min(dt,.04) for v in velocities)
        scroll = 0
        if self.mode == 'scroll':
            self.scroll_remainder += dy*.65
            scroll = math.trunc(self.scroll_remainder)
            self.scroll_remainder -= scroll
            dx = dy = 0.
        return dict(armed=True, dx=dx, dy=dy, scroll=scroll, click=click)


def reference_dead_zone():
    for path in sorted((ROOT/'logs').glob('motion-calibration-*.json'), reverse=True):
        if path.name.endswith('-analysis.json'): continue
        try:
            report = json.loads(path.read_text())
            value = report['reference']['suggested_dead_zone_deg']
            if report.get('hand') == 'left' and isinstance(value,(int,float)) and math.isfinite(value):
                return max(3., min(10., value))
        except (OSError,ValueError,KeyError,TypeError):
            pass
    return 6.


async def main(args):
    from mac_input import MacInput
    mac = MacInput()
    session = ControlSession()
    session.dead_zone = reference_dead_zone()
    settings = termios.tcgetattr(sys.stdin)
    started = time.monotonic()
    last_lease = last_paint = 0
    held = set()
    permission_opened = False
    try:
        tty.setcbreak(sys.stdin.fileno())
        sys.stdout.write('\x1b[?25l')
        while not args.seconds or time.monotonic()-started < args.seconds:
            now, tick = time.time(), time.monotonic()
            if tick-last_lease > 1:
                LEASE.write_text('control');os.chmod(LEASE,0o600);last_lease=tick
            try: state = json.loads(STATE.read_text())
            except (OSError,ValueError): state = {}
            trusted, keys = mac.trusted(), mac.can_read_keys()
            if not trusted or not keys:
                session.disarm('Enable Accessibility and Input Monitoring for this terminal, then return here.')
                if not permission_opened:
                    import subprocess
                    subprocess.run(['open','x-apple.systempreferences:com.apple.preference.security?Privacy_Accessibility'],check=False)
                    permission_opened = True
            key = None
            if select.select([sys.stdin], [], [], 0)[0]: key = sys.stdin.read(1).lower()
            if keys:
                current = {code for code in (49,46,35,36) if mac.key_down(code)} if mac.control_chord() else set()
                for code in current-held:
                    key = {49:' ',46:'m',35:'p',36:'\r'}[code]
                held = current
                if mac.key_down(53): key = '\x1b'
            if key == 'q': break
            if key is not None:
                action = session.key(key, state, now)
                if action == 'click' and trusted and keys: mac.click()
            command = session.step(state, now, tick)
            if not trusted or not keys:
                session.disarm('Enable Terminal in System Settings → Privacy & Security → Accessibility AND Input Monitoring.')
            if command['armed'] and trusted and keys:
                if command['dx'] or command['dy']: mac.move(command['dx'], command['dy'])
                if command['scroll']: mac.scroll(command['scroll'])
                if command['click']: mac.click()
            if tick-last_paint > .1:
                status = 'ACTIVE' if session.armed and trusted and keys else 'PAUSED'
                atomic_json(CONTROL_STATE, dict(status=status, mode=session.mode,
                    fresh=bool(session.fresh(state,now)), accessibility=trusted, global_keys=keys,
                    axes=list(session.axes), action=session.action, updated_at=now, pid=os.getpid()))
                text = '\n'.join([
                    f'VEXO {VEXO_BAND} · MAC CONTROL · {status.upper()} · {session.mode.upper()}',
                    '', 'SETUP:  C at neutral → X tilted RIGHT → Y tilted DOWN → return to neutral → SPACE',
                    'SPACE: start/pause   M: pointer/scroll   ENTER: click   ESC: emergency pause',
                    'P: precision   +/-: speed   T: optional band tap clicking   Q: quit',
                    'In any app: Control+Option+Command + Space / M / P / Enter',
                    'Double-tap the band: pause. Return to neutral: stop moving.', '',
                    session.action, '',
                    f"Motion: {'fresh' if session.fresh(state,now) else 'waiting/stale'} | "
                    f"Axes: {'right' if 'x' in session.axes else '-'} / {'down' if 'y' in session.axes else '-'}",
                    f'Gain {session.gain:.2f}x | Precision {session.precision} | Dead zone {session.dead_zone:.2f}°',
                    f'Band tap clicking: {session.tap_click} | Accessibility: {trusted} | Global keys: {keys}',
                    f"Output: x {command['dx']:+.1f}, y {command['dy']:+.1f}, scroll {command['scroll']:+d}",
                    '', 'This window controls the actual Mac when ACTIVE. Captions continue separately.',
                ])
                sys.stdout.write('\x1b[H\x1b[2J'+text+'\n');sys.stdout.flush();last_paint=tick
            await asyncio.sleep(.02)
    finally:
        session.disarm('Stopped.')
        LEASE.unlink(missing_ok=True)
        try:
            atomic_json(CONTROL_STATE, dict(status='STOPPED', updated_at=time.time(), pid=os.getpid()))
        finally:
            termios.tcsetattr(sys.stdin,termios.TCSADRAIN,settings)
            sys.stdout.write('\x1b[?25h\n')
