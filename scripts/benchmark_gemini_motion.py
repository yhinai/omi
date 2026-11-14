"""Bounded Gemini Decision motion experiment. Default is offline; --run sends motion data.

Reads an existing reference, never the live microphone, screen, or Mac input backend.
First labeled poses are references; repeated poses/end-neutral/ordinary activity are
held out within the same session. This is not a cross-session accuracy benchmark.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import statistics
import sys
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from motion import attitude, atomic_json

ENDPOINT = 'https://generativelanguage.googleapis.com/v1beta/decisions'
MODEL = 'gemini-decision-1'
INSTRUCTIONS = ('Classify the intended control pose or activity in target from the sensor data and schema. '
                'Use labeled_reference examples when provided. Pick the best supported option; '
                'choose unknown when the evidence is ambiguous or represents ordinary activity.')
AXES = ('ax','ay','az','gx','gy','gz')
CRITERIA = {
    'neutral': 'Comfortable stationary neutral control pose.',
    'left': 'Deliberate held wrist tilt LEFT relative to neutral.',
    'right': 'Deliberate held wrist tilt RIGHT relative to neutral.',
    'up': 'Deliberate held wrist tilt UP relative to neutral.',
    'down': 'Deliberate held wrist tilt DOWN relative to neutral.',
    'unknown': 'Ordinary activity, ambiguous pose, insufficient reference, or no deliberate control gesture.',
}


def features(samples):
    poses = [attitude(s) for s in samples]
    roll = math.degrees(math.atan2(sum(math.sin(math.radians(r)) for r,p in poses),
                                   sum(math.cos(math.radians(r)) for r,p in poses)))
    return dict(roll_deg=round(roll,2), pitch_deg=round(statistics.median(p for r,p in poses),2),
                mean_acceleration_g=[round(statistics.mean(s[k] for s in samples),4) for k in AXES[:3]],
                gyro_rms_dps=round(math.sqrt(statistics.mean(sum(s[k]**2 for k in AXES[3:]) for s in samples)),2))


def observation(trial):
    samples = trial['samples']
    points = [samples[round(i*(len(samples)-1)/11)] for i in range(12)]
    return dict(summary=features(samples),
                duration_ms=round(samples[-1]['t']-samples[0]['t']),
                columns=['elapsed_ms',*AXES],
                samples=[[round(s['t']-samples[0]['t']), *[round(s[k],4) for k in AXES]] for s in points])


def make_input(trial, reference, *, calibrated, encoding):
    data = dict(sensor=dict(type='wrist six-axis IMU', wrist='left',
        acceleration_unit='g including gravity', gyro_unit='degrees per second',
        attitude_convention='roll=atan2(ay,az); pitch=atan2(-ax,hypot(ay,az)); degrees',
        note='Sensor axes follow the band. No camera or absolute hand position.'),
        target=observation(trial))
    if calibrated: data['labeled_reference'] = reference
    if encoding == 'json': return json.dumps(data,separators=(',',':'))
    # Same values in XML; formatting is the only intended encoding difference.
    def convert(parent, key, value):
        element = ET.SubElement(parent,key)
        if isinstance(value,dict):
            for name,item in value.items():convert(element,name,item)
        elif isinstance(value,list):
            for item in value:convert(element,'item',item)
        else:element.text=str(value)
    root=ET.Element('motion_input')
    for key,value in data.items():convert(root,key,value)
    return ET.tostring(root,encoding='unicode')


def key_from_file(path):
    if path is None:
        key=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY')
        if key:return key
    else:
        for line in Path(path).read_text().splitlines():
            match=re.match(r'^\s*(?:export\s+)?(?:GEMINI_API_KEY|GOOGLE_API_KEY)\s*=\s*(.*)$',line)
            if match:
                parts=shlex.split(match[1],comments=True)
                if len(parts)==1 and parts[0]:return parts[0]
    raise ValueError('No Gemini key found; no request sent')


def decide(key, text, *, criteria=None, instructions=None):
    criteria = CRITERIA if criteria is None else criteria
    instructions = INSTRUCTIONS if instructions is None else instructions
    payload=dict(model=MODEL,input=text,questions=dict(movement=dict(type='choice',
        instructions=instructions,criteria=criteria)))
    request=urllib.request.Request(ENDPOINT,data=json.dumps(payload).encode(),
        headers={'x-goog-api-key':key,'Content-Type':'application/json'})
    start=time.monotonic()
    try:
        with urllib.request.urlopen(request,timeout=10) as response:result=json.load(response)
    except urllib.error.HTTPError as error:
        raise RuntimeError(f'Gemini decision HTTP {error.code}; provider body omitted') from None
    except Exception as error:
        raise RuntimeError('Gemini decision failed: '+type(error).__name__) from None
    elapsed=1000*(time.monotonic()-start)
    answer=result.get('answers',{}).get('movement',{})
    choice=answer.get('choice');confidence=answer.get('confidence');probabilities=answer.get('probabilities')
    if (choice not in criteria or not isinstance(confidence,(int,float)) or isinstance(confidence,bool)
            or not math.isfinite(confidence) or not 0<=confidence<=1 or not isinstance(probabilities,dict)
            or set(probabilities)!=set(criteria)
            or any(not isinstance(p,(int,float)) or isinstance(p,bool) or not math.isfinite(p) or not 0<=p<=1
                   for p in probabilities.values()) or abs(sum(probabilities.values())-1)>.01):
        raise RuntimeError('Invalid Gemini decision response; no action available')
    return dict(choice=choice,confidence=confidence,probabilities=probabilities,
                elapsed_ms=round(elapsed,1),input_tokens=result.get('usage',{}).get('total_input_tokens'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--env-file',type=Path)
    parser.add_argument('--run',action='store_true',help='Send this bounded set of saved motion observations to Google')
    args=parser.parse_args()
    source=json.loads(args.reference.read_text())
    trials={trial['name']:trial for trial in source['trials']}
    reference={label:features(trials[name]['samples']) for label,name in
               [('neutral','neutral'),('left','left_1'),('right','right_1'),('up','up_1'),('down','down_1')]}
    held_out=[('neutral_end','neutral'),('left_2','left'),('right_2','right'),('up_2','up'),('down_2','down'),('ordinary','unknown')]
    rows=[]
    key=key_from_file(args.env_file) if args.run else None
    for calibrated,encoding in [(False,'json'),(False,'xml'),(True,'json'),(True,'xml')]:
        mode=('reference_' if calibrated else 'schema_')+encoding
        for name,expected in held_out:
            text=make_input(trials[name],reference,calibrated=calibrated,encoding=encoding)
            row=dict(mode=mode,trial=name,expected=expected,input_characters=len(text),
                     observed_duration_ms=observation(trials[name])['duration_ms'])
            if args.run:
                try:row.update(decide(key,text))
                except RuntimeError as error:
                    row['error']=str(error);rows.append(row)
                    atomic_json(args.output,dict(model=MODEL,rows=rows,incomplete=True))
                    print(row['error']);return 1
                row['correct']=row['choice']==expected
                print(mode,name,'→',row['choice'],f"confidence={row['confidence']:.3f}",f"{row['elapsed_ms']}ms",flush=True)
            rows.append(row)
    summaries={}
    for mode in sorted({r['mode'] for r in rows}):
        subset=[r for r in rows if r['mode']==mode]
        summaries[mode]=dict(cases=len(subset),correct=sum(r.get('correct',False) for r in subset) if args.run else None,
            mean_input_characters=round(statistics.mean(r['input_characters'] for r in subset)))
        if args.run:
            summaries[mode].update(median_ms=round(statistics.median(r['elapsed_ms'] for r in subset),1),
                                  max_ms=max(r['elapsed_ms'] for r in subset),
                                  input_tokens=sum(r['input_tokens'] or 0 for r in subset))
    report=dict(model=MODEL,endpoint=ENDPOINT,ran_api=args.run,question_instructions=INSTRUCTIONS,
        rows=rows,summary=summaries,
        limitations=['Six held-out trials in one previously recorded session, one API result per condition.',
            'First direction trials and starting neutral are references; repeats/end-neutral/ordinary are tests.',
            'Whole 6–20 second trials downsampled to 12 points; not a live-window gesture benchmark.',
            'XML vs JSON conditions run sequentially, with no randomization; timing differences are descriptive.',
            'No screen/audio input, no live Mac actions, no automatic integration into the control loop.'])
    atomic_json(args.output,report)
    print(json.dumps(summaries,indent=2));return 0


if __name__=='__main__':raise SystemExit(main())
