"""Speaker turns and an optional bounded, local verification worker (no ASR)."""
import asyncio
import base64
import json
from pathlib import Path

ASSETS = Path.home()/'Library/Application Support/OmiSpeaker'
MODEL = ASSETS/'wespeaker_en_voxceleb_resnet34.onnx'
PROFILE = ASSETS/'my-voice.json'


def turns(words, epoch='', identities=None):
    result=[]
    for word in words:
        speaker=word.get('speaker')
        if not isinstance(speaker,int) or isinstance(speaker,bool) or speaker<0:
            speaker=None
        key=f'{epoch}:{speaker}' if speaker is not None else None
        if not result or result[-1]['speaker_key']!=key:
            result.append(dict(speaker_id=speaker,speaker_key=key,text='',
                sample_offset=word['sample_offset'],sample_end_offset=word['sample_end_offset']))
        row=result[-1]
        row['text']=(row['text']+' '+word['word']).strip()
        row['sample_end_offset']=word['sample_end_offset']
        if isinstance(word.get('voice_identity'),dict):row['voice_identity']=word['voice_identity']
    for index,row in enumerate(result):
        identity=(identities or {}).get(str(index))
        if identity is not None:row['voice_identity']=identity
        row['label']=label(row)
    return result


def label(turn):
    identity=turn.get('voice_identity',{})
    if identity.get('status')=='match':return 'You'
    if identity.get('status')=='other':return 'Other speaker'
    number=turn.get('speaker_id')
    suffix=f'Speaker {number+1}' if isinstance(number,int) else 'Speaker unknown'
    return f'{suffix} (uncertain)' if identity else suffix


def display_text(event):
    rows=event.get('speaker_turns')
    if not isinstance(rows,list) or not rows:return event.get('text','')
    return ' '.join(f'[{label(t)}] {t["text"]}' for t in rows)


class VoiceClient:
    """One process, serialized inference, bounded PCM and response timeout."""
    def __init__(self,python,root,model=MODEL,profile=PROFILE):
        self.python,self.root,self.model,self.profile=map(str,(python,root,model,profile))
        self.process=None
        self.lock=asyncio.Lock()

    async def close(self):
        if self.process is not None:
            process,self.process=self.process,None
            if process.returncode is None:process.kill()
            await process.wait()

    async def classify(self,pcm):
        if not Path(self.profile).is_file():return {'status':'uncertain','reason':'not_enrolled'}
        if not 48000<=len(pcm)<=256000:return {'status':'uncertain','reason':'duration'}
        async with self.lock:
            try:
                async with asyncio.timeout(4):
                    if self.process is None or self.process.returncode is not None:
                        self.process=await asyncio.create_subprocess_exec(self.python,str(Path(self.root)/'voice_worker.py'),
                            '--model',self.model,'--profile',self.profile,
                            stdin=asyncio.subprocess.PIPE,stdout=asyncio.subprocess.PIPE,
                            stderr=asyncio.subprocess.DEVNULL,limit=4096)
                    self.process.stdin.write((json.dumps({'pcm':base64.b64encode(pcm).decode()})+'\n').encode())
                    await self.process.stdin.drain()
                    line=await self.process.stdout.readline()
                    result=json.loads(line)
                    if result.get('status') not in {'match','other','uncertain'}:raise ValueError('Worker response')
                    return result
            except BaseException as error:
                await self.close()
                if isinstance(error,asyncio.CancelledError):raise
                return {'status':'uncertain','reason':'worker_unavailable'}
