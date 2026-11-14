"""Local speaker embeddings only. No network or transcription; no raw audio storage."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import wave

import numpy as np
import sherpa_onnx
from speakers import MODEL,PROFILE


class Engine:
    def __init__(self,model):
        with Path(model).open('rb') as weights:
            self.digest=hashlib.file_digest(weights,'sha256').hexdigest()
        config=sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=str(model),num_threads=1,provider='cpu')
        if not config.validate():raise ValueError('Invalid speaker model')
        self.extractor=sherpa_onnx.SpeakerEmbeddingExtractor(config)

    def embedding(self,raw):
        if not 48000<=len(raw)<=256000 or len(raw)%2:raise ValueError('Use 1.5 to 8 seconds of PCM16')
        samples=np.frombuffer(raw,dtype='<i2').astype(np.float32)/32768
        if np.sqrt(np.mean(samples*samples))<.002:raise ValueError('Insufficient speech level')
        stream=self.extractor.create_stream()
        stream.accept_waveform(sample_rate=16000,waveform=samples)
        stream.input_finished()
        if not self.extractor.is_ready(stream):raise ValueError('Insufficient speech')
        vector=np.asarray(self.extractor.compute(stream),dtype=np.float32)
        if not np.all(np.isfinite(vector)) or np.linalg.norm(vector)==0:raise ValueError('Invalid embedding')
        return vector/np.linalg.norm(vector)

    def enroll(self,path,profile):
        with wave.open(str(path),'rb') as audio:
            if (audio.getnchannels(),audio.getsampwidth(),audio.getframerate(),audio.getcomptype())!=(1,2,16000,'NONE'):
                raise ValueError('Enrollment requires 16 kHz mono PCM16 WAV')
            if not 144000<=audio.getnframes()<=480000:raise ValueError('Use 9 to 30 seconds of only your voice')
            raw=audio.readframes(audio.getnframes())
        vectors=[self.embedding(raw[i:i+96000]) for i in range(0,len(raw)-95999,96000)]
        center=np.mean(vectors,axis=0);center/=np.linalg.norm(center)
        consistency=min(float(v@center) for v in vectors)
        if consistency<.75:raise ValueError('Enrollment inconsistent; use a quiet sample of one speaker')
        data={'version':1,'model_sha256':self.digest,'embedding':center.tolist(),
              'match_threshold':max(.8,consistency-.10),'other_threshold':.25,
              'threshold_status':'provisional; validate on held-out wearer and non-wearer speech',
              'segments':len(vectors)}
        profile=Path(profile);profile.parent.mkdir(parents=True,exist_ok=True)
        if profile.exists():raise ValueError('Profile exists; remove it explicitly before reenrolling')
        fd,temporary=tempfile.mkstemp(dir=profile.parent,prefix='.voice-')
        try:
            with os.fdopen(fd,'w') as output:json.dump(data,output)
            os.link(temporary,profile) # atomic no-overwrite publication; mode 0600
        finally:os.unlink(temporary)
        return {'enrolled':True,'segments':len(vectors),'consistency':round(consistency,3),'raw_audio_saved':False}

    def classify(self,raw,profile):
        if not Path(profile).is_file():return {'status':'uncertain','reason':'not_enrolled'}
        data=json.loads(Path(profile).read_text())
        if data.get('model_sha256')!=self.digest:raise ValueError('Profile model mismatch')
        target=np.asarray(data['embedding'],dtype=np.float32)
        if target.shape!=(self.extractor.dim,) or not np.all(np.isfinite(target)) or np.linalg.norm(target)==0:
            raise ValueError('Invalid profile')
        high,low=float(data['match_threshold']),float(data['other_threshold'])
        if not 0<=low<high<=1:raise ValueError('Invalid thresholds')
        vector=self.embedding(raw)
        score=float(vector@(target/np.linalg.norm(target)))
        status='match' if score>=high else 'other' if score<=low else 'uncertain'
        return {'status':status,'similarity':round(score,4),'threshold_status':data['threshold_status']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',default=str(MODEL));parser.add_argument('--profile',default=str(PROFILE))
    parser.add_argument('--enroll-wav');args=parser.parse_args()
    engine=Engine(args.model)
    if args.enroll_wav:
        print(json.dumps(engine.enroll(args.enroll_wav,args.profile)));return
    while line:=sys.stdin.buffer.readline(400000):
        try:
            if len(line)>=400000:raise ValueError('Input exceeds bound')
            raw=base64.b64decode(json.loads(line)['pcm'],validate=True)
            answer=engine.classify(raw,args.profile)
        except Exception:answer={'status':'uncertain','reason':'invalid_or_insufficient_audio'}
        print(json.dumps(answer),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as error:sys.exit(str(error))
