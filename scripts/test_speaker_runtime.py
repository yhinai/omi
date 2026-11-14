"""Optional real model checks. Run in .venv-speaker; only synthesized voices are used."""
import asyncio
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import wave

from voice_worker import Engine
from speakers import MODEL,VoiceClient

class SpeakerRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_enroll_heldout_and_other_voice_with_private_profile(self):
        engine=Engine(MODEL)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            samples=[('reference','Samantha','The morning light came through the window while I prepared breakfast. Later I walked to the library and read a book about the ocean. In the afternoon I visited a friend and we talked about our plans for the weekend. We decided to take a train to the mountains.'),
                     ('same','Samantha','My favorite activities include exploring new places and meeting people. The weather today is pleasant and I look forward to spending time outside.'),
                     ('other','Daniel','My favorite activities include exploring new places and meeting people. The weather today is pleasant and I look forward to spending time outside.')]
            for name,voice,text in samples:
                subprocess.run(['say','-v',voice,'-o',str(root/(name+'.aiff')),text],check=True,capture_output=True)
                subprocess.run(['afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(root/(name+'.aiff')),str(root/(name+'.wav'))],check=True,capture_output=True)
            profile=root/'profile.json'
            self.assertTrue(engine.enroll(root/'reference.wav',profile)['enrolled'])
            self.assertEqual(profile.stat().st_mode&0o777,0o600)
            with self.assertRaises(ValueError):engine.enroll(root/'reference.wav',profile)
            client=VoiceClient(os.sys.executable,Path(__file__).resolve().parents[1],profile=profile)
            try:
                for name in ['same','other']:
                    with wave.open(str(root/(name+'.wav')),'rb') as audio:pcm=audio.readframes(16000*7)
                    answer=await client.classify(pcm)
                    if name=='same':self.assertEqual(answer['status'],'match')
                    else:self.assertNotEqual(answer['status'],'match')
                self.assertEqual((await client.classify(bytes(16000)))['status'],'uncertain')
                profile.unlink()
                self.assertEqual((await client.classify(pcm))['reason'],'not_enrolled')
            finally:await client.close()
