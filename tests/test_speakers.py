import asyncio
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock
from urllib.parse import parse_qs,urlsplit

from speakers import turns,display_text,VoiceClient
from backend_settings import DeepgramConfig
import bridge
import tui
from storage import TranscriptStore,DailyNotes
from test_bridge import live_state
from deepgram_stt import DeepgramSTT,_Packet

class SpeakerTests(unittest.TestCase):
    def words(self):return [dict(word='Hello.',speaker=0,sample_offset=0,sample_end_offset=100),dict(word='Hi.',speaker=1,sample_offset=100,sample_end_offset=200)]
    def test_turns_preserve_voice_changes_and_scope_ids(self):
        result=turns(self.words(),'epoch-a')
        self.assertEqual([t['label'] for t in result],['Speaker 1','Speaker 2'])
        self.assertNotEqual(result[0]['speaker_key'],turns(self.words(),'epoch-b')[0]['speaker_key'])
    def test_only_voice_match_becomes_you(self):
        result=turns(self.words(),'epoch',{'0':{'status':'match'},'1':{'status':'uncertain'}})
        self.assertEqual(result[0]['label'],'You')
        self.assertIn('uncertain',result[1]['label'])
        self.assertNotIn('You',display_text({'speaker_turns':turns(self.words(),'epoch')}))
    def test_voice_labels_never_rewrite_original_final(self):
        runtime=live_state();recognition={'words':self.words(),'connection_epoch':'abc'}
        with mock.patch.object(bridge,'state',runtime):
            event=bridge.make_event('transcript','Original punctuation!',recognition=recognition)
        self.assertEqual(event['text'],'Original punctuation!')
        self.assertEqual(len(event['speaker_turns']),2)
    def test_notes_and_viewer_include_labels_without_merging_unknown_into_speaker(self):
        event=dict(type='transcript',text='Hello. Hi.',timestamp='2026-11-14 12:00:00',event_id='one',
                   session_id=1,stream_id='s',audio_start_at=10,speaker_turns=turns(self.words(),'epoch'))
        viewer=tui.Viewer();viewer.add(event)
        viewer.add({**event,'event_id':'two','speaker_turns':[],'text':'No attribution.','audio_start_at':10.1})
        self.assertEqual(len(viewer.captions),3)
        self.assertEqual(viewer.captions[0].speaker_label,'Speaker 1')
        self.assertEqual(viewer.captions[0].text,'Hello.')
        with tempfile.TemporaryDirectory() as td:
            store=TranscriptStore(directory=td,append=lambda _:None,create_event=lambda *a,**kw:event,
                on_commit=lambda _:None,on_error=lambda *a:self.fail('save failed'),on_recovered=lambda:None)
            store.notes=DailyNotes(td);store.write(event)
            self.assertIn('[Speaker 2]',Path(td,'2026-11-14.md').read_text())
    def test_keyterms_are_bounded_and_encoded_separately(self):
        config=DeepgramConfig.from_env({'DEEPGRAM_API_KEY':'synthetic','OMI_DEEPGRAM_KEYTERMS':'["Vexo", "Yahya Alhinai"]'})
        stream=DeepgramSTT(api_key='synthetic',on_partial=lambda _:None,on_final=lambda *a,**k:True,
            on_gap=lambda *a,**k:None,on_latency=lambda *a:None,on_connection=lambda *a:None,keyterms=config.keyterms)
        self.assertEqual(parse_qs(urlsplit(stream.connection_url()).query)['keyterm'],list(config.keyterms))
        with self.assertRaises(ValueError):DeepgramConfig.from_env({'DEEPGRAM_API_KEY':'synthetic','OMI_DEEPGRAM_KEYTERMS':'"wrong"'})

class VoiceAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_profile_is_uncertain_and_does_not_spawn(self):
        with tempfile.TemporaryDirectory() as td:
            client=VoiceClient('missing',td,profile=Path(td,'absent'))
            self.assertEqual((await client.classify(bytes(64000)))['status'],'uncertain')
            self.assertIsNone(client.process)
    async def test_final_classifies_original_pcm_per_turn_before_release(self):
        classify=mock.AsyncMock(return_value={'status':'match','similarity':.99})
        stream=DeepgramSTT(api_key='synthetic',diarize=True,enrich_speaker=classify,
            on_partial=lambda _:None,on_final=lambda *a,**k:True,on_gap=lambda *a,**k:None,
            on_latency=lambda *a:None,on_connection=lambda *a:None)
        stream.retained.append(_Packet(b'\x00\x10'*64000,1,1,0,7,1))
        epoch=SimpleNamespace(start=0,end=64000,session=7)
        event={'is_final':True,'channel':{'alternatives':[{'words':[dict(word='test',start=0,end=2,speaker=0),dict(word='other',start=2,end=4,speaker=1)]}]}}
        await stream._annotate_speakers(event,epoch)
        self.assertEqual(classify.await_count,2)
        self.assertEqual(len(classify.call_args.args[0]),64000)
        self.assertEqual(event['channel']['alternatives'][0]['words'][0]['_voice_identity']['status'],'match')
    async def test_missing_or_short_audio_is_never_matched(self):
        classify=mock.AsyncMock()
        stream=DeepgramSTT(api_key='synthetic',diarize=True,enrich_speaker=classify,
            on_partial=lambda _:None,on_final=lambda *a,**k:True,on_gap=lambda *a,**k:None,
            on_latency=lambda *a:None,on_connection=lambda *a:None)
        event={'is_final':True,'channel':{'alternatives':[{'words':[dict(word='short',start=0,end=.5,speaker=0)]}]}}
        await stream._annotate_speakers(event,SimpleNamespace(start=0,end=10000,session=1))
        classify.assert_not_awaited()
