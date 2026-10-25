"""Offline tests for the isolated magazine prototype."""
import base64
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch

from aiohttp.test_utils import TestClient, TestServer

spec=importlib.util.spec_from_file_location('magazine_engine',Path(__file__).with_name('engine.py'))
engine=importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)

class EditorialTests(unittest.TestCase):
    def edition(self):
        data=deepcopy(engine.DEMO)
        for group in ('sections','takeaways','followups'):
            for item in data[group]:
                item['source_ids']=['original']
        return data

    def test_unknown_sources_and_missing_sections_are_rejected(self):
        data=self.edition()
        engine.validate_edition(data,[{'id':'original'}])
        data['sections'][0]['source_ids']=['invented']
        with self.assertRaises(ValueError):
            engine.validate_edition(data,[{'id':'original'}])
        with self.assertRaises(ValueError):
            engine.validate_edition({'title':'x','standfirst':'x'},[])

    def test_transcript_reader_filters_date_deduplicates_and_counts_gaps(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder,'events.jsonl')
            events=[{'event_id':'a','timestamp':'2026-10-24 10:00:00','text':'Recorded evidence'},
                    {'event_id':'g','timestamp':'2026-10-24 10:00:01','type':'gap','text':'gap'},
                    {'event_id':'b','timestamp':'2026-10-23 10:00:00','text':'Other day'}]
            path.write_text('\n'.join(json.dumps(e) for e in events)+'\ninvalid\nnull\n[]\n42\n')
            path.with_name(path.name+'.1').write_text(json.dumps(events[0])+'\n')
            with patch.object(engine,'SOURCE',path):
                sources,gaps=engine.day_events('2026-10-24')
            self.assertEqual([s['id'] for s in sources],['a'])
            self.assertEqual(gaps,1)

class RoutesTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.client=TestClient(TestServer(engine.application()))
        await self.client.start_server()
        self.addAsyncCleanup(self.client.close)

    async def test_preview_does_not_call_ai(self):
        with patch.object(engine,'gemini',new_callable=AsyncMock) as api:
            page=await self.client.get('/')
            self.assertEqual(page.status,200)
            self.assertIn('fieldnotes',await page.text())
            response=await self.client.get('/api/demo')
            self.assertTrue((await response.json())['demo'])
            api.assert_not_awaited()

    async def test_cross_origin_generation_is_rejected(self):
        response=await self.client.post('/api/generate',json={'date':'2026-10-24'},headers={'Origin':'https://example.invalid'})
        self.assertEqual(response.status,403)

    async def test_hostile_host_cannot_read_editions_or_invoke_generation(self):
        response=await self.client.get('/api/demo',headers={'Host':'attacker.invalid'})
        self.assertEqual(response.status,403)
        response=await self.client.post('/api/generate',json={},headers={'Host':'attacker.invalid','Origin':'http://attacker.invalid'})
        self.assertEqual(response.status,403)

    async def test_malformed_long_day_notes_are_rejected(self):
        sources=[{'id':'a','time':'2026-10-24','text':'a'*21000},
                 {'id':'b','time':'2026-10-24','text':'b'*21000}]
        for result in ([],{'notes':None}):
            with patch.object(engine,'day_events',return_value=(sources,0)), \
                 patch.object(engine,'editorial_call',new_callable=AsyncMock,return_value=result):
                with self.assertRaises(ValueError):
                    await engine.analyze(None,{},'2026-10-24','','')

    async def test_saved_edition_and_asset_paths_do_not_escape_cache(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(engine,'CACHE',Path(folder)):
            response=await self.client.get('/api/edition/2026-10-24')
            self.assertEqual(response.status,404)
            Path(folder,'2026-10-24.json').write_text(json.dumps(engine.DEMO))
            response=await self.client.get('/api/edition/2026-10-24')
            self.assertTrue((await response.json())['demo'])
            response=await self.client.get('/assets/.env')
            self.assertEqual(response.status,404)

    async def test_illustration_is_labeled_and_written_inside_cache(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(engine,'CACHE',Path(folder)), \
             patch.object(engine,'gemini',new_callable=AsyncMock) as api:
            api.return_value=[{'inlineData':{'mimeType':'image/png','data':base64.b64encode(b'example-image').decode()}}]
            image=await engine.generate_illustration(None,{'GEMINI_API_KEY':'offline'},'concept only')
            self.assertEqual(image['kind'],'ai')
            self.assertIn('not documentary',image['caption'])
            self.assertEqual(next(Path(folder).iterdir()).read_bytes(),b'example-image')

if __name__=='__main__':
    unittest.main()
