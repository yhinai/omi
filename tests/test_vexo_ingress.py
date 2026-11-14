"""Source identity must follow immutable sessions; ownership must be exclusive."""
import asyncio
from types import SimpleNamespace
import unittest
from unittest import mock
from aiohttp import web
import bridge
from test_bridge import live_state
from test_session_safety import PhoneSocket

class VexoIngressTests(unittest.IsolatedAsyncioTestCase):
    async def test_vexo_cannot_replace_any_active_source_even_same_remote(self):
        runtime=live_state()
        runtime.stream_clients[object()]=object()
        runtime.metrics['direct_ws_remote']='127.0.0.1'
        req=SimpleNamespace(headers={'X-Audio-Source':'vexo','X-Vexo-Band':'43'}, query={},remote='127.0.0.1')
        with mock.patch.object(bridge,'state',runtime),mock.patch.object(bridge,'STREAM_TOKEN',''):
            with self.assertRaises(web.HTTPConflict):
                await bridge.handle_omi_ws(req)

    async def test_omi_cannot_replace_vexo_even_same_remote(self):
        runtime=live_state()
        runtime.stream_clients[object()]=object()
        runtime.metrics.update(direct_ws_remote='127.0.0.1',audio_source='vexo')
        req=SimpleNamespace(headers={},query={},remote='127.0.0.1')
        with mock.patch.object(bridge,'state',runtime),mock.patch.object(bridge,'STREAM_TOKEN',''):
            with self.assertRaises(web.HTTPConflict):
                await bridge.handle_omi_ws(req)

    def test_late_result_keeps_original_identity(self):
        runtime=live_state()
        runtime.source_sessions[1]={'kind':'vexo','band':43}
        runtime.source_sessions[2]={'kind':'omi'}
        runtime.metrics['source_identity']={'kind':'omi'}
        with mock.patch.object(bridge,'state',runtime):
            event=bridge.make_event('transcript','synthetic',session_id=1)
            runtime.source_sessions[1]['band']=99
            self.assertEqual(event['source'],{'kind':'vexo','band':43})

    async def test_bluetooth_identity_is_bound_to_received_audio_session(self):
        import aiohttp
        runtime=live_state()
        socket=PhoneSocket()
        req=SimpleNamespace(headers={'X-Audio-Source':'vexo','X-Vexo-Band':'43','X-Vexo-Transport':'ble'},query={},remote='127.0.0.1')
        with mock.patch.object(bridge,'state',runtime),mock.patch.object(bridge,'STREAM_TOKEN',''),mock.patch.object(bridge.web,'WebSocketResponse',return_value=socket):
            task=asyncio.create_task(bridge.handle_omi_ws(req))
            await socket.prepared.wait()
            socket.frames.put_nowait(SimpleNamespace(type=aiohttp.WSMsgType.BINARY,data=bytes(640)))
            socket.frames.put_nowait(None)
            await asyncio.wait_for(task,3)
            event=bridge.make_event('transcript','synthetic',session_id=runtime.live_stt.session_id)
            self.assertEqual(event['source']['transport'],'ble')
            self.assertEqual(event['source']['band'],43)

    async def test_invalid_vexo_transport_is_rejected_before_upgrade(self):
        req=SimpleNamespace(headers={'X-Audio-Source':'vexo','X-Vexo-Band':'43','X-Vexo-Transport':'wifi'},query={},remote='127.0.0.1')
        with mock.patch.object(bridge,'STREAM_TOKEN',''):
            with self.assertRaises(web.HTTPBadRequest):
                await bridge.handle_omi_ws(req)
