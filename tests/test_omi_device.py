"""Omi notification framing and provenance; no real device or audio recording."""
import struct
import unittest
from unittest import mock
from types import SimpleNamespace
from aiohttp import web
import bridge
from test_bridge import live_state

# Framing module imports Bleak only when a real device is opened.
from omi_device import Reassembler,PacketError

def frame(seq,index,payload=b'x'):
    return struct.pack('<HB',seq,index)+payload

class FramingTests(unittest.TestCase):
    def test_reassembles_fragments_and_flushes_last_frame(self):
        r=Reassembler()
        self.assertIsNone(r.feed(frame(7,0,b'ab')))
        self.assertIsNone(r.feed(frame(8,1,b'cd')))
        self.assertEqual(r.feed(frame(9,0,b'ef')),b'abcd')
        self.assertEqual(r.flush(),b'ef')

    def test_counter_wrap_and_duplicate_are_safe(self):
        r=Reassembler();r.feed(frame(65535,0,b'a'))
        self.assertIsNone(r.feed(frame(65535,0,b'a')))
        self.assertEqual(r.feed(frame(0,0,b'b')),b'a')
        self.assertEqual(r.old,1)

    def test_loss_is_explicit_and_not_concealed(self):
        r=Reassembler();r.feed(frame(1,0))
        with self.assertRaises(PacketError):r.feed(frame(3,0))
        self.assertEqual(r.lost,1)

    def test_fragment_order_and_bounds(self):
        r=Reassembler();r.feed(frame(1,0))
        with self.assertRaises(PacketError):r.feed(frame(2,2))
        r=Reassembler()
        with self.assertRaises(PacketError):r.feed(frame(1,0,b'x'*4097))

class OmiIngressTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_device_mismatch_is_rejected_before_audio(self):
        req=SimpleNamespace(query={},remote='127.0.0.1',headers={'X-Audio-Source':'omi_ble','X-Omi-Device':'00000000-0000-0000-0000-000000000001'})
        with mock.patch.object(bridge,'STREAM_TOKEN',''),mock.patch.object(bridge,'load_env',return_value={'OMI_DEVICE_ID':'00000000-0000-0000-0000-000000000002'}):
            with self.assertRaises(web.HTTPForbidden):await bridge.handle_omi_ws(req)

    async def test_named_device_cannot_replace_any_existing_source(self):
        runtime=live_state();runtime.stream_clients[object()]=object()
        req=SimpleNamespace(query={},remote='127.0.0.1',headers={'X-Audio-Source':'omi_ble','X-Omi-Device':'00000000-0000-0000-0000-000000000001'})
        with mock.patch.object(bridge,'state',runtime),mock.patch.object(bridge,'STREAM_TOKEN',''),mock.patch.object(bridge,'load_env',return_value={}):
            with self.assertRaises(web.HTTPConflict):await bridge.handle_omi_ws(req)
