"""WebSocket protocol of live.py (manual 9), on an in-process server with the test model."""

import asyncio
import json
import unittest

from aiohttp import WSMsgType, WSServerHandshakeError
from aiohttp.test_utils import TestClient, TestServer

import live
from common import MODEL, clip, detector, tracker


class Protocol(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        app = live.make_app({MODEL: tracker()}, MODEL, detector())
        self.client = TestClient(TestServer(app))
        await self.client.start_server()
        self.ws = await self.client.ws_connect("/ws", max_msg_size=0)
        self.hello = json.loads((await self.ws.receive()).data)

    async def asyncTearDown(self):
        await self.ws.close()
        await self.client.close()

    async def send(self, msg):
        await self.ws.send_str(json.dumps(msg))

    async def reply(self):
        return json.loads((await self.ws.receive()).data)

    async def frame(self, i=0, data=None):
        """Send one frame; return its header (the logits message, if any, is checked and dropped)."""
        await self.ws.send_bytes(data or clip()[0][i])
        head = await self.reply()
        if head.get("areas"):
            grids = await self.ws.receive_bytes()
            self.assertEqual(len(grids), 2 * len(head["areas"]) * head["h"] * head["w"])  # float16 per object
        return head

    async def test_hello_and_model_allowlist(self):
        self.assertEqual([m["id"] for m in self.hello["models"]], list(live.MODELS))
        self.assertEqual(self.hello["model"], MODEL)
        await self.send({"model": "evil/repo"})
        self.assertEqual(await self.reply(), {"model": MODEL, "error": "unknown model 'evil/repo'"})
        await self.send({"model": MODEL})
        self.assertEqual(await self.reply(), {"model": MODEL})

    async def test_second_viewer_is_refused(self):
        second = await self.client.ws_connect("/ws")
        self.assertEqual(((await second.receive()).type, second.close_code), (WSMsgType.CLOSE, live.BUSY))
        await self.ws.close()  # the first viewer leaves: the next one gets in, as the page does on retry
        for _ in range(20):
            ws = await self.client.ws_connect("/ws")
            msg = await ws.receive()
            if msg.type == WSMsgType.TEXT:
                break
            await asyncio.sleep(0.05)  # the server has not seen the close yet
        else:
            self.fail("still refused after the first viewer left")
        self.assertIn("models", json.loads(msg.data))
        await ws.close()

    async def test_other_sites_are_refused(self):
        with self.assertRaises(WSServerHandshakeError) as e:
            await self.client.ws_connect("/ws", headers={"Origin": "http://evil.example"})
        self.assertEqual(e.exception.status, 403)

    async def test_bad_messages_are_answered(self):
        for bad in ("not json", "5", '{"x": 0.5}', '{"box": [0, 0, 1]}', '{"model": ["sam2"]}', '{"remove": "1"}',
                    '{"x": 0.2, "y": 0.5, "add": 1, "obj": 0}'):
            await self.ws.send_str(bad)
            self.assertTrue((await self.reply())["message_error"].startswith("bad message"), bad)
        self.assertEqual((await self.frame(0))["areas"], [])  # still connected, nothing tracked

    async def test_tracking_failure_clears_the_track(self):
        await self.send({"x": 0.49, "y": 0.48})
        self.assertEqual(len((await self.frame(0))["areas"]), 1)
        def boom(rgb):
            raise ValueError("boom")

        tracker().step = boom  # an instance attribute hides the method for one frame
        try:
            self.assertIn("track cleared", (await self.frame(1))["error"])
        finally:
            del tracker().step
        self.assertEqual((await self.frame(2))["areas"], [])

    async def test_new_viewer_starts_clean(self):
        await self.send({"x": 0.49, "y": 0.48})
        self.assertEqual(len((await self.frame(0))["areas"]), 1)
        await self.ws.close()
        for _ in range(20):  # the server sees the close a little later
            self.ws = await self.client.ws_connect("/ws", max_msg_size=0)
            if (await self.ws.receive()).type == WSMsgType.TEXT:
                break
            await asyncio.sleep(0.05)
        self.assertEqual((await self.frame(1))["areas"], [])

    async def test_prompt_add_remove_clear(self):
        self.assertEqual((await self.frame(0))["areas"], [])  # no prompt: header only
        await self.send({"x": 0.49, "y": 0.48})
        await self.send({"box": [0.59, 0.32, 0.82, 0.61], "obj": "new"})
        self.assertEqual((await self.frame(1))["ids"], [1, 2])
        await self.send({"remove": 1})
        head = await self.frame(2)
        self.assertEqual((head["ids"], len(head["areas"])), ([2], 1))
        await self.send({})
        self.assertEqual((await self.frame(3))["areas"], [])

    async def test_bad_queries(self):
        for text in (5, "x" * 300, " . "):
            await self.send({"text": text, "id": 9})
            reply = await self.reply()
            self.assertEqual(reply["find_id"], 9)
            self.assertIn("find_error", reply)

    async def test_query_runs_once_with_its_id(self):
        await self.send({"text": "dark bird", "id": 7})
        head = await self.frame(0)
        labels = [f["label"] for f in head["found"]]
        self.assertEqual(head["find_id"], 7)
        self.assertTrue(labels and set(labels) == {"dark bird"}, labels)
        head = await self.frame(1)
        self.assertNotIn("found", head)
        self.assertEqual(head["labels"], labels)

    async def test_clear_click_or_invalid_search_cancel_a_query(self):
        for cancel in ({}, {"x": 0.49, "y": 0.48}, {"text": "", "id": 2}):
            await self.send({"text": "dark bird", "id": 1})
            await self.send(cancel)
            if "text" in cancel:
                await self.reply()  # its find_error
            self.assertNotIn("found", await self.frame(0), cancel)

    async def test_bad_frame_consumes_the_query(self):
        await self.send({"text": "dark bird", "id": 3})
        await self.ws.send_bytes(b"bad")
        err = await self.reply()
        self.assertEqual(err["find_id"], 3)
        self.assertTrue(err["error"].startswith("bad frame"))
        self.assertNotIn("found", await self.frame(0))


if __name__ == "__main__":
    unittest.main()
