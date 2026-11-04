"""Live tracking server: browser streams camera JPEGs over a WebSocket, gets back a mask overlay.

uv run live.py [--model sam2|sam3|etam-s|etam-ti|<hf id>] [--port 8765]
Bind is 127.0.0.1: reach it through `ssh -L` (see live_spark2.sh); browsers allow the camera only on localhost/https.
"""

import os

os.environ.setdefault("TQDM_DISABLE", "1")  # EfficientTAM opens a progress bar per propagate call, i.e. per frame

import argparse
import asyncio
import inspect
import json
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
import torch.nn.functional as F
from aiohttp import WSMsgType, web
from torchvision.io import decode_jpeg, encode_jpeg
from torchvision.ops import box_iou, masks_to_boxes

from detector import DETECTOR, Detector, split_names

MODELS = {  # id -> (label, hint) for the page's picker; hints are live round-trip FPS measured on spark2
    "etam-ti": ("EfficientTAM-Ti", "~95 FPS"),
    "etam-s": ("EfficientTAM-S", "~80 FPS"),
    "sam2": ("SAM2.1 small", "~19 FPS"),
    "sam3": ("SAM3", "~3 FPS"),
}
HF = {"sam2": "facebook/sam2.1-hiera-small", "sam3": "facebook/sam3"}  # sam3 is gated: needs HF_TOKEN
MAX_OBJ = 8  # objects per track; the page has 8 colours
# DUP_IOU: in Object mode a found box that overlaps a tracked object is that object again (the girl came back as
# "boy" at 0.55). The detector and its thresholds are in detector.py.
DUP_IOU = 0.5
BUSY = 4001  # WebSocket close code: another viewer has the tracker
KEEP = 16  # frames of history kept; SAM2/3 attend to 6 past masks + up to 16 object pointers

def decode(jpeg):
    """JPEG bytes -> uint8 (3, H, W) RGB on the GPU (nvJPEG)."""
    return decode_jpeg(torch.frombuffer(bytearray(jpeg), dtype=torch.uint8), device="cuda")


def pack(logits) -> tuple[dict, bytes]:
    """Low-res mask logits (n, h, w), one grid per object -> header + float16 bytes (uint8 moved edges: IoU 0.91)."""
    _, h, w = logits.shape
    areas = (logits > 0).flatten(1).float().mean(1).tolist()
    return {"areas": areas, "w": w, "h": h}, logits.half().cpu().numpy().tobytes()


def prune(frames, outputs_per_obj, tracked_per_obj, old):
    """Streaming state keeps every frame and output; drop what memory attention no longer reads."""
    for d in (frames, *(o["non_cond_frame_outputs"] for o in outputs_per_obj), *tracked_per_obj):
        for k in [k for k in d if k <= old]:
            del d[k]


class Prompted:
    """Objects, prompts and correction points of one track. The event loop writes them; step() reads them in the
    worker. Each object keeps its id for the whole track (the page colours by id), also when another object is
    removed. `ids` is in upstream order: the objects in the session first (upstream index = position), then the
    ones that wait for the next frame."""

    def set_prompt(self, prompt, label=""):
        """New track with one object (or none): the session starts on the next frame."""
        self.session, self.t, self.fixes, self.last, self.removals = None, 0, [], None, []  # last: logits of step()
        self.pending = [(1, prompt)] if prompt else []  # (object id, prompt) to add on the next frame
        self.ids, self.next_id = ([1], 2) if prompt else ([], 1)
        self.labels = [label] if prompt else []  # detector label of each object ('' for clicks and boxes)

    @property
    def count(self):
        return len(self.ids)

    def add_object(self, prompt, label=""):
        """One more object, prompted on the next frame; the other objects keep tracking."""
        if not self.ids:
            self.set_prompt(prompt, label)
        elif len(self.ids) < MAX_OBJ:
            k, self.next_id = self.next_id, self.next_id + 1
            self.ids.append(k)
            self.pending.append((k, prompt))
            self.labels.append(label)

    def remove(self, obj):
        """{"remove": k}: stop tracking object k. The worker applies it before the next frame."""
        if obj in self.ids:
            self.removals.append(obj)

    def apply_removals(self, drop):
        """Worker thread. drop(i, k) deletes object k at upstream index i from the session."""
        gone, self.removals = self.removals, []
        for k in gone:
            if k not in self.ids:
                continue
            i = self.ids.index(k)
            if any(pk == k for pk, _ in self.pending):  # not in the session yet
                self.pending = [(pk, pp) for pk, pp in self.pending if pk != k]
            elif self.session is not None:
                drop(i, k)
            del self.ids[i], self.labels[i]
            self.last = None  # its rows no longer match the objects
        if gone and not self.ids:
            self.set_prompt(None)

    def add_boxes(self, found, add):
        """Detector boxes -> objects on this frame: a new track, or (add) more objects. Returns the boxes kept."""
        if add and found and self.last is not None and len(taken := mask_boxes(self.last)):
            # a box on a tracked object is that object again (the girl came back as "boy" at 0.55)
            overlap = box_iou(torch.tensor([f[2] for f in found]), taken).amax(1).tolist()
            found = [f for f, o in zip(found, overlap) if o < DUP_IOU]
        found = found[:MAX_OBJ - self.count if add else MAX_OBJ]
        for i, (label, _, box) in enumerate(found):  # nothing found leaves the track as it was
            self.add_object({"box": box}, label) if add or i else self.set_prompt({"box": box}, label)
        return found

    def correct(self, p):
        """{"x","y","add":1|0[,"obj":k]}: positive or negative point on object k (default 1)."""
        label, obj = int(bool(p["add"])), p.get("obj", 1)
        if not self.ids:  # nothing tracked: a positive point starts a track, a negative one means nothing
            if label:
                self.set_prompt({"x": p["x"], "y": p["y"]})
        elif obj in self.ids:
            self.fixes.append((obj, p["x"], p["y"], label))

    def fix(self, per_obj, run):
        """Worker thread. Apply pending correction points to the last tracked frame f = t-1. per_obj(i) gives object
        i's (outputs, frames tracked), or None while it is still pending; run(i, f, points, logits) re-runs f.

        On a frame that was tracked, memory attention overrides the clicks: SAM2.1 kept a 1.8% mask with two
        positive points on the rest of the object. So f runs again as a new conditioning frame, without memory,
        with its current logits as a dense prompt: the interactive image refinement SAM is trained for. Then the
        frames before f are forgotten, or the older, uncorrected memories out-vote the fix within a few frames
        (upstream's clear_non_cond_mem_around_input).
        """
        # shortcut: conditioning frames are never pruned and all are attended; cap them if users correct a lot.
        pending, self.fixes, f = self.fixes, [], self.t - 1
        for k in {q[0] for q in pending}:
            if k not in self.ids or not (ot := per_obj(i := self.ids.index(k))):
                continue  # removed, or added after this frame: nothing to correct
            outputs, tracked = ot
            fixes, prev = as_cond_frame(outputs, tracked, f, [q[1:] for q in pending if q[0] == k])
            if fixes is None:
                continue  # empty mask: nothing to remove from
            run(i, f, fixes, prev.clamp(-32, 32))
            outputs["non_cond_frame_outputs"].pop(f, None)  # f is a conditioning frame now

    def take_new(self):
        """Worker thread: objects to add on this frame."""
        new, self.pending = self.pending, []
        return new


def drop_index(per_obj, i):
    """Delete key i from per-object dicts keyed 0..n-1 and shift the later keys down, as upstream remove_object does.
    (EfficientTAM's own remove_object also re-runs prompt frames, whose images prune() has already freed.)"""
    for d in per_obj:
        items = [d.pop(k) for k in sorted(d)]
        del items[i]
        d.update(enumerate(items))


def as_cond_frame(outputs, tracked, f, points):
    """Turn frame f of one object into a new conditioning frame for correction points (x, y, label) in 0..1; shared by
    Prompted.fix and track.py --fix. Returns (points to run, f's current output logits), or (None, logits) when the
    points only remove and the mask is empty (nothing to remove: the state stays as it was). Remove-only points get an
    anchor() keep point (SAM drops the whole mask otherwise). f leaves the tracked frames (no memory attention there)
    and the object's older non-conditioning memory goes (it out-votes the fix within a few frames). Frame f's own
    output stays: upstream EfficientTAM reads it as the dense prompt."""
    prev = (outputs["cond_frame_outputs"].get(f) or outputs["non_cond_frame_outputs"][f])["pred_masks"]
    if points and not any(label for *_, label in points):
        if not (keep := anchor(prev[0, 0] > 0, points)):
            return None, prev
        points = points + keep
    tracked.pop(f, None)
    non = outputs["non_cond_frame_outputs"]
    for t in [t for t in non if t < f]:
        del non[t]
    return points, prev


def anchor(mask, negatives):
    """[positive point] at the mask pixel farthest from the negative points, or [] for an empty mask."""
    # shortcut: the guess can land on the part the user removes when that part is the larger one; if that bites,
    # let Remove carry an explicit keep point instead.
    ys, xs = torch.nonzero(mask, as_tuple=True)
    if not len(ys):
        return []
    h, w = mask.shape
    px, py = (xs + 0.5) / w, (ys + 0.5) / h
    k = torch.stack([(px - x) ** 2 + (py - y) ** 2 for x, y, _ in negatives]).min(0).values.argmax()
    return [(float(px[k]), float(py[k]), 1)]


class Tracker(Prompted):
    def __init__(self, model_id):
        if "sam3" in model_id:
            from transformers import Sam3TrackerVideoModel as M, Sam3TrackerVideoProcessor as P
        else:
            from transformers import Sam2VideoModel as M, Sam2VideoProcessor as P
        self.model = M.from_pretrained(model_id, dtype=torch.bfloat16).to("cuda").eval()
        self.processor = P.from_pretrained(model_id)
        # shortcut: _fix wraps a private HF method (transformers 5.x); fail at load, not silently, if it changes.
        assert "prev_sam_mask_logits" in inspect.signature(self.model._run_single_frame_inference).parameters
        # _drop rewrites these private session fields; check that they still exist under these names.
        probe = self.processor.init_video_session(inference_device="cuda", dtype=torch.bfloat16)
        assert all(hasattr(probe, n) for n in ("obj_ids", "_obj_id_to_idx", "_obj_idx_to_id", "obj_with_new_inputs",
                                               "point_inputs_per_obj", "mask_inputs_per_obj", "output_dict_per_obj",
                                               "frames_tracked_per_obj")), "transformers session layout changed"
        self.set_prompt(None)

    def _fix(self, size, i, f, fixes, prev):
        """Prompted.fix callback: run frame f again with object i's points and its current logits."""
        h, w = size
        self.processor.add_inputs_to_inference_session(
            inference_session=self.session, frame_idx=f, obj_ids=self.ids[i], original_size=size, clear_old_inputs=False,
            input_points=[[[[x * w, y * h] for x, y, _ in fixes]]], input_labels=[[[label for *_, label in fixes]]])
        # HF forward() has no argument for the dense prompt that upstream SAM2 passes here (prev_sam_mask_logits).
        # It also re-runs frame f for the other objects, which gives them the same output again.
        run = self.model._run_single_frame_inference
        self.model._run_single_frame_inference = lambda **kw: run(
            **kw, prev_sam_mask_logits=prev if kw["obj_idx"] == i else None)
        try:
            self.model(inference_session=self.session, frame_idx=f)
        finally:
            del self.model._run_single_frame_inference

    def _drop(self, i, k):
        s = self.session
        drop_index((s.point_inputs_per_obj, s.mask_inputs_per_obj, s.output_dict_per_obj, s.frames_tracked_per_obj), i)
        s.obj_ids = [x for x in s.obj_ids if x != k]
        s._obj_id_to_idx = OrderedDict((x, j) for j, x in enumerate(s.obj_ids))
        s._obj_idx_to_id = OrderedDict(enumerate(s.obj_ids))
        s.obj_with_new_inputs = [x for x in s.obj_with_new_inputs if x != k]

    @torch.inference_mode()
    def step(self, rgb):
        """Track on one (3, H, W) uint8 CUDA frame. Returns low-res mask logits (n_objects, h, w), or None."""
        if not self.count:
            return None
        inputs = self.processor(images=rgb, device="cuda", return_tensors="pt")
        h, w = size = list(rgb.shape[1:])
        if self.removals:
            self.apply_removals(self._drop)
            if not self.ids:
                return None
        if self.fixes and self.session is not None:
            s = self.session
            self.fix(lambda i: (s.output_dict_per_obj[i], s.frames_tracked_per_obj[i]) if i in s.output_dict_per_obj
                     else None, lambda *fx: self._fix(size, *fx))
        new = self.take_new()
        if self.session is None:
            self.session = self.processor.init_video_session(inference_device="cuda", dtype=torch.bfloat16)
        for k, p in new:  # this frame becomes object k's prompt frame
            kw = ({"input_boxes": [[[p["box"][0] * w, p["box"][1] * h, p["box"][2] * w, p["box"][3] * h]]]}
                  if "box" in p else {"input_points": [[[[p["x"] * w, p["y"] * h]]]], "input_labels": [[[1]]]})
            self.processor.add_inputs_to_inference_session(
                inference_session=self.session, frame_idx=self.t, obj_ids=k, original_size=size, **kw)
        if new:  # each add_inputs call replaces this list, so same-frame objects but the last would be "tracked"
            self.session.obj_with_new_inputs = [k for k, _ in new]
        out = self.model(inference_session=self.session, frame=inputs.pixel_values[0].to(torch.bfloat16),
                         frame_idx=self.t)
        s = self.session
        prune(s.processed_frames, s.output_dict_per_obj.values(), s.frames_tracked_per_obj.values(), self.t - KEEP)
        self.t += 1
        self.last = out.pred_masks[:, 0]
        return self.last  # the page upsamples to frame size (post_process_masks does the same bilinear)


class EfficientTAMTracker(Prompted):
    """Streams frames into the upstream offline video predictor: frames go into a dict that grows and is pruned."""

    def __init__(self, size):
        from track import load_efficienttam

        self.predictor = load_efficienttam(size)
        self.mean = torch.tensor([0.485, 0.456, 0.406], device="cuda")[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], device="cuda")[:, None, None]
        self.set_prompt({"x": 0.5, "y": 0.5})  # warm up torch.compile so the first click doesn't stall
        for _ in range(3):
            self.step(torch.zeros((3, 480, 640), dtype=torch.uint8, device="cuda"))
        self.set_prompt(None)

    @torch.inference_mode()
    def step(self, rgb):
        """Same as Tracker.step."""
        if not self.count:
            return None
        with torch.autocast("cuda", dtype=torch.bfloat16):  # autocast is thread-local and step runs in a worker
            return self._step(rgb)

    def _fix(self, w, h, i, f, fixes, _prev):
        """Prompted.fix callback; upstream reads frame f's logits for the dense prompt itself."""
        p = self.predictor
        p.add_new_points_or_box(self.session, frame_idx=f, obj_id=self.ids[i], clear_old_points=False,
                                points=(np.array([q[:2] for q in fixes]) * [w, h]).astype(np.float32),
                                labels=np.array([q[2] for q in fixes], np.int32))
        p.propagate_in_video_preflight(self.session)  # moves the fix from temp to the output dict

    def _drop(self, i, k):
        s = self.session
        drop_index([s[n] for n in ("point_inputs_per_obj", "mask_inputs_per_obj", "output_dict_per_obj",
                                   "temp_output_dict_per_obj", "frames_tracked_per_obj")], i)
        s["obj_ids"] = [x for x in s["obj_ids"] if x != k]
        s["obj_id_to_idx"] = {x: j for j, x in enumerate(s["obj_ids"])}
        s["obj_idx_to_id"] = dict(enumerate(s["obj_ids"]))

    def _step(self, rgb):
        p, (h, w) = self.predictor, rgb.shape[1:]
        if self.removals:
            self.apply_removals(self._drop)
            if not self.ids:
                return None
        # Upstream resizes with PIL bicubic, antialiased; torch's antialiased bicubic is the closest GPU equivalent.
        img = F.interpolate(rgb[None].float() / 255, (p.image_size, p.image_size), mode="bicubic", antialias=True)[0]
        img = (img - self.mean) / self.std
        if self.fixes and self.session is not None:
            s = self.session
            self.fix(lambda i: (s["output_dict_per_obj"][i], s["frames_tracked_per_obj"][i])
                     if i in s["output_dict_per_obj"] else None, lambda *fx: self._fix(w, h, *fx))
        new = self.take_new()
        if self.session is None:
            with tempfile.TemporaryDirectory() as d:  # init_state only accepts a path; one JPEG per new prompt
                # re-encode the decoded frame: the received bytes may be truncated, which nvJPEG tolerates and PIL not
                Path(d, "00000.jpg").write_bytes(encode_jpeg(rgb).cpu().numpy().tobytes())
                self.session = p.init_state(d)
            self.session["images"] = {}
        s = self.session
        s["images"][self.t], s["num_frames"] = img, self.t + 1
        for k, pr in new:  # this frame becomes object k's prompt frame
            kw = ({"box": (np.array(pr["box"]) * [w, h, w, h]).astype(np.float32)} if "box" in pr else
                  {"points": np.array([[pr["x"] * w, pr["y"] * h]], np.float32), "labels": np.array([1], np.int32)})
            p.add_new_points_or_box(s, frame_idx=self.t, obj_id=k, **kw)
        # num_frames = t+1 with max_frame_num_to_track=0 makes propagate run exactly frame t (end = min(t, n-1)).
        next(p.propagate_in_video(s, start_frame_idx=self.t, max_frame_num_to_track=0))
        outs = [s["output_dict_per_obj"][i] for i in range(len(s["output_dict_per_obj"]))]
        low = torch.cat([(o["cond_frame_outputs"].get(self.t) or o["non_cond_frame_outputs"][self.t])["pred_masks"]
                         for o in outs])
        prune(s["images"], s["output_dict_per_obj"].values(), s["frames_tracked_per_obj"].values(), self.t - KEEP)
        self.t += 1
        self.last = low[:, 0]
        return self.last  # low-res logits; the yielded full-res mask is the same bilinear upsample the page does


def mask_boxes(logits):
    """(n, h, w) logits -> (k, 4) CPU boxes [x0, y0, x1, y1] in 0..1 of the k non-empty masks."""
    _, h, w = logits.shape
    m = logits > 0
    m = m[m.flatten(1).any(1)]
    return (masks_to_boxes(m) / torch.tensor([w, h, w, h], device=m.device)).cpu()


def valid(p):
    """Page message fields have the types the handler and the tracker read (trust boundary)."""
    num = lambda v: isinstance(v, (int, float)) and not isinstance(v, bool)  # noqa: E731
    pos = lambda v: isinstance(v, int) and not isinstance(v, bool) and v >= 1  # noqa: E731
    obj = p.get("obj", 1)
    return all([
        "x" not in p and "add" not in p or num(p.get("x")) and num(p.get("y")),
        "box" not in p or isinstance(p["box"], list) and len(p["box"]) == 4 and all(map(num, p["box"])),
        "model" not in p or isinstance(p["model"], str),
        "remove" not in p or pos(p["remove"]),
        obj == "new" or pos(obj),
    ])


def parse_query(q):
    """'boy . red cup' -> ['boy', 'red cup']. [] for anything else: the page sends free text."""
    if not isinstance(q, str) or len(q) > 200:
        return []
    return split_names(q)[:2 * MAX_OBJ]


def load(key):
    return EfficientTAMTracker(key[5:]) if key in ("etam-s", "etam-ti") else Tracker(HF.get(key, key))


async def switch(state, key):
    """Make `key` the active model, loading it on first use. Returns the reply for the page."""
    if key not in MODELS:  # only listed ids: the page must not make the server download arbitrary repos
        return {"model": state["model"], "error": f"unknown model {key!r}"}
    trackers = state["trackers"]
    if key not in trackers:
        print(f"loading {key}", flush=True)
        try:  # shortcut: loaded models stay resident (all four fit in GB10's 128 GB); evict if memory gets tight
            trackers[key] = await asyncio.to_thread(load, key)
        except Exception as e:  # gated repo without HF_TOKEN, download or OOM failure: keep the current model
            return {"model": state["model"], "error": f"cannot load {MODELS[key][0]}: {e}"}
    trackers[state["model"]].set_prompt(None)
    state["model"] = key
    return {"model": key}


class BadFrame(Exception):
    """nvJPEG could not decode the frame; the track is fine."""


def step(tracker, jpeg, detector=None, query=None):
    """Worker thread: decode, find (once per text query), track, pack."""
    try:
        rgb = decode(jpeg)
    except RuntimeError as e:
        raise BadFrame(e) from e
    found, error, find_id = None, None, None
    if query and detector:  # every kept box becomes an object prompted on this frame
        labels, add, find_id = query
        try:
            found = tracker.add_boxes(detector.find(rgb, labels), add)
        except Exception as e:  # e.g. OOM: report it once and keep tracking (the caller consumed the query)
            found, error = [], f"search failed: {e}"
    logits = tracker.step(rgb)
    head: dict
    head, data = ({"areas": []}, b"") if logits is None else pack(logits)
    head["labels"], head["ids"] = tracker.labels[:len(head["areas"])], tracker.ids[:len(head["areas"])]
    if found is not None:
        head["found"] = [{"label": label, "score": score} for label, score, _ in found]
        head["find_id"] = find_id  # the page drops a reply for a search that it has replaced
    if error:
        head["find_error"] = error
    return head, data


async def ws_handler(request):
    state = request.app["state"]
    origin = request.headers.get("Origin")  # browsers let any site open a WebSocket to localhost; scripts send none
    if origin and urlparse(origin).hostname not in ("localhost", "127.0.0.1"):
        raise web.HTTPForbidden(text="the live app accepts its own page only")
    ws = web.WebSocketResponse(max_msg_size=8 * 2**20)
    await ws.prepare(request)
    if state["viewer"]:  # one shared tracker: a second viewer would drive it too, so refuse it (the page retries)
        print("second viewer refused", flush=True)
        await ws.close(code=BUSY, message=b"another viewer is open")
        return ws
    state["viewer"] = True
    state["trackers"][state["model"]].set_prompt(None)  # a reloaded page starts clean, not on the last viewer's track
    print("viewer connected", flush=True)
    try:
        return await serve(ws, state)
    finally:
        state["viewer"] = False
        print("viewer disconnected", flush=True)


async def serve(ws, state):
    await ws.send_str(json.dumps({"models": [{"id": k, "label": l, "hint": h} for k, (l, h) in MODELS.items()],
                                  "model": state["model"]}))
    # shortcut: one shared tracker, so one viewer at a time; per-connection trackers if several people need it
    query = None  # (labels, add) from a text message, run on the next frame
    async for msg in ws:
        # text: {"model": id}; {"x","y"} point, {"box":[x0,y0,x1,y1]} (0..1) or {"text": "boy . cup"} starts a track,
        # with "obj":"new" it adds objects; {"x","y","add":1|0,"obj":k} corrects object k; {"remove":k}; {} clears
        if msg.type == WSMsgType.TEXT:
            try:
                p, tracker = json.loads(msg.data), state["trackers"][state["model"]]
                if not isinstance(p, dict) or not valid(p):
                    raise ValueError(msg.data[:100])
            except ValueError as e:  # trust boundary: a bad message is answered, it does not end the connection
                await ws.send_str(json.dumps({"message_error": f"bad message: {e}"}))  # not a frame reply
                continue
            if "text" not in p and "add" not in p and "remove" not in p:
                query = None  # a model switch, new prompt or clear supersedes a query that has not run yet
            if "model" in p:
                await ws.send_str(json.dumps(await switch(state, p["model"])))
            elif "add" in p:
                tracker.correct(p)
            elif "remove" in p:
                tracker.remove(p["remove"])
            elif "text" in p:
                query = None  # a new search replaces a queued one, also when it is not valid
                if not (labels := parse_query(p["text"])):
                    await ws.send_str(json.dumps({"find_error": "type one or more names, for example: cup . phone",
                                                  "find_id": p.get("id")}))
                    continue
                if state["detector"] is None:  # first query: load in a worker, like a model switch
                    print(f"loading {DETECTOR}", flush=True)
                    try:
                        state["detector"] = await asyncio.to_thread(Detector)
                    except Exception as e:  # download failure or OOM: keep tracking, report it
                        await ws.send_str(json.dumps({"find_error": f"cannot load the detector: {e}",
                                                      "find_id": p.get("id")}))
                        continue
                query = (labels, p.get("obj") == "new", p.get("id"))  # runs on the next frame
            elif p.pop("obj", None) == "new":
                tracker.add_object(p)
            else:
                tracker.set_prompt(p if p else None)
        elif msg.type == WSMsgType.BINARY:
            t0, q, query = time.perf_counter(), query, None  # a query runs on one frame only, even if it fails
            try:
                head, data = await asyncio.to_thread(step, state["trackers"][state["model"]], msg.data,
                                                     state["detector"], q)
            except BadFrame as e:  # nvJPEG rejects corrupt data; the search on this frame (if any) is lost
                await ws.send_str(json.dumps({"error": f"bad frame: {e}", "find_id": q[2] if q else None}))
                continue
            except Exception as e:  # tracking failed: the state may be half-changed, so start clean, not fail every frame
                state["trackers"][state["model"]].set_prompt(None)
                await ws.send_str(json.dumps({"error": f"tracking failed, track cleared: {e}",
                                              "find_id": q[2] if q else None}))
                continue
            await ws.send_str(json.dumps({"ms": round((time.perf_counter() - t0) * 1000, 1), **head}))
            if data:  # raw float16 (h, w) logits: no PNG on either side, the page's shader scales and colours it
                await ws.send_bytes(data)
    return ws


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sam2", help="model loaded at start (others load when picked in the page): "
                    "sam2, sam3, etam-s, etam-ti, or a Hugging Face SAM2 id")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    MODELS.setdefault(a.model, (a.model, ""))  # a custom HF id becomes pickable too
    app = make_app({a.model: load(a.model)}, a.model)
    print(f"model {MODELS[a.model][0]} on {torch.cuda.get_device_name(0)}")
    web.run_app(app, host="127.0.0.1", port=a.port)


def make_app(trackers, model, detector=None):
    """The page and the WebSocket around already loaded trackers (tests pass their own)."""
    app = web.Application()
    # mutable dict: aiohttp freezes app keys after startup
    app["state"] = {"model": model, "trackers": trackers, "detector": detector, "viewer": False}
    page = Path(__file__).with_name("live.html")
    app.router.add_get("/", lambda _: web.FileResponse(page))
    app.router.add_get("/ws", ws_handler)
    return app


if __name__ == "__main__":
    main()
