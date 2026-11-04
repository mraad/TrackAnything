"""track.py --redetect: objects that enter later get a track, gone objects end, no duplicates (albatross.mp4 with a cover)."""

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from common import ROOT, detector
import live
import track


class OnOld(unittest.TestCase):
    def test_on_old(self):
        m = torch.zeros((3, 100, 100), dtype=torch.bool)
        m[0, 10:30, 10:30] = m[1, 12:28, 12:28] = m[2, 60:70, 60:70] = True
        a = m.flatten(1).sum(1)
        self.assertTrue(track.on_old(m, a, 1, [1, 2]))  # inside object 0
        self.assertFalse(track.on_old(m, a, 2, [1, 2]))
        self.assertFalse(track.on_old(m, a, 1, [1, 2], [0]))  # object 0 is a background mask: it says nothing


class FakeTracker:
    """Object 1 is small, object 2 covers the frame (a sports field): only logic, no model."""

    def __init__(self):
        self.ids, self.removed = [1, 2], []

    def add_boxes(self, found, add):
        pass

    def remove(self, k):
        self.removed.append(k)
        self.ids = [x for x in self.ids if x != k]

    def step(self, rgb):
        grids = {1: torch.full((16, 16), -1.0), 2: torch.ones((16, 16))}
        grids[1][2:4, 2:4] = 1
        return torch.stack([grids[k] for k in self.ids])


class FakeDetector:
    def find(self, rgb, names, box_threshold):
        return []


class TooBig(unittest.TestCase):
    def test_background_mask_ends_and_is_not_drawn(self):
        with tempfile.TemporaryDirectory() as d:
            video = Path(d, "grey.mp4")
            out = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (64, 64))
            for _ in range(3):
                out.write(np.full((64, 64, 3), 128, np.uint8))
            out.release()
            tr = FakeTracker()
            s = track.track_redetect(tr, FakeDetector(), video, ["x"], 15, 0.35, Path(d, "o.mp4"))
            _, first = cv2.VideoCapture(str(Path(d, "o.mp4"))).read()
        self.assertEqual((tr.removed, s["too_big"], s["most"]), ([2], 1, 1))
        self.assertEqual(live.MAX_OBJ, 8)  # the tool's limit does not stay for the live code
        self.assertLess(abs(int(first[40, 40, 0]) - 128), 12)  # outside object 1: the frame, not a colour


class Redetect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        cls.video = cls.dir / "covered.mp4"
        cap = cv2.VideoCapture(str(ROOT / "data/albatross.mp4"))
        out = cv2.VideoWriter(str(cls.video), cv2.VideoWriter_fourcc(*"mp4v"), 30, (960, 540))
        for f in range(120):  # the right bird is covered for the first 30 frames, the full frame from frame 90
            _, img = cap.read()
            if f < 30:
                img[:, 560:] = 0
            if f >= 90:
                img[:] = 0
            out.write(img)
        out.release()
        cap.release()
        cls.tr = live.load("etam-s")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        del cls.tr  # test_tracker and the subprocess tests need the GPU next

    def test_late_object_added_gone_objects_end(self):
        self.tr.set_prompt(None)
        s = track.track_redetect(self.tr, detector(), self.video, ["dark bird"], 15, 0.35, self.dir / "o.mp4")
        self.assertEqual(s["frames"], 120)
        self.assertLessEqual(s["added"] - s["duplicates"] - s["too_big"], 4, s)  # no object added again and again
        self.assertGreaterEqual(s["most"], 2, s)  # the right bird arrives at frame 30
        self.assertEqual(self.tr.ids, [], s)  # everything is covered from frame 90: all tracks ended
        self.assertTrue((self.dir / "o.mp4").exists())


if __name__ == "__main__":
    unittest.main()
