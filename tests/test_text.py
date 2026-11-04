"""Text queries through live.step() with LLMDet tiny (manual 6.5, 7.6)."""

import unittest

import numpy as np

import live
from common import H, LEFT, REF_BOXES, W, clip, detector, iou, tracker


def masks(tr, det, query, n=100):
    """Run live.step() over n frames; the query runs on frame 0. Returns (masks per frame, frame-0 header)."""
    out, heads = [], []
    for i, j in enumerate(clip()[0][:n]):
        head, data = live.step(tr, j, det, query if i == 0 else None)
        heads.append(head)
        out.append(np.frombuffer(data, np.float16).reshape(-1, head["h"], head["w"]) > 0 if data else None)
    return out, heads[0]


class Text(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr, cls.det = tracker(), detector()

    def test_dark_birds_match_box_tracks(self):
        ref = []
        for x0, y0, x1, y1 in REF_BOXES:
            self.tr.set_prompt({"box": [x0 / W, y0 / H, x1 / W, y1 / H]})
            ref.append([m[0] for m in masks(self.tr, None, None)[0]])
        self.tr.set_prompt(None)
        T, head = masks(self.tr, self.det, (["dark bird"], False, 1))
        self.assertEqual([f["label"] for f in head["found"]], ["dark bird", "dark bird"])
        self.assertEqual(head["labels"], ["dark bird", "dark bird"])
        self.assertEqual(head["find_id"], 1)
        for r in ref:  # the detector gives the birds in score order
            self.assertGreater(max(iou([t[k] for t in T], r) for k in range(2)), 0.9)  # SAM2.1: 0.91

    def test_nothing_found_keeps_the_track(self):
        self.tr.set_prompt(LEFT)
        live.step(self.tr, clip()[0][0])
        head, _ = live.step(self.tr, clip()[0][1], self.det, (["giraffe"], False, 2))
        self.assertEqual((head["found"], len(head["areas"])), ([], 1))

    def test_object_mode_adds_and_skips_tracked(self):
        x0, y0, x1, y1 = REF_BOXES[0]
        self.tr.set_prompt({"box": [x0 / W, y0 / H, x1 / W, y1 / H]})  # whole bird: a point is partial on SAM2
        live.step(self.tr, clip()[0][0])
        head, data = live.step(self.tr, clip()[0][1], self.det, (["rocks", "dark bird"], True, 3))
        labels = [f["label"] for f in head["found"]]
        self.assertIn("rocks", labels)
        m = np.frombuffer(data, np.float16).reshape(-1, head["h"], head["w"]) > 0
        self.assertEqual(len(m), 1 + len(labels))
        for k in range(1, len(m)):  # no new object is the tracked bird again
            self.assertLess(iou([m[k]], [m[0]]), live.DUP_IOU, labels[k - 1])

    def test_parse_query(self):
        self.assertEqual(live.parse_query("boy . red cup, ; "), ["boy", "red cup"])
        self.assertEqual(live.parse_query(5), [])
        self.assertEqual(live.parse_query("x" * 201), [])


if __name__ == "__main__":
    unittest.main()
