"""Offline go-back-and-correct: track.py --fix / --fix-box through track() on albatross.mp4 (960 x 540)."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from common import ROOT  # first: it puts the project on sys.path and turns off the progress bars
import track  # noqa: E402

W, H = 960, 540
LEFT = ("point", [470, 260])  # the left dark bird
RIGHT = ("box", [570, 172, 788, 330])  # the right dark bird
WING = ("point", [758, 208])  # the right bird's wing tip


def iou(A, B):
    return float(np.mean([(a & b).sum() / max(1, (a | b).sum()) for a, b in zip(A, B)]))


def area(A):
    return float(np.mean([a.mean() for a in A]))


def covered(masks, ref):
    """Mean share of the object in ref (frames of bool masks) that masks cover."""
    return float(np.mean([(m & r).sum() / max(1, r.sum()) for m, r in zip(masks, ref)]))


def spread(region, n=2):
    """n pixel points spread top to bottom over a bool mask."""
    ys, xs = np.nonzero(region)
    assert len(ys), "empty region"
    o = np.argsort(ys)
    return [[float(xs[k]), float(ys[k])] for k in o[[len(o) * (j + 1) // (n + 1) for j in range(n)]]]


class Fix(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)
        _, cls.n = track.extract_frames(ROOT / "data/albatross.mp4", cls.dir)
        cls.pred = track.load_efficienttam("s")
        cls.left, cls.right = cls.masks([LEFT]), cls.masks([RIGHT])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()
        del cls.pred  # test_offline's subprocesses need the GPU next

    @classmethod
    def masks(cls, prompts, fixes=()):
        """Per object: list of (H, W) bool masks over all frames."""
        masks, _ = track.track(cls.pred, cls.dir, prompts, list(fixes))
        return [[masks[f][k] for f in range(cls.n)] for k in range(len(prompts))]

    def on_right(self, f, label=1, obj=1):
        """Two fixes on the right bird (its box track minus the left bird) at frame f."""
        return [(f, obj, "point", p, label) for p in spread(self.right[0][f] & ~self.left[0][f])]

    def test_add_points_take_the_right_bird(self):
        # two add points on another object: SAM takes the right bird (here it also drops the left bird)
        fixed = self.masks([LEFT], self.on_right(5))[0]
        self.assertGreater(covered(fixed[10:], self.right[0][10:]), 0.8)
        self.assertEqual(iou(fixed[:5], self.left[0][:5]), 1.0)  # frames before the fix keep their masks

    def test_remove_points_take_out_a_part(self):
        # SAM picks one object here, so a mask with two areas does not occur: the mask loses part of the bird and stays
        rm = [(40, 1, "point", p, 0) for p in spread(self.left[0][40])]
        cut = self.masks([LEFT], rm)[0]
        self.assertLess(covered(cut[60:], self.left[0][60:]), 0.9)
        self.assertGreater(covered(cut[60:], self.left[0][60:]), 0.2)
        self.assertEqual(iou(cut[:40], self.left[0][:40]), 1.0)

    def test_fix_on_object_2_leaves_object_1(self):
        base = self.masks([LEFT, WING])
        fixed = self.masks([LEFT, WING], self.on_right(2, obj=2))
        self.assertGreater(iou(fixed[1][10:], self.right[0][10:]), 0.9)
        self.assertGreater(iou(fixed[0], base[0]), 0.95)

    def test_fix_box_moves_the_track(self):
        ys, xs = np.nonzero(self.right[0][30])
        box = [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]  # the right bird on frame 30
        moved = self.masks([LEFT], [(30, 1, "box", box, 1)])[0]
        self.assertGreater(iou(moved[35:], self.right[0][35:]), 0.8)
        self.assertEqual(iou(moved[:30], self.left[0][:30]), 1.0)  # frames before the fix keep their masks

    def test_cli_rejects_bad_fixes(self):
        for args in (["--fix", "5:1:10,10:+", "--model", "sam3"], ["--fix", "5:1:10,10:x"], ["--fix-box", "5:0:1,2,3,4"],
                     ["--fix", "99999:1:10,10:+"], ["--fix", "5:2:10,10:+"]):
            p = subprocess.run([sys.executable, "track.py", "data/albatross.mp4", "--point", "470,260", *args],
                               cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(p.returncode, 2, (args, p.stderr[-300:]))


if __name__ == "__main__":
    unittest.main()
