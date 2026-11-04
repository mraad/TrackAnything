"""Offline tools as the user runs them: track.py regression, several objects, --text, and tools/anonymize.py."""

import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import cv2

from common import ROOT

BLUE, BROWN = "248,617,392,785", "392,624,498,761"  # garbage_cans.mp4 frame-0 boxes


def run(*args, script="track.py"):
    p = subprocess.run([sys.executable, script, *args], cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stdout[-2000:] + p.stderr[-2000:]
    return p.stdout


def frame0(out, prefix="mask area"):
    """frame-0 area in % from a track.py line that starts with prefix."""
    return [float(v) for v in re.findall(rf"^{prefix}.*?frame0 ([\d.]+)%", out, re.M)]


class Offline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.dir = Path(cls.tmp.name)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_regression_albatross_point(self):
        out = run("data/albatross.mp4", "--point", "470,260", "--runs", "1", "--out", str(self.dir / "b.mp4"))
        (a,) = frame0(out)
        self.assertTrue(3.015 <= a <= 3.045, f"frame0 {a}% (expected ~3.03%)")  # compiled encoder: not exact

    def test_two_boxes_match_one_box_each(self):
        cans = ["data/garbage_cans.mp4", "--runs", "1", "--out", str(self.dir / "c.mp4")]
        both = frame0(run(*cans, "--box", BLUE, "--box", BROWN), r"object \d")
        alone = [frame0(run(*cans, "--box", b))[0] for b in (BLUE, BROWN)]
        self.assertEqual(len(both), 2)
        for x, y in zip(both, alone):
            self.assertAlmostEqual(x, y, delta=0.02)

    def test_text_finds_both_cans(self):
        out = run("data/garbage_cans.mp4", "--text", "garbage can . trash can", "--runs", "1",
                  "--out", str(self.dir / "t.mp4"))
        self.assertEqual(len(re.findall(r"^found ", out, re.M)), 2, out)
        self.assertEqual(len(frame0(out, r"object \d")), 2)

    def test_anonymize(self):
        dst = self.dir / "anon.mp4"
        out = run("data/garbage_cans.mp4", "--subject", "garbage can . trash can", "--model", "etam-s",
                  "--out", str(dst), script="tools/anonymize.py")
        self.assertIn("tags left: none", out)
        self.assertGreater(int(re.findall(r"pixelated (\d+) boxes", out)[0]), 0, out)  # the privacy step ran
        self.assertTrue((self.dir / "anon_sheet.jpg").exists())
        cap = cv2.VideoCapture(str(dst))
        self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 321)


if __name__ == "__main__":
    unittest.main()
