"""Tracker behaviour through Prompted + step(): several objects, correction points, removal (manual 14.4)."""

import unittest

from common import BACK, BACK2, LEFT, MODEL, N, RIGHT, WHOLE, WING, area, clip, iou, points, run, tracker

W = slice(10, N)  # frames after the corrections settle


def share(mask, ref):
    """Share of the object in ref (one frame, one object) that mask covers."""
    return float((mask[0] & ref[0]).sum() / max(1, ref[0].sum()))


class MultiObject(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = tracker()
        cls.left, cls.right = run(cls.tr, LEFT), run(cls.tr, RIGHT)

    def test_same_frame_matches_single_tracks(self):
        both = run(self.tr, LEFT, {0: [lambda o: self.tr.add_object(RIGHT)]})
        self.assertEqual(both[-1].shape[0], 2)
        self.assertGreater(iou([b[0] for b in both], [b[0] for b in self.left]), 0.95)
        self.assertGreater(iou([b[1] for b in both], [g[0] for g in self.right]), 0.95)

    def test_late_add(self):
        late = run(self.tr, LEFT, {30: [lambda o: self.tr.add_object(RIGHT)]})
        self.assertEqual((late[29].shape[0], late[30].shape[0]), (1, 2))
        self.assertGreater(iou([b[0] for b in late], [b[0] for b in self.left]), 0.95)
        self.assertGreater(area([b[1] for b in late[40:]]), 0.02)

    def test_fix_on_object_2_leaves_object_1(self):
        add2 = lambda o: self.tr.add_object(WING)  # noqa: E731
        fixed = run(self.tr, LEFT, {0: [add2], 2: [lambda o: [self.tr.correct({**p, "add": 1, "obj": 2}) for p in WHOLE]]})
        self.assertGreater(iou([f[1] for f in fixed[W]], [g[0] for g in self.right[W]]), 0.9)
        self.assertGreater(iou([f[0] for f in fixed[W]], [b[0] for b in self.left[W]]), 0.95)


class Corrections(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = tracker()
        cls.left, cls.right = run(cls.tr, LEFT), run(cls.tr, RIGHT)

    def correct(self, pts):
        return lambda o: [self.tr.correct(p) for p in pts]

    def test_add_points_on_the_other_bird(self):
        if MODEL != "etam-s":  # the others keep the left bird; test_wing_point_plus_fix_gives_the_bird covers their add points
            self.skipTest(f"{MODEL} ignores these add points (manual 5.7)")
        # two add points on the right bird at frame 40: the left bird's track takes it in
        grown = run(self.tr, LEFT, {40: [self.correct(points(self.right[39][0] & ~self.left[39][0], 1))]})
        got = [share(g, r) for g, r in zip(grown[80:], self.right[80:])]
        self.assertGreater(sum(got) / len(got), 0.5)

    def test_remove_points_take_out_a_part(self):
        # SAM picks one object here, so a mask with two areas does not occur: the mask loses part of the bird and stays
        cut = run(self.tr, LEFT, {40: [lambda o: [self.tr.correct(p) for p in points(o[-1][0], 0)]]})
        got = sum(share(c, r) for c, r in zip(cut[60:], self.left[60:])) / len(cut[60:])
        self.assertLess(got, 0.9)
        self.assertGreater(got, 0.02)  # 0.40 EfficientTAM-S, 0.64 Ti, 0.06 SAM2.1
        self.assertEqual(iou([c[0] for c in cut[:40]], [r[0] for r in self.left[:40]]), 1.0)

    def test_wing_point_plus_fix_gives_the_bird(self):
        fixed = run(self.tr, WING, {2: [self.correct([{**p, "add": 1} for p in WHOLE])]})
        self.assertGreater(iou([f[0] for f in fixed[W]], [g[0] for g in self.right[W]]), 0.9)

    def test_no_track(self):
        self.tr.set_prompt(None)
        self.tr.correct({"x": 0.5, "y": 0.5, "add": 0})
        self.assertIsNone(self.tr.step(clip()[1][0]))  # a remove point alone starts nothing
        self.tr.correct({**BACK, "add": 1})
        self.assertEqual(self.tr.step(clip()[1][0]).shape[0], 1)  # an add point starts a track


class Removal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tr = tracker()
        cls.base = cls.three({}, 60)

    @classmethod
    def three(cls, events, n=60):
        ids = []
        def setup(o):
            cls.tr.add_object(RIGHT)
            cls.tr.add_object(BACK)
        ev = {0: [setup], **events}
        out = run(cls.tr, LEFT, {i: ev.get(i, []) + [lambda o: ids.append(list(cls.tr.ids))] for i in range(n)}, n)
        return out, ids

    def test_others_unchanged_and_ids_stable(self):
        (rm, ids), (base, _) = self.three({20: [lambda o: self.tr.remove(2)]}), self.base
        self.assertEqual((ids[20], ids[21]), ([1, 2, 3], [1, 3]))
        self.assertEqual(rm[21].shape[0], 2)
        w = range(21, 60)
        self.assertGreater(iou([rm[i][0] for i in w], [base[i][0] for i in w]), 0.99)
        self.assertGreater(iou([rm[i][1] for i in w], [base[i][2] for i in w]), 0.99)

    def test_fix_after_removal_hits_the_right_object(self):
        rm, _ = self.three({20: [lambda o: self.tr.remove(2)]})
        fx, _ = self.three({20: [lambda o: self.tr.remove(2)],
                            30: [lambda o: self.tr.correct({**BACK2, "add": 1, "obj": 3})]})
        w = range(31, 60)
        self.assertGreater(iou([fx[i][0] for i in w], [rm[i][0] for i in w]), 0.95)  # left bird untouched
        self.assertLess(iou([fx[i][1] for i in w], [rm[i][1] for i in w]), 0.5)  # object 3 changed (the add moves the track)

    def test_pending_last_and_readd(self):
        tr, fr = self.tr, clip()[1]
        tr.set_prompt(LEFT); tr.step(fr[0]); tr.add_object(RIGHT); tr.remove(2)
        self.assertEqual((tr.ids, tr.step(fr[1]).shape[0]), ([1], 1))  # removed before its first frame
        tr.remove(1)
        self.assertIsNone(tr.step(fr[2]))  # last object gone: no track
        self.assertEqual(tr.ids, [])
        tr.set_prompt(LEFT); tr.step(fr[0]); tr.add_object(RIGHT); tr.step(fr[1]); tr.remove(1); tr.step(fr[2])
        tr.add_object(BACK)
        lg = tr.step(fr[3])
        self.assertEqual(tr.ids, [2, 3])  # ids are never reused
        self.assertTrue(all((g > 0).any() for g in lg))


if __name__ == "__main__":
    unittest.main()
