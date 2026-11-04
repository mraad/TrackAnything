"""Shared clip, prompts and model cache for the GPU tests (run on spark2/spark3: ./test_spark2.sh)."""

import os

os.environ.setdefault("TQDM_DISABLE", "1")

import sys
from functools import cache
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import live  # noqa: E402

MODEL = os.environ.get("TEST_MODEL", "etam-s")  # etam-s | etam-ti | sam2 | sam3
N, W, H = 120, 640, 360  # frames used; albatross.mp4 at this size
LEFT = {"x": 0.49, "y": 0.48}  # a point on the left dark bird
RIGHT = {"box": [0.59, 0.32, 0.82, 0.61]}  # the right dark bird
WING = {"x": 0.79, "y": 0.385}  # the right bird's wing tip: part of it on SAM2.1 (46%), nearly all on EfficientTAM-S
WHOLE = [{"x": 0.65, "y": 0.46}, {"x": 0.63, "y": 0.35}]  # body and head of the right bird, frame 0
BACK, BACK2 = {"x": 0.22, "y": 0.40}, {"x": 0.32, "y": 0.33}  # two small birds behind them
REF_BOXES = [[227, 105, 392, 236], [381, 118, 524, 218]]  # left, right: frame 0, pixels, from EfficientTAM-S tracks


@cache
def clip():
    """(JPEG bytes, GPU RGB tensors) of the first N frames."""
    cap, jpegs = cv2.VideoCapture(str(ROOT / "data/albatross.mp4")), []
    while len(jpegs) < N and (r := cap.read())[0]:
        jpegs.append(cv2.imencode(".jpg", cv2.resize(r[1], (W, H)), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())
    return jpegs, [live.decode(j) for j in jpegs]


@cache
def tracker():
    return live.load(MODEL)


@cache
def detector():
    return live.Detector()


def run(tr, first, events=None, n=N):
    """Track from prompt `first`; events {frame: [fn(out)]} run before that frame. Returns CPU bool masks or None."""
    frames, out = clip()[1], []
    tr.set_prompt(first)
    for i in range(n):
        for e in (events or {}).get(i, []):
            e(out)
        lg = tr.step(frames[i])
        out.append(None if lg is None else (lg > 0).cpu())
    return out


def iou(A, B):
    return float(np.mean([float((a & b).sum() / max(1, (a | b).sum())) for a, b in zip(A, B)]))


def area(A):
    return float(np.mean([float(a.float().mean()) for a in A]))


def points(region, add, n=2):
    """n correction points spread top to bottom over a low-res bool mask."""
    ys, xs = np.nonzero(region.numpy())
    assert len(ys), "empty region"
    o, (h, w) = np.argsort(ys), region.shape
    return [{"x": float(xs[k] / w), "y": float(ys[k] / h), "add": add} for k in o[[len(o) * (j + 1) // (n + 1) for j in range(n)]]]
