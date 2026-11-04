"""Remove personal data from a phone clip before it goes into data/ (manual 5.5). Runs on a GPU host.

uv run --with imageio-ffmpeg python tools/anonymize.py SRC --subject "garbage can . trash can" --out OUT.mp4

1. The detector finds the subject on frame 0, and a tracker follows it through the clip.
2. Everything outside the subject (plus a margin) gets a strong blur: plates, houses, painted numbers.
3. Text and faces on the subject (house numbers on bins, plates on a tracked car) are pixelated.
4. The output is H.264 without sound, metadata tracks or tags (GPS, device, dates).
5. OUT_sheet.jpg shows frames for a human check; the report lists frames where an object had no detection of its
   own (they only got boxes moved from nearby frames, or none: fine for an object without text, else check them).
Memory: the clip is read twice from disk; only low-res masks (~8 KB per object per frame) and boxes stay in memory.
Use anonymize_spark2.sh from the Mac: it copies the clip to the host and always deletes that copy afterwards.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import live  # noqa: E402  (trackers; the project root is not on sys.path for tools/)
from detector import Detector, split_names  # noqa: E402

PII = "license plate . house number . address number . face"


def ffmpeg_exe():
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise SystemExit("needs ffmpeg: install it, or run with `uv run --with imageio-ffmpeg`")


def pixelate(img, a, b, c, e, blocks=4):
    """Mosaic img[b:e, a:c] with `blocks` cells across its short side: digits and faces become unreadable."""
    k = max(3, min(c - a, e - b) // blocks)
    small = cv2.resize(img[b:e, a:c], (max(1, (c - a) // k), max(1, (e - b) // k)), interpolation=cv2.INTER_AREA)
    img[b:e, a:c] = cv2.resize(small, (c - a, e - b), interpolation=cv2.INTER_NEAREST)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("--subject", required=True, help='names of what stays sharp, e.g. "garbage can . trash can"')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default="sam2", choices=["sam2", "etam-s", "etam-ti"], help="tracker for the subject")
    ap.add_argument("--subject-threshold", type=float, default=0.35)
    ap.add_argument("--pii", default=PII, help=f"names to pixelate on the subject (default: {PII!r})")
    ap.add_argument("--pii-threshold", type=float, default=0.2, help="low on purpose: a false mosaic is cheap")
    ap.add_argument("--max-area", type=float, default=0.01,
                    help="ignore number boxes larger than this (they are whole objects); other names have no limit")
    ap.add_argument("--spread", type=int, default=2, help="also pixelate a box in this many frames before and after")
    ap.add_argument("--margin", type=int, default=12, help="sharp pixels round the subject")
    ap.add_argument("--max-objects", type=int, default=8)
    a = ap.parse_args()
    if a.spread < 0:
        ap.error("--spread must be >= 0 (0: only the frame where a box was found)")

    cap = cv2.VideoCapture(str(a.src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    ok, first = cap.read()
    if not ok:
        raise SystemExit(f"cannot read {a.src}")
    H, W = first.shape[:2]
    to_t = lambda f: torch.from_numpy(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).cuda()  # noqa: E731

    # Pass 1 (GPU), streamed: per frame only the low-res subject masks (packed bits, ~8 KB per object), the centroid
    # of each object and the PII boxes are kept, so memory does not hold the decoded clip.
    live.MAX_OBJ = a.max_objects
    tracker, det = live.load(a.model), Detector()
    found = det.find(to_t(first), split_names(a.subject), box_threshold=a.subject_threshold)
    kept = tracker.add_boxes(found, False)
    if not kept:
        raise SystemExit(f"subject {a.subject!r} not found on frame 0; try a lower --subject-threshold")
    print(f"subject: {len(kept)} object(s): " + ", ".join(f"{label} {score:.2f}" for label, score, _ in kept), flush=True)
    names = split_names(a.pii)
    low, cen, pii, shape = [], [], [], None
    with torch.inference_mode():
        f = first
        while f is not None:
            rgb = to_t(f)
            lg = tracker.step(rgb)  # None only when no object is left
            m = np.zeros((0, 1, 1), bool) if lg is None else (lg > 0).cpu().numpy()
            shape = m.shape[1:] if len(m) else shape
            low.append((len(m), m.shape[1:], np.packbits(m)))
            cen.append([(xs.mean() / m.shape[2], ys.mean() / m.shape[1]) if len(xs) else None
                        for ys, xs in (np.nonzero(o) for o in m)])
            pii.append(det.find(rgb, names, box_threshold=a.pii_threshold, text_threshold=a.pii_threshold))
            ok, f = cap.read()
            f = f if ok else None
    cap.release()
    N = len(low)

    def masks_at(i):
        n, hw, bits = low[i]
        return np.unpackbits(bits)[: n * hw[0] * hw[1]].reshape(n, *hw).astype(bool)

    def owner(i, x, y):
        """Object whose low-res mask (widened by 2 cells) holds the point (x, y) in 0..1 at frame i, or None."""
        for k, o in enumerate(masks_at(i)):
            h, w = o.shape
            if cv2.dilate(o.astype(np.uint8), np.ones((5, 5), np.uint8))[min(h - 1, int(y * h)), min(w - 1, int(x * w))]:
                return k
        return None

    # Pass 2: read the clip again, compose and encode without sound or metadata.
    out = subprocess.Popen([ffmpeg_exe(), "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
                            "-s", f"{W}x{H}", "-r", f"{fps:g}", "-i", "-", "-map_metadata", "-1", "-map_chapters", "-1",
                            "-an", "-sn", "-dn", "-c:v", "libx264", "-crf", "23", "-preset", "slow",
                            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(a.out)], stdin=subprocess.PIPE)
    pipe = out.stdin
    assert pipe is not None
    kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * a.margin + 1, 2 * a.margin + 1))
    small = (90, max(1, round(90 * H / W)))  # the background is rebuilt from ~90 px across: nothing small stays readable
    unchecked = {k: [] for k in range(len(kept))}  # frames with no detection of their own on the object
    sheet_at = set(np.linspace(0, N - 1, 12).astype(int).tolist())
    sheet, done = [], 0  # done: boxes pixelated (the test checks that the privacy step ran)
    cap = cv2.VideoCapture(str(a.src))
    for i in range(N):
        ok, f = cap.read()
        if not ok:
            raise SystemExit(f"{a.src} gave fewer frames on the second read")
        each = [cv2.dilate(cv2.resize(o.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR), kern)
                for o in masks_at(i)]  # one margin-widened mask per object
        subj = np.maximum.reduce(each) if each else np.zeros((H, W), np.uint8)
        alpha = cv2.GaussianBlur(subj.astype(np.float32), (31, 31), 0)[..., None]
        bg = cv2.GaussianBlur(cv2.resize(cv2.resize(f, small, interpolation=cv2.INTER_AREA), (W, H)), (0, 0), 6)
        img = (f * alpha + bg * (1 - alpha)).astype(np.uint8)
        seen = set()
        for j in range(max(0, i - a.spread), min(N, i + a.spread + 1)):
            for label, _, (x0, y0, x1, y1) in pii[j]:
                if (x1 - x0) * (y1 - y0) > a.max_area and "number" in label:
                    continue  # a large "number" box is the whole bin; any other large box is still pixelated
                k = owner(j, (x0 + x1) / 2, (y0 + y1) / 2)
                if k is None:
                    continue  # not on the subject: the background blur covers it
                if j != i:  # a box from a nearby frame moves with its object (centroid shift)
                    if k >= len(cen[i]) or cen[i][k] is None or cen[j][k] is None:
                        continue
                    dx, dy = cen[i][k][0] - cen[j][k][0], cen[i][k][1] - cen[j][k][1]
                    x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
                else:
                    seen.add(k)
                pw, ph = (x1 - x0) * 0.2 + 4 / W, (y1 - y0) * 0.2 + 4 / H
                a0, b0 = max(0, int((x0 - pw) * W)), max(0, int((y0 - ph) * H))
                c0, e0 = min(W, int((x1 + pw) * W) + 1), min(H, int((y1 + ph) * H) + 1)
                if c0 - a0 >= 2 and e0 - b0 >= 2:
                    pixelate(img, a0, b0, c0, e0)
                    done += 1
        for k in unchecked:
            if k not in seen:
                unchecked[k].append(i)
        pipe.write(img.tobytes())
        if i in sheet_at:
            sheet.append(cv2.resize(img, (240, round(240 * H / W))))
    cap.release()
    pipe.close()
    if out.wait():
        raise SystemExit("ffmpeg failed")
    while len(sheet) < 12:
        sheet.append(np.zeros_like(sheet[0]))
    cv2.imwrite(str(a.out.with_name(a.out.stem + "_sheet.jpg")),
                np.vstack([np.hstack(sheet[r:r + 6]) for r in range(0, 12, 6)]), [cv2.IMWRITE_JPEG_QUALITY, 85])
    tags = subprocess.run([ffmpeg_exe(), "-loglevel", "error", "-i", str(a.out), "-f", "ffmetadata", "-"],
                          capture_output=True, text=True).stdout.splitlines()
    tags = [t for t in tags if "=" in t and not t.startswith(("encoder=", "major_brand=", "minor_version=", "compatible_brands="))]
    print(f"wrote {a.out}: {N} frames {W}x{H}, no sound; tags left: {tags or 'none'}; pixelated {done} boxes")
    for k, gaps in unchecked.items():
        print(f"object {k + 1} ({kept[k][0]}): no detection of its own in {len(gaps)} of {N} frames"
              + (f", e.g. {gaps[:8]}" if gaps else "")
              + (" - check them (they only have boxes moved from nearby frames)" if gaps else ""))

if __name__ == "__main__":
    main()
