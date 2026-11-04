"""Track objects through a video with EfficientTAM or SAM3, prompted on frame 0 by points, boxes or text.

uv run track.py VIDEO.mp4 --point 210,350 [--point x,y] [--box x0,y0,x1,y1] [--text "cup . phone"]
                [--fix FRAME:OBJ:x,y:+|-] [--fix-box FRAME:OBJ:x0,y0,x1,y1] [--model s|ti|sam3] [--runs 3] [--out out.mp4]
uv run track.py VIDEO.mp4 --text "car . truck" --redetect 15 [--threshold 0.3] [--model s|ti] [--out out.mp4]
Each --point / --box is one object, in the order given; --text adds one object per box that the detector finds.
--fix / --fix-box correct object OBJ (1 = first) on a frame; tracking then runs again from there (EfficientTAM).
--redetect N runs the detector every N frames, so objects that enter later get a track, and ends the tracks of objects
that have gone (EfficientTAM, one run).
"""

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path

if "--redetect" in sys.argv:  # one progress bar per frame otherwise; tqdm reads this when it is imported
    os.environ.setdefault("TQDM_DISABLE", "1")

import cv2
import numpy as np
import torch
from hydra import initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from huggingface_hub import hf_hub_download

from efficient_track_anything.build_efficienttam import build_efficienttam_video_predictor

CONFIGS = Path(__file__).parent / "configs"  # upstream wheel ships without its YAML configs


def load_efficienttam(size, compile=True):
    """EfficientTAM-{s,ti} 512x512 video predictor on CUDA."""
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=str(CONFIGS.resolve()), version_base="1.2")
    ckpt = hf_hub_download("yunyangx/efficient-track-anything", f"efficienttam_{size}_512x512.pt")
    return build_efficienttam_video_predictor(
        f"efficienttam_{size}_512x512.yaml", ckpt,
        hydra_overrides_extra=[] if compile else ["++model.compile_image_encoder=False"],
    )


def floats(s):
    return [float(v) for v in s.split(",")]


def extract_frames(video, out_dir):
    """Decode video to 00000.jpg... (the predictor's JPEG-folder input). Returns (fps, frame count)."""
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps, n = cap.get(cv2.CAP_PROP_FPS) or 25.0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        cv2.imwrite(str(out_dir / f"{n:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        n += 1
    if n == 0:
        raise SystemExit(f"no frames decoded from {video}")
    return fps, n


# Mask colour of each object (BGR), the same order as the live page's palette.
PALETTE = [(255, 144, 30), (0, 149, 255), (89, 199, 52), (146, 45, 255), (10, 214, 255), (222, 82, 175), (199, 176, 48),
           (58, 69, 255)]


def colour(k):
    """Mask colour (BGR) of object k: the 8 page colours first, then hues spread by the golden ratio (all distinct)."""
    if k < len(PALETTE):
        return PALETTE[k]
    import colorsys

    r, g, b = colorsys.hsv_to_rgb((0.11 + 0.618034 * k) % 1, 0.85, 1.0)
    return int(255 * b), int(255 * g), int(255 * r)


def parse_fix(v):
    """'40:2:310,250:+' -> (40, 2, "point", [310, 250], 1); '-' is a remove point."""
    f, k, xy, sign = v.split(":")
    if sign not in "+-" or len(sign) != 1:
        raise ValueError(v)
    return int(f), int(k), "point", floats(xy), int(sign == "+")


def parse_fix_box(v):
    """'40:2:x0,y0,x1,y1' -> (40, 2, "box", [x0, y0, x1, y1], 1)."""
    f, k, box = v.split(":")
    return int(f), int(k), "box", floats(box), 1


def correct(predictor, state, fixes):
    """Add the --fix / --fix-box inputs to a tracked state. Each corrected frame becomes a new conditioning frame
    (live.as_cond_frame, the same step as the live corrections); a box also drops f's old mask, which would override it."""
    from live import as_cond_frame

    w, h = state["video_width"], state["video_height"]
    px = lambda pts: {"points": np.array([(x * w, y * h) for x, y, _ in pts], np.float32),  # noqa: E731
                      "labels": np.array([label for *_, label in pts], np.int32)} if pts else {}
    for f, k in sorted({(f, k) for f, k, *_ in fixes}):
        group = [x for x in fixes if x[:2] == (f, k)]
        boxes = [v for _, _, kind, v, _ in group if kind == "box"]
        pts = [(v[0] / w, v[1] / h, label) for _, _, kind, v, label in group if kind == "point"]
        i = state["obj_id_to_idx"][k]
        out = state["output_dict_per_obj"][i]
        keep, _ = as_cond_frame(out, state["frames_tracked_per_obj"][i], f, [] if boxes else pts)
        if keep is None:
            continue  # remove points on an empty mask: nothing to remove, the state stays as it is
        if boxes:  # a box says where the object is now (upstream needs clear_old_points with a box)
            out["cond_frame_outputs"].pop(f, None)
            out["non_cond_frame_outputs"].pop(f, None)
            predictor.add_new_points_or_box(state, frame_idx=f, obj_id=k, box=np.array(boxes[-1], np.float32), **px(pts))
        else:
            predictor.add_new_points_or_box(state, frame_idx=f, obj_id=k, clear_old_points=False, **px(keep))


def track(predictor, frames_dir, prompts, fixes=()):
    """prompts: [("point", [x, y]) | ("box", [x0, y0, x1, y1])] in frame-0 pixels -> {frame: (n, H, W) bool}, seconds.
    With fixes (parse_fix / parse_fix_box), the video is tracked, corrected, and tracked again from the first corrected
    frame; the time covers both passes."""
    state = predictor.init_state(video_path=str(frames_dir))
    for k, (kind, v) in enumerate(prompts, 1):
        kw = ({"points": np.array([v], np.float32), "labels": np.array([1], np.int32)} if kind == "point"
              else {"box": np.array(v, np.float32)})
        predictor.add_new_points_or_box(state, frame_idx=0, obj_id=k, **kw)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    masks = {f: (logits[:, 0] > 0).cpu().numpy() for f, _, logits in predictor.propagate_in_video(state)}
    if fixes:  # track again from the first corrected frame; the frames before it keep their masks
        correct(predictor, state, fixes)
        start = min(f for f, *_ in fixes)
        masks |= {f: (logits[:, 0] > 0).cpu().numpy()
                  for f, _, logits in predictor.propagate_in_video(state, start_frame_idx=start)}
    torch.cuda.synchronize()
    return masks, time.perf_counter() - t0


def track_sam3(model, processor, frames, prompts):
    """Same protocol as track(): session setup untimed, propagation timed."""
    session = processor.init_video_session(video=frames, inference_device="cuda", dtype=torch.bfloat16)
    for k, (kind, v) in enumerate(prompts, 1):
        kw = ({"input_points": [[[v]]], "input_labels": [[[1]]]} if kind == "point" else {"input_boxes": [[v]]})
        processor.add_inputs_to_inference_session(inference_session=session, frame_idx=0, obj_ids=k, **kw)
    session.obj_with_new_inputs = list(range(1, len(prompts) + 1))  # each add call replaces this list (live.py too)
    model(inference_session=session, frame_idx=0)  # segment prompt frame, like EfficientTAM's add_new_points_or_box
    size = [[session.video_height, session.video_width]]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    masks = {
        out.frame_idx: processor.post_process_masks([out.pred_masks], original_sizes=size)[0][:, 0].cpu().numpy()
        for out in model.propagate_in_video_iterator(session)
    }
    torch.cuda.synchronize()
    return masks, time.perf_counter() - t0


def find_text(frames_dir, text, threshold):
    """Detector boxes on frame 0 for the names in `text` -> ([("box", pixels)], labels)."""
    from detector import Detector, split_names

    bgr = cv2.imread(str(frames_dir / "00000.jpg"))
    h, w = bgr.shape[:2]
    rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).cuda()
    found = Detector().find(rgb, split_names(text), box_threshold=threshold)
    for label, score, b in found:
        print(f"found {label} {score:.2f} at {[round(b[0] * w), round(b[1] * h), round(b[2] * w), round(b[3] * h)]}")
    return [("box", [b[0] * w, b[1] * h, b[2] * w, b[3] * h]) for _, _, b in found], [label for label, _, _ in found]


SAME = 0.5  # share of the smaller mask that another mask must cover to be the same object
LOST = 15  # frames with an empty mask after which a track ends
TOO_BIG = 0.25  # share of the frame: a mask this big is the background (a sports field), not an object
MAX_OBJECTS = 64  # each object adds time per frame; boxes above this wait for the next detection after tracks end


def track_redetect(tracker, detector, video, names, every, threshold, out):
    """Detect every `every` frames, track, write the overlay to `out`. Returns counts and times. tracker: live tracker."""
    import live

    limit, live.MAX_OBJ = live.MAX_OBJ, MAX_OBJECTS  # the page's 8 colours do not limit this tool
    try:
        return redetect_loop(tracker, detector, video, names, every, threshold, out)
    finally:
        live.MAX_OBJ = limit


def redetect_loop(tracker, detector, video, names, every, threshold, out):
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps_in, writer = cap.get(cv2.CAP_PROP_FPS) or 25.0, None
    masks, areas, empty = None, None, {}  # masks of the last frame, in tracker.ids order; frames each id has been empty
    stats = dict(frames=0, added=0, ended=0, duplicates=0, too_big=0, most=0, detect_s=0.0, area0=0.0)
    t0 = time.perf_counter()
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        h, w = bgr.shape[:2]
        rgb = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).cuda()
        known = set(tracker.ids)
        if stats["frames"] % every == 0:
            t1 = time.perf_counter()
            # add_boxes drops boxes that match a mask box. A box round more than a mask can be a new object next to or in
            # front of it (the girl's box holds the boy's mask), so only the new object's first mask decides (on_old).
            tracker.add_boxes(detector.find(rgb, names, box_threshold=threshold), True)
            stats["detect_s"] += time.perf_counter() - t1
        logits = tracker.step(rgb)
        ids = list(tracker.ids)
        if logits is None:
            masks, areas = None, None
            layers = []
        else:
            masks = torch.nn.functional.interpolate(logits[:, None].float(), (h, w), mode="bilinear", align_corners=False)[:, 0] > 0
            areas = masks.flatten(1).sum(1)
            layers = masks.cpu().numpy()
        big = {k for k, m in zip(ids, layers) if m.mean() >= TOO_BIG}
        for k, m in zip(ids, layers):
            if k not in big:
                bgr[m] = (0.5 * bgr[m] + 0.5 * np.array(colour(k - 1))).astype(np.uint8)
        if writer is None:
            writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        writer.write(bgr)
        if not stats["frames"] and len(layers):
            stats["area0"] = float(np.any(layers, 0).mean())
        new = [i for i, k in enumerate(ids) if k not in known]
        stats["added"] += len(new)
        gone = set()
        for i in new:  # the first mask shows what the box really selected: a mask on a tracked object is that object again
            if masks is not None and on_old(masks, areas, i, new, [j for j, k in enumerate(ids) if k in big]):
                gone.add(ids[i])
                stats["duplicates"] += 1
        for k, m in zip(ids, layers):
            empty[k] = empty.get(k, 0) + 1 if not m.any() else 0
            if empty[k] == LOST:
                gone.add(k)
                stats["ended"] += 1
        stats["too_big"] += len(big - gone)
        for k in gone | big:
            tracker.remove(k)
        stats["most"] = max(stats["most"], len(set(ids) - gone - big))
        stats["frames"] += 1
    if writer is not None:
        writer.release()
    if not stats["frames"]:
        raise SystemExit(f"no frames decoded from {video}")
    stats["seconds"] = time.perf_counter() - t0
    return stats


def on_old(masks, areas, i, new, skip=()):
    """Does mask i, added on this frame, lie on an older mask or on one added before it on this frame? skip: background
    masks, which hold every object on the turf."""
    older = [j for j in range(len(masks)) if j != i and j not in skip and (j not in new or j < i)]
    if not older or not areas[i]:
        return False
    inter = (masks[older] & masks[i]).flatten(1).sum(1).float()
    return bool((inter / areas[older].clamp(max=int(areas[i])).clamp(min=1)).max() >= SAME)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", type=Path)
    ap.add_argument("--point", type=lambda v: ("point", floats(v)), action="append", dest="prompts", default=[],
                    help="x,y in frame-0 pixels; one object (repeat for more)")
    ap.add_argument("--box", type=lambda v: ("box", floats(v)), action="append", dest="prompts",
                    help="x0,y0,x1,y1 in frame-0 pixels; one object (repeat for more)")
    ap.add_argument("--text", help='names for the detector on frame 0, e.g. "garbage can . trash can"; one object per box')
    ap.add_argument("--threshold", type=float, default=0.35, help="detector box score for --text (default 0.35)")
    ap.add_argument("--fix", type=parse_fix, action="append", dest="fixes", default=[],
                    help="FRAME:OBJ:x,y:+ or - : add (+) or remove (-) point on object OBJ at FRAME (pixels); repeatable")
    ap.add_argument("--fix-box", type=parse_fix_box, action="append", dest="fixes",
                    help="FRAME:OBJ:x0,y0,x1,y1 : new box for object OBJ at FRAME (pixels); repeatable")
    ap.add_argument("--redetect", type=int, metavar="N", help="with --text: run the detector every N frames, add new objects, "
                    "end tracks of objects that have gone (EfficientTAM, one run; slower than one detection)")
    ap.add_argument("--model", choices=["s", "ti", "sam3"], default="s")
    ap.add_argument("--runs", type=int, default=3, help="first run includes warm-up/compile")
    ap.add_argument("--no-compile", action="store_true", help="skip torch.compile of the image encoder")
    ap.add_argument("--out", type=Path, default=Path("out.mp4"))
    a = ap.parse_args()
    if any(len(v) != (2 if kind == "point" else 4) for kind, v in a.prompts):
        ap.error("--point needs x,y; --box needs x0,y0,x1,y1")
    if not a.prompts and not a.text:
        ap.error("give at least one --point, --box or --text")
    if a.runs < 1:
        ap.error("--runs must be >= 1")
    if any(len(v) != (2 if kind == "point" else 4) or f < 0 or k < 1 for f, k, kind, v, _ in a.fixes):
        ap.error("--fix needs FRAME:OBJ:x,y:+|-, --fix-box FRAME:OBJ:x0,y0,x1,y1 (FRAME >= 0, OBJ >= 1)")
    if a.fixes and a.model == "sam3":
        ap.error("--fix / --fix-box: EfficientTAM only (s, ti)")

    if a.redetect is not None and (a.redetect < 1 or not a.text or a.prompts or a.fixes or a.model == "sam3" or a.no_compile):
        ap.error("--redetect N needs N >= 1 and --text; it does not work with --point, --box, --fix, --model sam3, --no-compile")

    assert torch.cuda.is_available(), "CUDA GPU required"
    print(f"GPU: {torch.cuda.get_device_name(0)}  torch {torch.__version__}")
    if a.redetect:
        import live
        from detector import Detector, split_names

        tracker = live.load(f"etam-{a.model}")  # its own autocast: step() runs like in the live server
        s = track_redetect(tracker, Detector(), a.video, split_names(a.text), a.redetect, a.threshold, a.out)
        print(f"{s['frames']} frames in {s['seconds']:.2f}s -> {s['frames'] / s['seconds']:.1f} FPS (detector {s['detect_s']:.1f}s, "
              f"peak GPU mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)")
        print(f"objects: {s['added'] - s['duplicates'] - s['too_big']} tracked in all, at most {s['most']} at a time, {s['ended']} ended, "
              f"{s['duplicates']} duplicates and {s['too_big']} background masks removed; frame0 {s['area0']:.3%}; wrote {a.out}")
        return
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()  # as upstream examples
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True

    model = processor = predictor = None
    frames = []
    if a.model == "sam3":  # gated repo: needs HF_TOKEN with approved access
        from transformers import Sam3TrackerVideoModel, Sam3TrackerVideoProcessor

        model = Sam3TrackerVideoModel.from_pretrained("facebook/sam3", dtype=torch.bfloat16).to("cuda").eval()
        processor = Sam3TrackerVideoProcessor.from_pretrained("facebook/sam3")
    else:
        predictor = load_efficienttam(a.model, compile=not a.no_compile)

    with tempfile.TemporaryDirectory() as tmp:
        frames_dir = Path(tmp)
        t0 = time.perf_counter()
        fps_in, n_frames = extract_frames(a.video, frames_dir)
        print(f"decode: {time.perf_counter() - t0:.2f}s")
        prompts, labels = list(a.prompts), [""] * len(a.prompts)
        if a.text:
            boxes, names = find_text(frames_dir, a.text, a.threshold)
            prompts, labels = prompts + boxes, labels + names
        if not prompts:
            raise SystemExit(f"nothing found for {a.text!r}; try a lower --threshold or a different name")
        if bad := [x for x in a.fixes if x[0] >= n_frames or x[1] > len(prompts)]:
            ap.error(f"--fix: frame must be < {n_frames} and object <= {len(prompts)}; got {bad}")
        if a.model == "sam3":
            frames = [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in sorted(frames_dir.glob("*.jpg"))]
        masks = {}  # --runs >= 1, so the loop always fills it
        for r in range(a.runs):
            torch.cuda.reset_peak_memory_stats()
            if a.model == "sam3":
                assert model is not None and processor is not None
                masks, dt = track_sam3(model, processor, frames, prompts)
            else:
                assert predictor is not None
                masks, dt = track(predictor, frames_dir, prompts, a.fixes)
            print(f"run {r}: {len(masks)} frames{' (+ again from the first --fix frame)' if a.fixes else ''} in {dt:.2f}s -> "
                  f"{len(masks) / dt:.1f} FPS "
                  f"(peak GPU mem {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB)")

        assert len(masks) == n_frames, "every frame must get a mask"
        n, h, w = masks[0].shape
        vw = cv2.VideoWriter(str(a.out), cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))
        for f in sorted(masks):
            img = cv2.imread(str(frames_dir / f"{f:05d}.jpg"))
            for k, m in enumerate(masks[f]):
                img[m] = (0.5 * img[m] + 0.5 * np.array(colour(k))).astype(np.uint8)
            vw.write(img)
        vw.release()
    if n > 1:
        for k in range(n):
            area = [m[k].mean() for m in masks.values()]
            print(f"object {k + 1}{' ' + labels[k] if labels[k] else ''}: frame0 {area[0]:.3%}, min {min(area):.3%}, "
                  f"max {max(area):.3%}, empty in {sum(v == 0 for v in area)} frames")
    area = [m.any(0).mean() for m in masks.values()]  # union of all objects (one object: the same as before)
    print(f"mask area: frame0 {area[0]:.3%}, min {min(area):.3%}, max {max(area):.3%}; wrote {a.out}")


if __name__ == "__main__":
    main()
