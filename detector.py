"""Open-vocabulary detector shared by the live server, track.py and tools/anonymize.py.

Text prompts: the detector finds boxes once, a tracker follows them (Grounded-SAM-2 pattern). LLMDet tiny won the
bake-off in manual 10.4: same boxes as the other open detectors, widest vocabulary (LVIS minival 50.7 AP), ~180 ms.
"""

import torch
from torchvision.ops import nms

DETECTOR = "iSEE-Laboratory/llmdet_tiny"  # fixed id: the live page must not make the server download arbitrary repos
# 0.35: a 0.31 "giraffe" on a test pattern passed 0.30; a half-hidden boy still scored 0.40.
BOX_THRESHOLD, TEXT_THRESHOLD, NMS_IOU = 0.35, 0.25, 0.7


class Detector:
    """MM Grounding DINO architecture with LLMDet weights: text + frame -> boxes."""

    def __init__(self, model_id=DETECTOR):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to("cuda").eval()

    @torch.inference_mode()
    def find(self, rgb, labels, box_threshold=BOX_THRESHOLD, text_threshold=TEXT_THRESHOLD):
        """(3, H, W) uint8 CUDA frame, phrases -> [(label, score, [x0, y0, x1, y1] in 0..1)], best first.
        No count limit here: callers drop boxes they do not want first, then take as many as they have room for."""
        inputs = self.processor(images=rgb.permute(1, 2, 0).cpu().numpy(), text=[labels], return_tensors="pt").to("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):  # same boxes as fp32 in the bake-off, 1.3x faster
            out = self.model(**inputs)
        r = self.processor.post_process_grounded_object_detection(  # target size 1 x 1: boxes come out in 0..1
            out, threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(1, 1)])[0]
        keep = nms(r["boxes"].float(), r["scores"].float(), NMS_IOU).tolist()  # one box per thing, across labels
        scores, boxes = r["scores"][keep].tolist(), r["boxes"][keep].clamp(0, 1).tolist()
        return [(r["text_labels"][i], round(sc, 3), b) for i, sc, b in zip(keep, scores, boxes)]


def split_names(text):
    """'boy . red cup' -> ['boy', 'red cup']: dots, commas or semicolons separate the names."""
    import re

    return [t.strip().lower() for t in re.split(r"[.,;\n]", text) if t.strip()]
