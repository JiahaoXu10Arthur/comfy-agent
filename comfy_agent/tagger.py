"""tagassert with the backend and threshold pinned by the catalog.

Replaces tagassert 0.1.0's shipped ComfyTagger backend: that implementation never stages
the PNG where ComfyUI can load it (C1) and scans the whole /history oldest-first instead
of scoping to its own prompt_id, so every tag_check after the first returns stale scores
(C2). This backend uploads the PNG through ComfyUI's own /upload/image, submits the same
three-node graph tagassert.backends.ComfyTagger.tag builds, and polls /history/{prompt_id}
the way comfy.Comfy.run does.
"""
from tagassert import compare, normalise, NOT_JUDGEABLE
from tagassert.backends import BackendError, ComfyTagger

from .comfy import ComfyError


class ComfyBackend:
    def __init__(self, comfy, model):
        self.comfy, self.model = comfy, model

    def tag(self, image_path):
        name = self.comfy.upload_image(image_path)
        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": name}},
            "2": {"class_type": "WD14Tagger|pysssss",
                  "inputs": {"image": ["1", 0], "model": self.model,
                             "threshold": 0.01, "character_threshold": 0.01,
                             "replace_underscore": True, "trailing_comma": False,
                             "exclude_tags": ""}},
            "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        }
        outputs = self.comfy.run_for_outputs(graph)
        try:
            return ComfyTagger._parse(outputs)
        except BackendError as e:
            raise ComfyError(str(e)) from e


def comfy_backend(comfy, model):
    return ComfyBackend(comfy, model)


class Tagger:
    def __init__(self, backend, threshold, vocabulary):
        self.backend, self.threshold, self.vocabulary = backend, threshold, vocabulary

    def judge(self, png_path, must):
        detected = self.backend.tag(png_path)
        v = compare(must, detected, threshold=self.threshold, vocabulary=self.vocabulary)
        out = {}
        for r in v.results:
            score = 0.0 if r.verdict == NOT_JUDGEABLE or r.score is None else float(r.score)
            out[r.tag] = (r.verdict, score)
        return {m: out.get(normalise(m), ("NOT_JUDGEABLE", 0.0)) for m in must}
