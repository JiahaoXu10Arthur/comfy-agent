"""tagassert with the backend and threshold pinned by the catalog.

Replaces tagassert 0.1.0's shipped ComfyTagger backend, which never stages the PNG where
ComfyUI can load it (C1), scans the whole /history oldest-first instead of scoping to its
own prompt_id (C2), and can only read WD14Tagger|pysssss's comma-joined tag string, which
returns every tag at score 1.0 with the model's real confidence gone. This backend uploads
the PNG through ComfyUI's own /upload/image, submits a graph through the WD14Scores|comfy_agent
custom node (tools/wd14_scores) for real per-tag probabilities, and polls /history/{prompt_id}
the way comfy.Comfy.run does.
"""
import json

from tagassert import compare, normalise, NOT_JUDGEABLE

from .comfy import ComfyError


class ComfyBackend:
    def __init__(self, comfy, model):
        self.comfy, self.model = comfy, model

    def tag(self, image_path):
        name = self.comfy.upload_image(image_path)
        graph = {
            "1": {"class_type": "LoadImage", "inputs": {"image": name}},
            "2": {"class_type": "WD14Scores|comfy_agent",
                  "inputs": {"image": ["1", 0], "model": self.model,
                             "min_prob": 0.01, "replace_underscore": True}},
            "3": {"class_type": "PreviewAny", "inputs": {"source": ["2", 0]}},
        }
        outputs = self.comfy.run_for_outputs(graph)
        for node_id in ("2", "3"):
            for text in (outputs.get(node_id) or {}).get("text", []):
                try:
                    return json.loads(text)
                except (TypeError, ValueError):
                    continue
        raise ComfyError(f"no parseable tagger JSON in outputs: {outputs}")


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
