"""tagassert with the backend and threshold pinned by the catalog."""
from tagassert import compare, NOT_JUDGEABLE
from tagassert.backends import ComfyTagger


def comfy_backend(url, model):
    return ComfyTagger(url=url, model=model)


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
        return out
