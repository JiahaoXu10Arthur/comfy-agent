"""WD14 Scores: a ComfyUI custom node returning real per-tag probabilities.

WD14Tagger|pysssss joins its output into one comma-separated string, so every tag
comes back at score 1.0 with the model's confidence gone. This node runs the same
ONNX tagger and returns {tag: probability} as JSON, preprocessed exactly as pysssss
does so the scores match.
"""
import csv
import json
import os

try:
    import folder_paths
except ImportError:
    folder_paths = None

_SESSIONS = {}

def _model_dir():
    if folder_paths is not None:
        try:
            base = os.path.join(folder_paths.get_folder_paths("custom_nodes")[0],
                                 "comfyui-wd14-tagger", "models")
            if os.path.isdir(base):
                return base
        except Exception:
            pass
    return os.path.join(os.path.dirname(__file__), "..", "comfyui-wd14-tagger", "models")

def _list_models():
    try:
        return sorted(f[:-5] for f in os.listdir(_model_dir()) if f.endswith(".onnx"))
    except OSError:
        return []

def read_tags(csv_path):
    """selected_tags.csv (tag_id,name,category,count) -> tag names in row order."""
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    name_idx = rows[0].index("name")
    return [row[name_idx] for row in rows[1:]]

def select(probs, names, min_prob, replace_underscore=True):
    """{tag: round(prob, 4)} for every tag at or above min_prob."""
    out = {}
    for name, p in zip(names, probs):
        p = float(p)
        if p < min_prob:
            continue
        out[name.replace("_", " ") if replace_underscore else name] = round(p, 4)
    return out

def _session(model_dir, model_name):
    if model_name not in _SESSIONS:
        import onnxruntime
        path = os.path.join(model_dir, f"{model_name}.onnx")
        _SESSIONS[model_name] = onnxruntime.InferenceSession(
            path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    return _SESSIONS[model_name]

class WD14Scores:
    @classmethod
    def INPUT_TYPES(cls):
        models = _list_models() or ["wd-eva02-large-tagger-v3"]
        default = "wd-eva02-large-tagger-v3" if "wd-eva02-large-tagger-v3" in models else models[0]
        return {"required": {
            "image": ("IMAGE",),
            "model": (models, {"default": default}),
            "min_prob": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.01}),
            "replace_underscore": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "tag"
    CATEGORY = "comfy_agent"
    OUTPUT_NODE = True

    def tag(self, image, model, min_prob, replace_underscore):
        import numpy as np
        from PIL import Image

        model_dir = _model_dir()
        names = read_tags(os.path.join(model_dir, f"{model}.csv"))
        session = _session(model_dir, model)
        input_meta = session.get_inputs()[0]
        h = input_meta.shape[1]

        pil = Image.fromarray((image[0].cpu().numpy() * 255).astype("uint8"))
        w_img, h_img = pil.size
        ratio = h / max(pil.size)
        resized = pil.resize((int(w_img * ratio), int(h_img * ratio)), Image.LANCZOS)
        canvas = Image.new("RGB", (h, h), (255, 255, 255))
        canvas.paste(resized, ((h - resized.width) // 2, (h - resized.height) // 2))

        x = np.expand_dims(np.array(canvas).astype(np.float32)[:, :, ::-1], 0)
        probs = session.run([session.get_outputs()[0].name], {input_meta.name: x})[0][0]

        s = json.dumps(select(probs, names, min_prob, replace_underscore))
        return {"ui": {"text": [s]}, "result": (s,)}

NODE_CLASS_MAPPINGS = {"WD14Scores|comfy_agent": WD14Scores}
NODE_DISPLAY_NAME_MAPPINGS = {"WD14Scores|comfy_agent": "WD14 Scores (comfy-agent)"}
