# WD14 Scores (comfy-agent)

A ComfyUI custom node that runs a WD14-family tagger and returns real per-tag
probabilities as JSON, instead of `WD14Tagger|pysssss`'s comma-joined tag string
(which loses every score to 1.0).

Install: copy this folder to ComfyUI's `custom_nodes/wd14_scores`, restart ComfyUI.

Dependencies (`numpy`, `PIL`, `onnxruntime`, `folder_paths`) are ComfyUI's own, already
present because the pysssss tagger node needs them — this node adds nothing to
comfy-agent's own dependency list.
