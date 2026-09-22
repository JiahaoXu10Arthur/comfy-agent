# comfy-agent

An agent that, given a set of tags that must appear in an image, picks a ComfyUI workflow, base model and LoRAs from a catalogue, renders, judges the result with a deterministic tagger, and decides whether to turn a knob, repaint a region, or edit the graph. Stops when the tags land, or says what is missing.

Status: planning. Design notes live outside this repo.
