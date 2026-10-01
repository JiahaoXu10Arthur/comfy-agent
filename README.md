# comfy-agent

Measures whether an LLM agent is worth having at the *knob* tier of ComfyUI image generation: given a set of danbooru tags that must all appear in one render and a render budget, pick a workflow, a base model and LoRAs from a catalogue, turn one knob at a time, judge with a deterministic tagger, stop when every tag lands or say what the catalogue lacks.

The answer, measured on this machine and this catalogue: **not needed.** Three arms (a DeepSeek-driven agent, a fixed-seed baseline, a hard-coded ladder) each solved 30 of 30 runs; the agent averaged 1.07 renders per task, the ladder 1.13. The agent would have had to save a full render per task or judge more accurately to win. It did neither. That result is the deliverable, not a failure: the point of the project was to find out before building the next tier on top.

## Why this shape

- **The model never sees pixels.** Every verdict comes from the tagger ([WD14 eva02](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3), threshold 0.35) through [tagassert](https://pypi.org/project/tagassert/). Same image, same score, every time. An LLM judge would make the error rate a property of the judge, not of the agent.
- **One factor per render.** The model declares which single factor it changed (`weight:<tag>`, `move:<tag>`, `reword`, `lora:<file>`, `knob:<seed|cfg|steps>`, or `start` for a restart). The code checks the claim against the previous render ([attribution-gate](https://pypi.org/project/attribution-gate/) for weight changes) and rejects anything that changed two things. The rendered record is therefore a causal chain, not a pile of attempts.
- **The code decides when it is over.** Budget, three renders without the minimum tag score rising by 0.02, 24 turns, three bad tool calls, two consecutive ComfyUI errors: each forces a conclusion with a named stop reason. The model may only conclude `ACHIEVED` with a render whose tags all landed, and `NOT_ACHIEVABLE_HERE` only with a sentence about what is missing. Evidence is regenerated from the results database, never taken from the model.
- **Baselines go through the same tools.** Same gate, same judge, same conclude rules, no model. If the agent cannot beat a loop that increments the seed, the agent is the wrong tool for this tier.
- **Ground truth is a grid, not an opinion.** Before the eval, every task is rendered across base × LoRA × weight × six seeds (504 renders). A task is achievable if any cell lands. The agent's verdicts are scored against that, so "judged wrong" means something concrete.

## What was measured

10 tasks (single tag, multi-tag, counts, poses, two characters), budget 12 renders each, three runs with seeds 1001/1002/1003 shared across arms. Two workflows: an SDXL/Illustrious graph with two checkpoints and an Anima graph with two. 214 LoRAs in the catalogue, matched to tasks only through character or copyright tags.

| arm | correct | avg renders | switched workflow/base/LoRA | used history | pass^3 |
|---|---|---|---|---|---|
| agent (deepseek-chat) | 30/30 | 1.07 | 1/10 | 1/10 | 10/10 |
| hard-coded ladder | 30/30 | 1.13 | 1/10 | 0/10 | 10/10 |
| fixed seed | 30/30 | 1.07 | 0/10 | 0/10 | 10/10 |

98 renders, 15.7 minutes, under $0.60 of DeepSeek. n is small; the report is qualitative by design.

The grid came back with zero negative tasks: every task the selection table (RimoChan's per-checkpoint hit rates, judged with an older ML-Danbooru) said the SDXL checkpoint could not do, it did on the first seed. A probe of twelve obscure characters without their LoRAs found the checkpoint knew all of them. So the eval could only measure one kind of error, "said impossible when it was possible", and nobody made it.

## Compared with

- **[GenEval](https://github.com/djghosh13/geneval)** scores text-to-image alignment with an object detector on compositional prompts. Same instinct (a deterministic judge, not a VLM), different question: GenEval ranks models, this ranks *strategies for driving one model*.
- **[ComfyBench](https://github.com/xxyQwQ/ComfyBench)** benchmarks agents that build ComfyUI workflows from instructions. This project deliberately does not build graphs. The workflows are fixed JSON; the agent only chooses between them and turns knobs. Whether graph construction is worth an agent is a later tier.
- **RimoChan's hit-rate tables** for anime checkpoints were the task-selection basis. They did not predict this tagger's verdicts at the tested seeds (see above). They remain a good way to pick tags worth testing.

## What went wrong along the way

Every one of these is a test now.

- The stock `WD14Tagger|pysssss` node returns tag *names* joined by commas, so every tag over its own 0.01 floor scored 1.0 and the whole judging rule was void. [tools/wd14_scores](tools/wd14_scores) is a small custom node that replicates its preprocessing and returns real probabilities as JSON.
- On Windows (WDDM) `nvidia-smi` reports per-process memory as `[N/A]`, ComfyUI's `torch_vram_total` reads 0 under `cudaMallocAsync`, and GPU perf counters return nothing from an ssh session. The guard that keeps the eval off a GPU someone else is using now asks ComfyUI to free its models, then reads `vram_total − vram_free` from `/system_stats`.
- The tagger node escapes parentheses (`2b \(nier:automata\)`), so a character tag could never match until the backend unescaped it.
- The first live run: deepseek-chat rendered 12 times in 24 turns and never called `tag_check`. The model is stateless per turn, and the table it was shown listed only *judged* renders, so every turn looked like turn one. Unjudged renders are now listed with the exact `tag_check` call to make, and the prompt says a render has no score until judged. After the fix: one render, three turns.
- `seed_hint` was given to the baselines and not to the agent, which picked seed 1 on its own. Now in the prompt.
- Smaller: `read_text` without an encoding on a gbk locale; the catalogue took `tags[0]` (a franchise) as the LoRA's kind instead of the `char:` tag; the CLI opened the results database before creating its directory.

## Running it

Needs ComfyUI reachable on `127.0.0.1:8188` with the two pipeline graphs' models, the [comfyui-wd14-tagger](https://github.com/pythongosssss/ComfyUI-WD14-Tagger) models directory (for the csv and onnx), and `tools/wd14_scores` copied into `custom_nodes`. `DEEPSEEK_API_KEY` in the environment for the agent arm.

```bash
python -m comfy_agent --catalog catalog.yaml --out runs catalog build <lora-dir>
python -m comfy_agent --catalog catalog.yaml --out runs grid --tasks tasks
python -m comfy_agent --catalog catalog.yaml --out runs baseline --arm baseline-ladder --tasks tasks --run 1
python -m comfy_agent --catalog catalog.yaml --out runs run --task tasks/007.yaml --arm agent-deepseek --run 1
python -m comfy_agent --catalog catalog.yaml --out runs eval --tasks tasks --arms agent-deepseek,baseline-seed,baseline-ladder --runs 3
```

`eval` appends one line per finished (arm, run, task) to `runs/rows.jsonl` and resumes from it; a rerun overwrites exactly that scope. Every render's graph is recoverable from the PNG with `renderdb why`, and the eval spot-checks that against the results database.

Tests: `python -m pytest` (153; the loop is tested with a scripted provider, the tools with a fake ComfyUI, the single-factor gate with the real attribution-gate library).

## Layout

```
comfy_agent/
  catalog.py    build catalog.yaml from LoRA metadata; LoRA ↔ task matching
  tasks.py      load + lint tasks against the tagger vocabulary
  graph.py      edit pipeline JSON by node name; build the prompt
  comfy.py      submit, wait, fetch PNG; GPU guard
  factors.py    factor list, claims, single_factor gate
  results.py    results.sqlite
  tools.py      render / tag_check / history / conclude
  loop.py       the loop, stop rules, summary.md
  provider.py   openai client + cassette for replay tests
  baselines.py  fixed seed, hard-coded ladder
  grid.py       ground-truth grid
  eval.py       three runs, report.md, the one-line conclusion
pipelines/      two ComfyUI API graphs
tasks/          ten tasks
tools/wd14_scores/  the score-returning tagger node
tests/
```
