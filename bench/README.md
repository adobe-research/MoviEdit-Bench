# MoviEdit-Bench

A benchmark for **long-form, multi-shot video editing**. It tests what is specific to multi-shot editing: whether an edit still targets the right subject after a cut, and whether the edited subject looks the same each time it reappears.

Everything needed to run the benchmark on your own model is provided:

- **Benchmark data**: source videos, edit instructions, edited reference frames, and evaluation metadata. See [Data](#data).
- **Evaluation code**: the metric implementations and the VLM judge, in locked environments. See [Installation](#installation) and [Evaluation](#evaluation).
- **Reference results**: MoviEdit's edited videos and the per-scenario results of the paper's evaluation run.
- **Reproducibility check**: evaluate the reference videos yourself and compare with the numbers reported in the paper. See [Reproducing the paper's numbers](#reproducing-the-papers-numbers).

See the paper for the benchmark design, metric definitions, and results.

## Contents

- [Overview](#overview)
- [Data](#data)
- [Generating Your Model's Outputs](#generating-your-models-outputs)
- [Installation](#installation)
- [Evaluation](#evaluation)

## Overview

| | |
|---|---|
| Source videos | 64 generated multi-shot videos, 4–6 shots each, 361 frames @ 24 fps (~15 s), 1280×720 |
| Editing scenarios | 128 (2 per video) |
| Edit types | local: add, remove, appearance change, swap · global: environment change, stylization |
| Tasks | **V2V**: source video + edit instruction<br>**IV2V**: source video + edit instruction + one edited reference frame (first frame in 64 scenarios, a later frame in the other 64) |
| Metrics | non-subject preservation, edit localization, subject preservation, cross-shot edit consistency, edit-prompt alignment, video quality, edited-frame fidelity (IV2V), and overall edit success |

The metrics combine pixel- and embedding-based scores computed on one GPU with judgments from a VLM (Gemma 4 31B).

## Data

**Download:** [`moviedit_bench_data.zip`](https://drive.google.com/file/d/1RU2kO5eBggu2zxgMYSbreMx3oXlIL_iX/view?usp=sharing) (Google Drive, 604 MB). Unzip it inside `bench/` to get `bench/data/`.

```bash
cd bench
unzip moviedit_bench_data.zip        # creates data/
sha256sum moviedit_bench_data.zip    # 958621491d291170e14d516fa3308959793b55b56087f246d88172bf267ade02
```

The benchmark directory separates what a model needs from what only the evaluation needs:

```
data/
├── benchmark.jsonl         # inference inputs, V2V
├── benchmark_iv2v.jsonl    # inference inputs, IV2V (adds the edited reference frame)
├── source_videos/          # 64 source videos, 1280×720, 361 frames @ 24 fps
├── edited_frames/          # 128 edited reference frames for IV2V, 1280×720
├── eval_metadata.json      # evaluation only, shared by V2V and IV2V
└── masks/                  # evaluation only: mask videos of local-edit targets, frame-aligned with the source
```

**Inference inputs.** One line per scenario (128 lines). `benchmark_iv2v.jsonl` has the same lines plus the last two fields:

```jsonc
{
  "instruction_id": "C1_4_A__remove__01",
  "sample_id": "C1_4_A",
  "source_video": "source_videos/C1_4_A_shots4_dur15_seed43_1280x720_0e18129eab.mp4",
  "task": "remove",                    // add | remove | appearance_change | swap | environment_change | stylization
  "edit_instruction": "Remove the man",
  "reference_frame_indices": [88],     // IV2V: source frame index of the edited reference frame
  "edited_frame_paths": {"88": "edited_frames/C1_4_A__remove__01_f88.png"}   // IV2V
}
```

**Evaluation metadata.** `eval_metadata.json` is never given to the model:

```jsonc
{
  "samples": {
    "C1_4_A": {
      "category": "C1",
      "shots": [[0, 96], [96, 188], [188, 291], [291, 361]],   // [start, end) frame range of each shot
      "source_prompt": "<global + per-shot description of the whole source video>",
      "source_shot_prompts": ["<description of shot 0 of the source>", "..."]   // one per shot
    }
  },
  "instructions": {
    "C1_4_A__remove__01": {
      "target": {                      // all null except "type": "scene" for global edits
        "type": "character",           // character | animal | object | scene
        "id": "char_a",
        "mask": "masks/C1_4_A_char_a.mp4",
        "shots": [0, 1, 2, 3]          // shots in which the target appears
      },
      "target_prompt": "<global + per-shot description of the whole edited video>",
      "target_shot_prompts": ["<description of shot 0 after the edit>", "..."]   // one per shot
    }
  }
}
```

## Generating Your Model's Outputs

Run your model once per line of the inference file and save the result as **`{instruction_id}.mp4`**, all in one directory.

| Task | Inference file | Model inputs |
|---|---|---|
| V2V | `benchmark.jsonl` | `source_video`, `edit_instruction` |
| IV2V | `benchmark_iv2v.jsonl` | `source_video`, `edit_instruction`, and the edited frame in `edited_frame_paths`, which is the edited version of source frame `reference_frame_indices[0]` |

- **Do not use anything from `eval_metadata.json` or `masks/` as model input.** One exception: a method that is conditioned on descriptions of the source and edited videos instead of an edit instruction (e.g., SDEdit, Wan-Edit) may use the `source_*` and `target_*` prompts.
- **Length:** output all 361 frames at 24 fps, frame-aligned with the source. If a video is shorter, only the frames it has are evaluated, and the run prints a warning.
- **Resolution:** any. The evaluator resizes source, output, and masks to 960×544.
- Use separate output directories for V2V and IV2V.

A minimal loop:

```python
import json

for line in open("data/benchmark_iv2v.jsonl"):          # or data/benchmark.jsonl for V2V
    s = json.loads(line)
    ref_index = s["reference_frame_indices"][0]          # IV2V only
    ref_frame = "data/" + s["edited_frame_paths"][str(ref_index)]
    video = my_model.edit(
        video="data/" + s["source_video"],
        instruction=s["edit_instruction"],
        reference_frame=ref_frame, reference_frame_index=ref_index,   # IV2V only
    )
    video.save(f"outputs/my_method_iv2v/{s['instruction_id']}.mp4")
```

## Installation

Requires Python 3.12 and an NVIDIA driver that supports CUDA 12.8.

```bash
cd bench
uv sync --frozen            # recommended: installs exactly the locked environment into .venv
# or, with pip:
# pip install -e . --extra-index-url https://download.pytorch.org/whl/cu128
```

The environment is pinned to the one used for the paper (torch 2.11.0+cu128, transformers 5.13.1, numpy 2.3.5, ...; see `uv.lock`).
The pretrained scorers (LPIPS, CLIP ViT-L/14, ViCLIP, PickScore, DINOv2, NIQE, CoTracker3) download on first use, at the model revisions pinned in `moviedit_bench/models.py`.
ViCLIP is hosted in a **gated** Hugging Face repo: accept the terms at [OpenGVLab/ViCLIP](https://huggingface.co/OpenGVLab/ViCLIP) and run `hf auth login` first, or pass a local checkpoint with `--viclip-checkpoint`.

The VLM judge is [Gemma 4 31B](https://huggingface.co/google/gemma-4-31B-it) (bf16) served by [vLLM](https://github.com/vllm-project/vllm).
`scripts/serve_vlm.sh` creates its own locked environment (`vlm_server/uv.lock`: vLLM 0.25.1, torch 2.11.0+cu128) and pins the checkpoint revision used for the paper:

```bash
bash scripts/serve_vlm.sh            # http://localhost:8000/v1
```

Set `PORT` to serve on another port (e.g. `PORT=8100 bash scripts/serve_vlm.sh`), and pass the same address to the evaluator with `--vlm-base-url`.

### Hardware requirements

| Stage | GPU memory | Notes |
|---|---|---|
| `local` (LPIPS, MFS, CLIP, ViCLIP, DINO, NIQE, PickScore) | **24 GB** (1 GPU) | All scorers stay loaded (≈8 GB); peak ≈22.5 GB during CoTracker/DINO on a full shot. Measured on an RTX 3090. |
| `vlm`, paper setup | **≥ 80 GB** (1 GPU, TP=1) | bf16 weights take 59 GB. Paper numbers were produced with one TP=1 replica per 80GB-class GPU; reproduced on B200. |
| `vlm`, 4× 24 GB | 4 GPUs, `TP=4 GPU_MEM_UTIL=0.84` | ≈16 GB weights per GPU; verified on 4× RTX A5000. At the default 0.90 it runs out of memory after CUDA-graph capture. If the system `nvcc` is older than CUDA 12, also set `VLLM_USE_FLASHINFER_SAMPLER=0`; this does not change results (greedy decoding). |

The two stages can run on different machines: point `--vlm-base-url` at the server (e.g. through an SSH tunnel).

## Evaluation

With the VLM server running:

```bash
uv run python -m moviedit_bench.evaluate --mode v2v \
    --data-root data/ \
    --pred-dir /path/to/edited_videos \
    --out-dir outputs/my_method_v2v
```

Use `--mode iv2v` for IV2V outputs. Options:

- `--stages local` / `--stages vlm`: run only the GPU-model metrics or only the VLM-judge metrics. You can run them on different machines with the same `--out-dir`, since results are merged per scenario.
- Results are cached. Re-running computes only missing or failed metrics, and a finished run just prints the summary again; `--overwrite` recomputes everything.
- `--num-shards N --shard-id i`: split the scenarios across `N` parallel processes, e.g. one per GPU with `CUDA_VISIBLE_DEVICES`. On a fresh machine, start one process first and let it download the pretrained scorers; processes that download them at the same time can fail.
- `--vlm-base-url`, `--vlm-concurrency`: VLM endpoint and number of parallel requests.
- `--instruction-ids`: evaluate only the listed scenarios (for debugging).

### Outputs

The run prints a table in the paper's column order and writes:

- `summary.json`: the mean of each metric over all scenarios (`overall`), plus `by_task` and `by_category`. For IV2V it also has `first_frame_reference`, the 64-scenario subset whose reference is the first frame.
- `scenarios/{instruction_id}.json`: per-scenario scores and the raw VLM judgments.

A summary is marked **incomplete** if any scenario is missing or any metric failed. Only complete runs are comparable to the paper.

Metric keys and the paper's table columns:

| Key | Paper column | Stage |
|---|---|---|
| `non_subject_preservation/lpips` ↓ | Non-subject Preservation, LPIPS | local |
| `edit_localization/precision` ↑ | Edit Localization, Precision | vlm |
| `edit_localization/recall` ↑ | Edit Localization, Recall | vlm |
| `subject_preservation/mfs` ↑ | Subject Preservation, MFS | local |
| `subject_preservation/lpips` ↓ | Subject Preservation, LPIPS | local |
| `subject_preservation/vlm` ↑ | Subject Preservation, VLM | vlm |
| `cross_shot_consistency/vlm` ↑ | X-shot Edit Consistency, VLM | vlm |
| `edit_prompt_alignment/clip` ↑ | Edit Prompt Alignment, CLIP | local |
| `edit_prompt_alignment/viclip` ↑ | Edit Prompt Alignment, ViCLIP | local |
| `edit_prompt_alignment/vlm` ↑ | Edit Prompt Alignment, VLM | vlm |
| `edited_frame_fidelity/vlm` ↑ | Edited-frame Fidelity, VLM (IV2V only) | vlm |
| `video_quality/dino` ↑ | Video Quality, DINO | local |
| `video_quality/niqe` ↓ | Video Quality, NIQE | local |
| `video_quality/pickscore` ↑ | Video Quality, PickScore | local |
| `video_quality/vlm` ↑ | Video Quality, VLM | vlm |
| `overall_edit_success` ↑ | Overall Edit Success | vlm |

Not every metric is defined for every edit type (e.g. non-subject preservation does not apply to global edits). Each mean is taken over the scenarios where the metric is defined.

### Reproducing the paper's numbers

To check your setup, evaluate MoviEdit's own outputs and compare with the numbers reported in the paper.

**Download:** [`moviedit_reference.zip`](https://drive.google.com/file/d/1Ec072ZEkPjrksqoYq2BGxqm2oyZv4wnu/view?usp=sharing) (Google Drive, 616 MB). It contains MoviEdit's edited videos and the results of the paper's evaluation run. Unzip it inside `bench/`:

```bash
unzip moviedit_reference.zip         # creates moviedit_reference/
sha256sum moviedit_reference.zip     # 6d449215b1be0448b95caa2e9aa3e0dfb7449eaeebaf59bb80beddd71553f0d2
```

```
moviedit_reference/
├── v2v/                  # 128 edited videos, V2V
├── iv2v/                 # 128 edited videos, IV2V
└── reference_results/    # the paper's run, in the evaluator's output format
    ├── v2v/              # summary.json, scenarios/{instruction_id}.json (scores + raw VLM judgments)
    └── iv2v/
```

```bash
uv run python -m moviedit_bench.evaluate --mode v2v  --data-root data/ --pred-dir moviedit_reference/v2v  --out-dir outputs/moviedit_v2v
uv run python -m moviedit_bench.evaluate --mode iv2v --data-root data/ --pred-dir moviedit_reference/iv2v --out-dir outputs/moviedit_iv2v
```

Keep the evaluator's defaults (960×544, 361 frames, 9 px mask dilation, Gemma 4 31B at temperature 0). The `overall` row of your `summary.json` should match the first two rows below, and `first_frame_reference` the third:

| | Non-subj. LPIPS↓ | Prec.↑ | Rec.↑ | MFS↑ | Subj. LPIPS↓ | Subj. VLM↑ | X-shot VLM↑ | CLIP↑ | ViCLIP↑ | Align VLM↑ | Fidelity VLM↑ | DINO↑ | NIQE↓ | PickS.↑ | Quality VLM↑ | Overall↑ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| V2V | 0.0741 | 0.904 | 0.861 | 0.695 | 0.121 | 0.911 | 0.904 | 0.227 | 0.190 | 0.884 | – | 0.989 | 4.85 | 19.7 | 0.762 | 0.790 |
| IV2V | 0.0598 | 0.963 | 0.920 | 0.699 | 0.0996 | 0.995 | 0.933 | 0.229 | 0.192 | 0.925 | 0.805 | 0.989 | 4.78 | 19.7 | 0.807 | 0.786 |
| IV2V, first-frame reference | 0.0586 | 0.965 | 0.938 | 0.699 | 0.107 | 1.00 | 0.981 | 0.231 | 0.191 | 0.938 | 0.828 | 0.989 | 4.67 | 19.8 | 0.797 | 0.803 |

With the locked environments, the non-VLM metrics (LPIPS, MFS, CLIP, ViCLIP, DINO, NIQE, PickScore) reproduce these numbers to all reported digits. VLM verdicts depend slightly on the GPU and on request batching: re-running the judge on a B200 (TP=1) or on 4× A5000 (TP=4) flipped about 1% of the frame-level verdicts relative to the paper's run. This moves individual VLM metrics by up to ≈0.02 and overall edit success by up to ≈0.003.
