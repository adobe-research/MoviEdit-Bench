"""Loads MoviEdit-Bench into flat ``Scenario`` objects.

The benchmark directory separates what a model needs from what only the evaluation needs::

    data/
    ├── benchmark.jsonl         # inference inputs, V2V
    ├── benchmark_iv2v.jsonl    # inference inputs, IV2V (adds the edited reference frame)
    ├── source_videos/
    ├── edited_frames/          # IV2V reference frames
    ├── eval_metadata.json      # evaluation only, shared by V2V and IV2V
    └── masks/                  # evaluation only

One line of ``benchmark.jsonl`` / ``benchmark_iv2v.jsonl`` (paths relative to the data root)::

    {"instruction_id": "C1_4_A__remove__01", "sample_id": "C1_4_A", "task": "remove",
     "source_video": "source_videos/C1_4_A_....mp4", "edit_instruction": "Remove the man",
     "reference_frame_indices": [88],                                       # IV2V only
     "edited_frame_paths": {"88": "edited_frames/C1_4_A__remove__01_f88.png"}}  # IV2V only

``eval_metadata.json``::

    {"samples": {"C1_4_A": {"category": "C1", "shots": [[0, 96], [96, 188], ...]}},   # [start, end)
     "instructions": {"C1_4_A__remove__01": {
         "target": {"type": "character", "id": "char_a",
                    "mask": "masks/C1_4_A_char_a.mp4",   # null for global edits
                    "shots": [0, 1, 2, 3]},              # shots the target appears in
         "shot_prompts": ["<description of shot 0 after the edit>", ...],
         "target_prompt": "<description of the whole edited video>"}}}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

LOCAL_TASKS = ("add", "remove", "appearance_change", "swap")
GLOBAL_TASKS = ("environment_change", "stylization")
TASKS = LOCAL_TASKS + GLOBAL_TASKS

INFERENCE_FILES = {"v2v": "benchmark.jsonl", "iv2v": "benchmark_iv2v.jsonl"}
EVAL_FILE = "eval_metadata.json"


@dataclass(frozen=True)
class Scenario:
    instruction_id: str
    sample_id: str
    category: str
    task: str
    edit_instruction: str
    target_prompt: str  # input for methods conditioned on a target-video prompt; not used by the metrics
    source_video: Path
    shots: tuple[tuple[int, int], ...]  # every shot of the video, [start, end)
    shot_prompts: tuple[str, ...]  # per-shot description of the edited video, parallel to `shots`
    target_type: str  # character | animal | object | scene
    target_id: str | None
    target_mask: Path | None  # binary mask video of the edit target; None for global edits
    target_shots: tuple[int, ...]  # indices into `shots` where the target appears (all shots for global edits)
    reference_frame_index: int | None  # IV2V only
    reference_frame: Path | None  # IV2V only: edited version of the source frame at reference_frame_index

    @property
    def is_global(self) -> bool:
        return self.task in GLOBAL_TASKS


def load_scenarios(data_root: Path, mode: str) -> list[Scenario]:
    """Joins the inference file of `mode` ("v2v" or "iv2v") with the evaluation metadata."""
    with (data_root / INFERENCE_FILES[mode]).open(encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    meta = json.loads((data_root / EVAL_FILE).read_text(encoding="utf-8"))

    scenarios = []
    for row in rows:
        iid = row["instruction_id"]
        sample = meta["samples"][row["sample_id"]]
        ev = meta["instructions"][iid]
        shots = tuple((int(a), int(b)) for a, b in sample["shots"])
        target = ev["target"]
        is_global = row["task"] in GLOBAL_TASKS
        if row["task"] not in TASKS:
            raise ValueError(f"{iid}: unknown task {row['task']!r}")
        if len(ev["shot_prompts"]) != len(shots):
            raise ValueError(f"{iid}: shot_prompts must have one entry per shot")
        if not is_global and not target.get("mask"):
            raise ValueError(f"{iid}: local edits require a target mask")

        ref_index = ref_path = None
        if mode == "iv2v":
            (ref_index,) = row["reference_frame_indices"]
            ref_path = data_root / row["edited_frame_paths"][str(ref_index)]

        scenarios.append(Scenario(
            instruction_id=iid,
            sample_id=row["sample_id"],
            category=sample["category"],
            task=row["task"],
            edit_instruction=row["edit_instruction"],
            target_prompt=ev.get("target_prompt", ""),
            source_video=data_root / row["source_video"],
            shots=shots,
            shot_prompts=tuple(ev["shot_prompts"]),
            target_type=target["type"],
            target_id=target.get("id"),
            target_mask=None if is_global else data_root / target["mask"],
            target_shots=tuple(range(len(shots))) if is_global else tuple(sorted(target["shots"])),
            reference_frame_index=ref_index,
            reference_frame=ref_path,
        ))
    return scenarios
