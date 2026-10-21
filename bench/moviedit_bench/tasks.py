"""Metric names and their applicability per edit type (paper Appx. C.4, Table 9).

A metric is undefined for an edit type when its premise does not apply, e.g. there is no
"non-subject region" for a whole-scene edit and no persisting subject after a removal.
"""

from __future__ import annotations

from .data import GLOBAL_TASKS, LOCAL_TASKS, TASKS

_ALL = frozenset(TASKS)
_LOCAL = frozenset(LOCAL_TASKS)
_GLOBAL = frozenset(GLOBAL_TASKS)
_NOT_REMOVE = _ALL - {"remove"}

# name -> (tasks it is defined for, lower_is_better). Order = paper table column order.
METRICS: dict[str, tuple[frozenset[str], bool]] = {
    "non_subject_preservation/lpips": (_LOCAL, True),
    "edit_localization/precision": (_LOCAL, False),
    "edit_localization/recall": (_ALL, False),
    "subject_preservation/mfs": (_NOT_REMOVE, False),
    "subject_preservation/lpips": (frozenset({"add", "appearance_change"}), True),
    "subject_preservation/vlm": (frozenset({"add", "appearance_change"}) | _GLOBAL, False),
    "cross_shot_consistency/vlm": (_NOT_REMOVE, False),
    "edit_prompt_alignment/clip": (_ALL, False),
    "edit_prompt_alignment/viclip": (_ALL, False),
    "edit_prompt_alignment/vlm": (_ALL, False),
    "edited_frame_fidelity/vlm": (_NOT_REMOVE, False),  # IV2V only
    "video_quality/dino": (_ALL, False),
    "video_quality/niqe": (_ALL, True),
    "video_quality/pickscore": (_ALL, False),
    "video_quality/vlm": (_ALL, False),
    "overall_edit_success": (_ALL, False),
}

VLM_METRICS = frozenset(name for name in METRICS if name.endswith("/vlm") or name.startswith("edit_localization/")) | {
    "overall_edit_success"
}
LOCAL_METRICS = frozenset(METRICS) - VLM_METRICS  # computed with locally-loaded models


def applicable_metrics(task: str, mode: str) -> list[str]:
    """Metrics defined for `task` under `mode` ("v2v" or "iv2v")."""
    return [
        name for name, (tasks, _) in METRICS.items()
        if task in tasks and (mode == "iv2v" or name != "edited_frame_fidelity/vlm")
    ]
