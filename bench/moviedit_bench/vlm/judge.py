"""VLM-judge metrics (paper Appx. C.4).

Three kinds of VLM queries per scenario:

1. Frame-pair judgments: 4 uniformly sampled frames per shot (first and last included), from
   *every* shot so that edits leaking into shots without the target are caught. Each query sees
   the source frame, the edited frame (and, for IV2V, the edited reference frame) and answers
   binary checks. Most VLM metrics are frame-level means of one of these checks:

       is_target_exist                     (i)   target visible in the source frame
       edit_applied                        (ii)  requested edit present      -> edit_prompt_alignment/vlm
       edit_on_correct_target              (iii) edit on the intended target
       non_target_preserved                (iv)  unrelated content kept      -> edit_localization/precision
       subject_other_attributes_preserved  (v)   other target attributes kept -> subject_preservation/vlm
       edit_complete                       (vi)  edit covers the whole target -> edit_localization/recall
       reference_consistent                (IV2V) consistent with the edited frame -> edited_frame_fidelity/vlm

   overall_edit_success is the fraction of frames passing every applicable check.
2. One cross-shot consistency query over edited frames from all target-bearing shots.
3. One visual-quality query (1-5 rating) over the middle frame of each shot.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from ..data import GLOBAL_TASKS, Scenario
from ..video import ScenarioVideos, read_image, to_rgb
from . import prompts
from .client import VLMClient, VLMOutputError

FRAMES_PER_SHOT = 4
QUALITY_MAX_SHOTS = 8


def uniform_frame_indices(duration: int, count: int = FRAMES_PER_SHOT) -> list[int]:
    if duration <= 0:
        return []
    if duration == 1:
        return [0] * count
    return sorted({int(round(x)) for x in np.linspace(0, duration - 1, count)})


def _frame_pair_keys(task: str) -> tuple[str, ...]:
    if task in GLOBAL_TASKS:
        return ("edit_applied", "edit_on_correct_target", "non_target_preserved")
    keys = ("is_target_exist", "edit_applied", "edit_on_correct_target", "non_target_preserved")
    if task in ("add", "appearance_change"):
        keys += ("subject_other_attributes_preserved",)
    return keys + ("edit_complete",)


def _target_subject(s: Scenario) -> str:
    return "(the whole scene)" if s.is_global else f"{s.target_type} ({s.target_id})"


def _frame_pair_messages(s: Scenario, shot_index: int, frame_index: int, src_url: str, tgt_url: str, ref_url: str | None):
    system = {
        "remove": prompts.frame_pair_remove,
        "swap": prompts.frame_pair_swap,
        "environment_change": prompts.frame_pair_environment_change,
        "stylization": prompts.frame_pair_stylization,
    }.get(s.task, prompts.frame_pair_default)()

    reference_block = ""
    if ref_url is not None:
        reference_block = """
Image 3 is REFERENCE -- the single edited frame this video's generation was seeded from
(produced by an image-editing step, before the video model propagated/generated the rest of
the video from it).

Also answer one more independent yes/no check, in addition to the ones above:

reference_consistent:
Does this frame pair's TARGET content remain consistent with what REFERENCE shows -- same
edited subject/scene identity and attributes -- allowing for natural viewpoint, pose, lighting,
or motion differences from camera movement and time passing across the shot?

Include "reference_consistent": true/false in the same JSON response, alongside the checks above.
"""
    num_checks = "three" if s.is_global else "five" if s.task in ("add", "appearance_change") else "four"
    user_text = f"""\
Editing task: {s.task}
Editing prompt: {s.edit_instruction}
Target subject: {_target_subject(s)}
Edit detail / suggestion: (see editing prompt)
Shot index: {shot_index}
Frame index in shot: {frame_index}
Pair label: shot{shot_index}_f{frame_index}

Image 1 is SOURCE.
Image 2 is TARGET.
{reference_block}
Evaluate the {num_checks} checks for this single frame pair only.
"""
    content = [
        {"type": "text", "text": user_text},
        {"type": "text", "text": "SOURCE (before):"},
        {"type": "image_url", "image_url": {"url": src_url}},
        {"type": "text", "text": "TARGET (after):"},
        {"type": "image_url", "image_url": {"url": tgt_url}},
    ]
    if ref_url is not None:
        content += [
            {"type": "text", "text": "REFERENCE (generation seed):"},
            {"type": "image_url", "image_url": {"url": ref_url}},
        ]
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def _parse_verdict(obj: dict, keys: tuple[str, ...]) -> dict[str, Any]:
    verdict: dict[str, Any] = {}
    for key in keys:
        if not isinstance(obj.get(key), bool):
            raise ValueError(f"missing or non-bool {key!r} in VLM response")
        verdict[key] = obj[key]
    for key in ("confidence", "notes"):
        if isinstance(obj.get(key), str) and obj[key].strip():
            verdict[key] = obj[key].strip()
    return verdict


def _finalize(verdict: dict[str, Any], task: str) -> dict[str, Any]:
    """Direction-align the checks so that True always means "good".

    In frames where the target is absent from the source, nothing should change: any visible
    edit is off-target, so (ii)-(iv) become `not edit_applied`, and (v)/(vi) are undefined.
    For global edits, check (iii) asks whether the effect covers the whole scene (-> recall) and
    check (iv) whether the underlying scene content is preserved (-> subject preservation).
    """
    if verdict.get("is_target_exist") is False:
        ok = not verdict["edit_applied"]
        verdict["edit_applied_aligned"] = ok
        for key in ("edit_on_correct_target", "non_target_preserved"):
            verdict[key] = ok
        for key in ("subject_other_attributes_preserved", "edit_complete"):
            if key in verdict:
                verdict[key] = None
    else:
        verdict["edit_applied_aligned"] = verdict["edit_applied"]
    if task in GLOBAL_TASKS:
        verdict["edit_complete"] = verdict["edit_on_correct_target"]
        verdict["subject_other_attributes_preserved"] = verdict["non_target_preserved"]
    return verdict


def frame_success(verdict: dict[str, Any]) -> bool | None:
    """True iff every applicable check passed; None if the judgment itself failed."""
    required = [verdict.get(k) for k in ("edit_applied_aligned", "edit_on_correct_target", "non_target_preserved")]
    if not all(isinstance(v, bool) for v in required):
        return None
    optional = ("subject_other_attributes_preserved", "edit_complete", "reference_consistent")
    return all(required) and not any(verdict.get(k) is False for k in optional)


def _failure(e: Exception) -> dict[str, Any]:
    """Log entry for a failed query. Invalid answers (unparseable JSON, missing checks) are part of
    the protocol and simply excluded; anything else (e.g. server unreachable) should be retried."""
    return {"error": str(e)[:500], "invalid_answer": isinstance(e, (VLMOutputError, ValueError))}


def _mean(values: list) -> float | None:
    values = [float(v) for v in values if isinstance(v, bool)]
    return float(np.mean(values)) if values else None


class VLMJudge:
    def __init__(self, client: VLMClient, concurrency: int = 8):
        self.client = client
        self.concurrency = concurrency

    def frame_pairs(self, s: Scenario, v: ScenarioVideos, use_reference: bool) -> list[dict[str, Any]]:
        ref_url = None
        if use_reference and s.reference_frame is not None:
            ref_url = self.client.encode_image(to_rgb([read_image(s.reference_frame, v.size)])[0])
        keys = _frame_pair_keys(s.task) + (("reference_consistent",) if ref_url else ())
        target_shots = set(s.target_shots)

        jobs = []
        for shot_index, shot in enumerate(s.shots):
            start, end = v.shot_range(shot)
            n = min(len(v.source[start:end]), len(v.edited[start:end]))
            for idx in uniform_frame_indices(end - start):
                if idx < n:
                    jobs.append((shot_index, idx, start + idx, shot_index in target_shots))

        def judge(job):
            shot_index, idx, frame, in_target_shot = job
            src_url, tgt_url = (self.client.encode_image(f) for f in to_rgb([v.source[frame], v.edited[frame]]))
            try:
                obj = self.client.chat_json(_frame_pair_messages(s, shot_index, idx, src_url, tgt_url, ref_url))
                verdict = _finalize(_parse_verdict(obj, keys), s.task)
            except Exception as e:  # noqa: BLE001 -- a failed judgment is excluded, not fatal
                verdict = _failure(e)
            return {**verdict, "shot_index": shot_index, "frame_index_in_shot": idx, "frame": frame, "in_target_shot": in_target_shot}

        with ThreadPoolExecutor(self.concurrency) as pool:
            return list(pool.map(judge, jobs))

    def cross_shot_consistency(self, s: Scenario, v: ScenarioVideos, verdicts: list[dict[str, Any]]) -> dict | None:
        """One query over edited frames of all target-bearing shots (for swaps: only frames where
        the target is visible). Undefined for removals and single-shot targets."""
        if s.task == "remove" or len(s.target_shots) < 2:
            return None
        usable = [
            x for x in verdicts
            if x["in_target_shot"] and (s.task != "swap" or x.get("is_target_exist") is True)
        ]
        if len(usable) < 2:
            return None

        n = len(usable)
        system = (
            prompts.consistency_swap if s.task == "swap"
            else prompts.consistency_global if s.is_global
            else prompts.consistency_default
        )(num_frames=n)
        content = [{
            "type": "text",
            "text": (
                f"Editing task: {s.task}\n"
                f"Editing prompt: {s.edit_instruction}\n"
                f"Target subject: {_target_subject(s)}\n"
                f"Edit detail / suggestion: (see editing prompt)\n"
                f"Number of TARGET frames: {n}\n\n"
                "CONSISTENCY ONLY: compare how the edited TARGET looks across shots. "
                "Do NOT judge whether the edit was applied correctly, on the right target, "
                "or whether it is a good edit."
            ),
        }]
        for x in usable:
            content.append({"type": "text", "text": f"shot{x['shot_index']}_f{x['frame_index_in_shot']} (TARGET):"})
            content.append({"type": "image_url", "image_url": {"url": self.client.encode_image(to_rgb([v.edited[x['frame']]])[0])}})
        obj = self.client.chat_json([{"role": "system", "content": system}, {"role": "user", "content": content}])
        return _parse_verdict(obj, ("editing_consistent",))

    def video_quality(self, s: Scenario, v: ScenarioVideos) -> dict | None:
        """1-5 visual-quality rating from the middle frame of each shot."""
        frames = []
        for shot in s.shots[:QUALITY_MAX_SHOTS]:
            start, end = v.shot_range(shot)
            mid = start + (end - start) // 2
            if end > start and mid < len(v.edited):
                frames.append(v.edited[mid])
        if not frames:
            return None
        content: list[dict] = [{"type": "text", "text": f"Frame samples ({len(frames)} total, from different shots):"}]
        for i, f in enumerate(to_rgb(frames)):
            content.append({"type": "text", "text": f"Frame {i}:"})
            content.append({"type": "image_url", "image_url": {"url": self.client.encode_image(f)}})
        return self.client.chat_json([
            {"role": "system", "content": prompts.quality(num_frames=len(frames))},
            {"role": "user", "content": content},
        ])

    def evaluate(self, s: Scenario, v: ScenarioVideos, metrics: list[str], use_reference: bool) -> tuple[dict, dict]:
        """Returns ({metric: score}, raw judgments for logging) for the requested VLM metrics."""
        scores: dict[str, float | None] = {}
        logs: dict[str, Any] = {}

        verdicts = self.frame_pairs(s, v, use_reference)
        logs["frame_pairs"] = verdicts
        from_frames = {
            "edit_localization/precision": "non_target_preserved",
            "edit_localization/recall": "edit_complete",
            "subject_preservation/vlm": "subject_other_attributes_preserved",
            "edit_prompt_alignment/vlm": "edit_applied_aligned",
            "edited_frame_fidelity/vlm": "reference_consistent",
        }
        for name, key in from_frames.items():
            if name in metrics:
                scores[name] = _mean([x.get(key) for x in verdicts])
        if "overall_edit_success" in metrics:
            scores["overall_edit_success"] = _mean([frame_success(x) for x in verdicts])

        if "cross_shot_consistency/vlm" in metrics:
            try:
                verdict = self.cross_shot_consistency(s, v, verdicts)
                logs["cross_shot_consistency"] = verdict
                scores["cross_shot_consistency/vlm"] = None if verdict is None else float(verdict["editing_consistent"])
            except Exception as e:  # noqa: BLE001
                logs["cross_shot_consistency"] = _failure(e)
                scores["cross_shot_consistency/vlm"] = None

        if "video_quality/vlm" in metrics:
            try:
                obj = self.video_quality(s, v)
                logs["video_quality"] = obj
                raw = None if obj is None else obj.get("quality_score")
                scores["video_quality/vlm"] = (float(raw) - 1.0) / 4.0 if isinstance(raw, (int, float)) and 1 <= raw <= 5 else None
            except Exception as e:  # noqa: BLE001
                logs["video_quality"] = _failure(e)
                scores["video_quality/vlm"] = None

        return scores, logs
