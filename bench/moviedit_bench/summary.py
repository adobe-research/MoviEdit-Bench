"""Aggregates per-scenario results into the benchmark summary (paper Tables 1-2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .tasks import METRICS

TABLE_COLUMNS = [
    ("Non-subj. Pres.", "LPIPS", "non_subject_preservation/lpips"),
    ("Edit Loc.", "Prec.", "edit_localization/precision"),
    ("Edit Loc.", "Rec.", "edit_localization/recall"),
    ("Subj. Pres.", "MFS", "subject_preservation/mfs"),
    ("Subj. Pres.", "LPIPS", "subject_preservation/lpips"),
    ("Subj. Pres.", "VLM", "subject_preservation/vlm"),
    ("X-shot Cons.", "VLM", "cross_shot_consistency/vlm"),
    ("Prompt Align.", "CLIP", "edit_prompt_alignment/clip"),
    ("Prompt Align.", "ViCLIP", "edit_prompt_alignment/viclip"),
    ("Prompt Align.", "VLM", "edit_prompt_alignment/vlm"),
    ("Frame Fid.", "VLM", "edited_frame_fidelity/vlm"),
    ("Video Qual.", "DINO", "video_quality/dino"),
    ("Video Qual.", "NIQE", "video_quality/niqe"),
    ("Video Qual.", "PickS.", "video_quality/pickscore"),
    ("Video Qual.", "VLM", "video_quality/vlm"),
    ("Overall", "Success", "overall_edit_success"),
]


def _means(records: list[dict]) -> dict[str, dict]:
    out = {}
    for name in METRICS:
        values = [r["scores"][name] for r in records if r["scores"].get(name) is not None]
        if values:
            out[name] = {"mean": float(np.mean(values)), "n": len(values)}
    return out


def _group(records: list[dict], key: str) -> dict[str, dict]:
    groups: dict[str, list[dict]] = {}
    for r in records:
        groups.setdefault(r[key], []).append(r)
    return {k: _means(v) for k, v in sorted(groups.items())}


def build_summary(records: list[dict], *, mode: str, num_scenarios: int, config: dict) -> dict:
    missing = sorted(r["instruction_id"] for r in records if r["status"] == "missing")
    errors = {r["instruction_id"]: r["errors"] for r in records if r.get("errors")}
    scored = [r for r in records if r["status"] == "ok"]
    summary = {
        "mode": mode,
        "config": config,
        "coverage": {
            "num_scenarios": num_scenarios,
            "evaluated": len(scored),
            "missing": missing,
            "errors": errors,
            "complete": len(scored) == num_scenarios and not errors,
        },
        "overall": _means(scored),
        "by_task": _group(scored, "task"),
        "by_category": _group(scored, "category"),
    }
    if mode == "iv2v":
        # The paper additionally reports IV2V on the first-frame-reference subset (Table 2, "First").
        summary["first_frame_reference"] = _means([r for r in scored if r.get("reference_frame_index") == 0])
    return summary


def format_table(summary: dict, subsets: list[str] | None = None) -> str:
    columns = [c for c in TABLE_COLUMNS if summary["mode"] == "iv2v" or c[2] != "edited_frame_fidelity/vlm"]
    subsets = subsets or (["overall", "first_frame_reference"] if summary["mode"] == "iv2v" else ["overall"])
    header = f"{'':<22}" + "".join(f"{c[1] + ('↓' if METRICS[c[2]][1] else '↑'):>9}" for c in columns)
    lines = [header]
    for subset in subsets:
        row = summary.get(subset, {})
        cells = []
        for _, _, name in columns:
            m = row.get(name)
            digits = 2 if name in ("video_quality/niqe", "video_quality/pickscore") else 4
            cells.append(f"{m['mean']:>9.{digits}f}" if m else f"{'-':>9}")
        lines.append(f"{subset:<22}" + "".join(cells))
    cov = summary["coverage"]
    lines.append(f"evaluated {cov['evaluated']}/{cov['num_scenarios']} scenarios"
                 + ("" if cov["complete"] else "  [INCOMPLETE -- not comparable to the paper's numbers]"))
    return "\n".join(lines)


def load_records(results_dir: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(results_dir.glob("*.json"))]
