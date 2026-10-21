"""Evaluate a model's edited videos on MoviEdit-Bench.

    python -m moviedit_bench.evaluate --mode v2v \\
        --data-root data/ --pred-dir /path/to/edited_videos --out-dir outputs/my_method

`--pred-dir` holds one `{instruction_id}.mp4` per scenario. Metrics run in two stages that can be
executed separately (e.g. on different machines): `local` (LPIPS / MFS / CLIP / ViCLIP / DINO /
NIQE / PickScore, needs one GPU) and `vlm` (queries an OpenAI-compatible VLM server). Results are
cached per scenario under `--out-dir/scenarios/`, so re-running only computes what is missing.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

from tqdm import tqdm

from .data import Scenario, load_scenarios
from .metrics import LOCAL_METRIC_FNS
from .summary import build_summary, format_table, load_records
from .tasks import LOCAL_METRICS, VLM_METRICS, applicable_metrics
from .video import ScenarioVideos


def _record(s: Scenario, out_path: Path) -> dict:
    if out_path.is_file():
        return json.loads(out_path.read_text())
    return {
        "instruction_id": s.instruction_id, "sample_id": s.sample_id, "category": s.category, "task": s.task,
        "reference_frame_index": s.reference_frame_index, "status": "ok", "scores": {}, "errors": {},
    }


def _write_json(obj: dict, path: Path) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def _pending(s: Scenario, args, record: dict) -> tuple[list[str], list[str]]:
    """(local metrics, VLM metrics) still to compute for this scenario: the ones of the selected
    stages that are not cached yet or previously failed. Empty if the edited video is missing."""
    if not (args.pred_dir / f"{s.instruction_id}.mp4").is_file():
        return [], []
    todo = [
        m for m in applicable_metrics(s.task, args.mode)
        if args.overwrite or m not in record["scores"] or m in record["errors"]
    ]
    todo_local = [m for m in todo if m in LOCAL_METRICS] if "local" in args.stages else []
    todo_vlm = [m for m in todo if m in VLM_METRICS] if "vlm" in args.stages else []
    return todo_local, todo_vlm


def evaluate_scenario(s: Scenario, args, record: dict, models, judge) -> dict:
    pred = args.pred_dir / f"{s.instruction_id}.mp4"
    if not pred.is_file():
        return {**record, "status": "missing"}
    record["status"] = "ok"

    wanted = applicable_metrics(s.task, args.mode)
    todo_local, todo_vlm = _pending(s, args, record)
    if not todo_local and not todo_vlm:
        return record

    v = ScenarioVideos(
        s, pred, size=(args.eval_width, args.eval_height), max_frames=args.num_frames, mask_dilate_px=args.mask_dilate_px,
    )
    record["num_frames_evaluated"] = v.num_frames
    if len(v.edited) < args.num_frames:
        tqdm.write(f"[warn] {s.instruction_id}: edited video has {len(v.edited)}/{args.num_frames} frames; "
                   "only the overlapping frames are evaluated")

    for name in todo_local:
        record["errors"].pop(name, None)
        try:
            record["scores"][name] = LOCAL_METRIC_FNS[name](s, v, models)
        except Exception as e:  # noqa: BLE001 -- record and keep going
            record["scores"][name] = None
            record["errors"][name] = f"{type(e).__name__}: {e}"[:1000]
            tqdm.write(f"[error] {s.instruction_id} {name}: {e}")
            if args.debug:
                traceback.print_exc()

    if todo_vlm:
        for name in todo_vlm:
            record["errors"].pop(name, None)
        try:
            scores, logs = judge.evaluate(s, v, todo_vlm, use_reference=args.mode == "iv2v")
            record["scores"].update({name: scores.get(name) for name in todo_vlm})
            record.setdefault("vlm_logs", {}).update(logs)
            # Failed VLM calls are excluded from the means. Invalid answers are part of the protocol;
            # other failures (e.g. connection errors) are flagged so that a re-run retries them.
            failed = sum("error" in x and not x["invalid_answer"] for x in logs["frame_pairs"])
            if failed:
                msg = f"{failed}/{len(logs['frame_pairs'])} frame-pair VLM calls failed"
                tqdm.write(f"[warn] {s.instruction_id}: {msg}")
                for name in todo_vlm:
                    if name not in ("cross_shot_consistency/vlm", "video_quality/vlm"):
                        record["errors"][name] = msg
            for name in ("cross_shot_consistency/vlm", "video_quality/vlm"):
                log = logs.get(name.split("/")[0])
                if isinstance(log, dict) and "error" in log and not log["invalid_answer"]:
                    record["errors"][name] = log["error"]
        except Exception as e:  # noqa: BLE001
            for name in todo_vlm:
                record["scores"][name] = None
                record["errors"][name] = f"{type(e).__name__}: {e}"[:1000]
            tqdm.write(f"[error] {s.instruction_id} vlm: {e}")
            if args.debug:
                traceback.print_exc()

    # Drop metrics that are not defined for this scenario (e.g. left over from another --mode).
    record["scores"] = {k: v for k, v in record["scores"].items() if k in wanted}
    return record


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["v2v", "iv2v"],
                    help="v2v: text-guided editing; iv2v: edited-frame-guided propagation")
    ap.add_argument("--data-root", type=Path, required=True,
                    help="benchmark directory (benchmark[_iv2v].jsonl, eval_metadata.json, source_videos/, masks/, ...)")
    ap.add_argument("--pred-dir", type=Path, required=True, help="directory of {instruction_id}.mp4 edited videos")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--stages", default="local,vlm", help="comma-separated subset of: local,vlm")
    ap.add_argument("--instruction-ids", default="", help="comma-separated instruction ids to evaluate (default: all)")
    ap.add_argument("--num-shards", type=int, default=1, help="split scenarios across N parallel processes")
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true", help="recompute metrics that are already cached")
    ap.add_argument("--debug", action="store_true")

    g = ap.add_argument_group("evaluation protocol (keep the defaults to compare with the paper)")
    g.add_argument("--eval-width", type=int, default=960)
    g.add_argument("--eval-height", type=int, default=544)
    g.add_argument("--num-frames", type=int, default=361)
    g.add_argument("--mask-dilate-px", type=int, default=9)

    g = ap.add_argument_group("local metrics")
    g.add_argument("--device", default=None, help="default: cuda if available")
    g.add_argument("--viclip-checkpoint", default=None,
                   help="local ViClip-InternVid-10M-FLT.pth (default: download from the gated OpenGVLab/ViCLIP HF repo)")

    g = ap.add_argument_group("VLM judge")
    g.add_argument("--vlm-base-url", default="http://localhost:8000/v1")
    g.add_argument("--vlm-model", default="google/gemma-4-31B-it")
    g.add_argument("--vlm-api-key", default=os.environ.get("VLM_API_KEY", "EMPTY"))
    g.add_argument("--vlm-concurrency", type=int, default=8, help="parallel requests per scenario")
    args = ap.parse_args()

    args.stages = {x.strip() for x in args.stages.split(",") if x.strip()}
    if not args.stages <= {"local", "vlm"}:
        ap.error("--stages must be a subset of local,vlm")

    scenarios = load_scenarios(args.data_root, args.mode)
    num_scenarios = len(scenarios)
    if args.instruction_ids:
        wanted = set(args.instruction_ids.split(","))
        scenarios = [s for s in scenarios if s.instruction_id in wanted]
    scenarios = scenarios[args.shard_id :: args.num_shards]

    results_dir = args.out_dir / "scenarios"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load the scorers / contact the VLM server only if something is left to compute, so that
    # re-running a finished evaluation just prints the summary.
    pending = [_pending(s, args, _record(s, results_dir / f"{s.instruction_id}.json")) for s in scenarios]
    models = judge = None
    if any(local for local, _ in pending):
        import torch

        from .models import Models

        models = Models(args.device or ("cuda" if torch.cuda.is_available() else "cpu"), args.viclip_checkpoint)
        models.load_all()
    if any(vlm for _, vlm in pending):
        from .vlm.client import VLMClient
        from .vlm.judge import VLMJudge

        client = VLMClient(args.vlm_base_url, args.vlm_model, args.vlm_api_key)
        client.check_server()
        judge = VLMJudge(client, concurrency=args.vlm_concurrency)

    for s in tqdm(scenarios, desc=f"MoviEdit-Bench ({args.mode})", unit="scenario"):
        path = results_dir / f"{s.instruction_id}.json"
        try:
            record = evaluate_scenario(s, args, _record(s, path), models, judge)
        except Exception as e:  # noqa: BLE001 -- e.g. an undecodable video
            record = {**_record(s, path), "status": "error"}
            record["errors"]["scenario"] = f"{type(e).__name__}: {e}"[:1000]
            tqdm.write(f"[error] {s.instruction_id}: {e}")
            if args.debug:
                traceback.print_exc()
        _write_json(record, path)

    config = {k: getattr(args, k) for k in ("eval_width", "eval_height", "num_frames", "mask_dilate_px", "vlm_model")}
    summary = build_summary(load_records(results_dir), mode=args.mode, num_scenarios=num_scenarios, config=config)
    _write_json(summary, args.out_dir / "summary.json")
    print(format_table(summary))
    print(f"wrote {args.out_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
