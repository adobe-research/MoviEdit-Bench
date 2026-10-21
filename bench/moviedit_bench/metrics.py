"""Non-VLM metrics (pixel-, embedding-, and tracking-based). Each returns one score per scenario,
or None when nothing could be scored (e.g. the target is never visible)."""

from __future__ import annotations

import numpy as np

from .data import Scenario
from .models import Models, ViCLIP
from .video import ScenarioVideos, sample_uniform, to_rgb

MFS_GRID_SIZE = 20


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


# ── per-frame metrics ────────────────────────────────────────────────────────────────────────

def non_subject_lpips(v: ScenarioVideos, models: Models) -> float | None:
    """LPIPS between source and edited frames with the (dilated) target region blacked out."""
    pairs = []
    for i in range(v.num_frames):
        mask = v.mask(i)
        if mask.all():
            continue
        src, tgt = to_rgb([v.source[i], v.edited[i]])
        src[mask] = 0
        tgt[mask] = 0
        pairs.append((src, tgt))
    return _mean(models.lpips(pairs))


def subject_lpips(v: ScenarioVideos, models: Models) -> float | None:
    """LPIPS between source and edited frames, over frames where the target is visible."""
    pairs = [tuple(to_rgb([v.source[i], v.edited[i]])) for i in range(v.num_frames) if v.mask(i).any()]
    return _mean(models.lpips(pairs))


def niqe(v: ScenarioVideos, models: Models) -> float | None:
    return _mean(models.niqe(to_rgb(v.edited[: v.num_frames])))


# ── shot-level metrics ───────────────────────────────────────────────────────────────────────

def _motion_fidelity(tracks_edit: np.ndarray, tracks_orig: np.ndarray) -> float | None:
    """Motion Fidelity Score (Yatim et al., 2024), following the reference implementation."""
    if tracks_edit.shape[1] < 2 or tracks_orig.shape[1] < 2:
        return None

    def directions(tracks):
        d = tracks[:, 1:] - tracks[:, :-1]
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        norm[norm == 0] = 1e-8
        return d / norm

    d_edit, d_orig = directions(tracks_edit), directions(tracks_orig)
    sim = np.einsum("ntc,mtc->nm", d_edit, d_orig) / d_edit.shape[1]
    n = min(sim.shape)
    sim[np.arange(n), np.arange(n)] -= 1.0
    return float(sim.max(axis=1).mean())


def mfs(s: Scenario, v: ScenarioVideos, models: Models) -> float | None:
    """Per target-bearing shot: track a grid inside the target's bounding box in both videos and
    compare frame-to-frame motion directions; averaged over shots."""
    scores = []
    for shot_idx in s.target_shots:
        start, end = v.shot_range(s.shots[shot_idx])
        src, tgt = v.source[start:end], v.edited[start:end]
        n = min(len(src), len(tgt))
        if n < 2:
            continue
        masks = v.mask_range(start, start + n)
        if len(masks) < n or not any(m.any() for m in masks):
            continue
        union = np.logical_or.reduce(masks)
        ys, xs = np.nonzero(union)
        box = np.zeros_like(union)
        box[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1] = True

        src_tracks = models.cotracker.track(to_rgb(src[:n]), MFS_GRID_SIZE, box)
        tgt_tracks = models.cotracker.track(to_rgb(tgt[:n]), MFS_GRID_SIZE, box)
        if len(src_tracks) == 0 or len(tgt_tracks) == 0:
            continue
        score = _motion_fidelity(tgt_tracks, src_tracks)
        if score is not None:
            scores.append(score)
    return _mean(scores)


def _shot_clips(s: Scenario, v: ScenarioVideos):
    """(shot text, 8 uniformly sampled RGB frames of the edited shot) for every shot."""
    for shot, text in zip(s.shots, s.shot_prompts):
        start, end = v.shot_range(shot)
        frames = v.edited[start:end]
        if frames:
            yield text, sample_uniform(to_rgb(frames), ViCLIP.NUM_FRAMES)


def clip_text_alignment(s: Scenario, v: ScenarioVideos, models: Models) -> float | None:
    """Mean CLIP image-text similarity between each shot's frames and its target description."""
    scores = [float((models.clip.encode_images(clip) @ models.clip.encode_text(text)).mean()) for text, clip in _shot_clips(s, v)]
    return _mean(scores)


def viclip_text_alignment(s: Scenario, v: ScenarioVideos, models: Models) -> float | None:
    """ViCLIP video-text similarity between each shot clip and its target description."""
    scores = [float(np.dot(models.viclip.encode_clip(clip), models.viclip.encode_text(text))) for text, clip in _shot_clips(s, v)]
    return _mean(scores)


def pickscore(s: Scenario, v: ScenarioVideos, models: Models) -> float | None:
    """PickScore of every edited frame against its shot's target description (pooled over frames)."""
    scores = []
    for shot, text in zip(s.shots, s.shot_prompts):
        start, end = v.shot_range(shot)
        if end > start:
            scores.extend(models.pickscore(to_rgb(v.edited[start:end]), text))
    return _mean(scores)


def dino_temporal_consistency(s: Scenario, v: ScenarioVideos, models: Models) -> float | None:
    """Mean DINOv2 cosine similarity of adjacent edited frames within each shot (never across cuts)."""
    sims = []
    for shot in s.shots:
        start, end = v.shot_range(shot)
        frames = v.edited[start:end]
        if len(frames) >= 2:
            feats = models.dino.encode_images(to_rgb(frames))
            sims.extend((feats[:-1] * feats[1:]).sum(axis=-1).tolist())
    return _mean(sims)


LOCAL_METRIC_FNS = {
    "non_subject_preservation/lpips": lambda s, v, m: non_subject_lpips(v, m),
    "subject_preservation/mfs": mfs,
    "subject_preservation/lpips": lambda s, v, m: subject_lpips(v, m),
    "edit_prompt_alignment/clip": clip_text_alignment,
    "edit_prompt_alignment/viclip": viclip_text_alignment,
    "video_quality/dino": dino_temporal_consistency,
    "video_quality/niqe": lambda s, v, m: niqe(v, m),
    "video_quality/pickscore": pickscore,
}
