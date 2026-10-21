"""Video decoding onto the fixed evaluation canvas.

Every video (source, edited, mask) is resized to the same canvas (default 960x544) and truncated
to the benchmark length (361 frames), so methods with different output resolutions are scored on
identical inputs. Frames are aligned by index; a shorter edited video is evaluated only over the
frames it has.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _interp(src_w: int, src_h: int, dst_w: int, dst_h: int) -> int:
    return cv2.INTER_AREA if dst_w * dst_h < src_w * src_h else cv2.INTER_LINEAR


def resize(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    w, h = size
    return cv2.resize(frame, (w, h), interpolation=_interp(frame.shape[1], frame.shape[0], w, h))


def _decode(path: Path, max_frames: int):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {path}")
    try:
        for _ in range(max_frames):
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_video(path: Path, max_frames: int, size: tuple[int, int]) -> list[np.ndarray]:
    """BGR uint8 frames resized to `size` (w, h)."""
    return [resize(f, size) for f in _decode(path, max_frames)]


def read_mask(path: Path, max_frames: int, size: tuple[int, int], dilate_px: int) -> list[np.ndarray]:
    """Boolean (h, w) target masks, dilated by `dilate_px` to exclude ambiguous boundary pixels."""
    w, h = size
    kernel = None
    if dilate_px > 0:
        k = dilate_px if dilate_px % 2 == 1 else dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    masks = []
    for frame in _decode(path, max_frames):
        gray = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (w, h), interpolation=cv2.INTER_NEAREST)
        mask = gray > 127
        if kernel is not None:
            mask = cv2.dilate(mask.astype(np.uint8) * 255, kernel) > 127
        masks.append(mask)
    return masks


def read_image(path: Path, size: tuple[int, int]) -> np.ndarray:
    frame = cv2.imread(str(path))
    if frame is None:
        raise RuntimeError(f"failed to read image: {path}")
    return resize(frame, size)


def to_rgb(frames: list[np.ndarray]) -> list[np.ndarray]:
    return [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]


def sample_uniform(frames: list, count: int) -> list:
    """ViCLIP's own clip sampling (step = n // count); pads with the last frame for short shots."""
    n = len(frames)
    if n == 0:
        return []
    if n < count:
        return frames + [frames[-1]] * (count - n)
    return frames[:: n // count][:count]


class ScenarioVideos:
    """Source, edited, and mask frames of one scenario, decoded once and shared by all metrics."""

    def __init__(self, scenario, edited_path: Path, *, size: tuple[int, int], max_frames: int, mask_dilate_px: int):
        self.size = size
        self.source = read_video(scenario.source_video, max_frames, size)
        self.edited = read_video(edited_path, max_frames, size)
        self.masks = (
            read_mask(scenario.target_mask, max_frames, size, mask_dilate_px) if scenario.target_mask else None
        )
        # Frames scored by per-frame metrics: the overlap of every stream.
        lengths = [len(self.source), len(self.edited)] + ([len(self.masks)] if self.masks is not None else [])
        self.num_frames = min(lengths)

    def mask(self, i: int) -> np.ndarray:
        """Target mask of frame i; the whole frame for global edits (no mask)."""
        if self.masks is None:
            w, h = self.size
            return np.ones((h, w), dtype=bool)
        return self.masks[i]

    def mask_range(self, start: int, end: int) -> list[np.ndarray]:
        if self.masks is None:
            return [self.mask(0)] * max(0, end - start)
        return self.masks[start:end]

    def shot_range(self, shot: tuple[int, int]) -> tuple[int, int]:
        """[start, end) of a shot, clipped to the edited video's length (end <= start if the shot is missing)."""
        return shot[0], min(shot[1], len(self.edited))
