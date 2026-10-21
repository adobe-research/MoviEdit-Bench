"""Pretrained models used by the non-VLM metrics. Each is loaded lazily on first use."""

from __future__ import annotations

from functools import cached_property

import cv2
import numpy as np
import torch

# (repo, pinned revision) of every pretrained scorer, so results do not drift with upstream updates.
CLIP_MODEL = ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41")
PICKSCORE_MODEL = ("yuvalkirstain/PickScore_v1", "a4e4367c6dfa7288a00c550414478f865b875800")
PICKSCORE_PROCESSOR = ("laion/CLIP-ViT-H-14-laion2B-s32B-b79K", "1c2b8495b28150b8a4922ee1c8edee224c284c0c")
DINO_MODEL = ("facebook/dinov2-base", "f9e44c814b77203eaa57a6bdbbd535f21ede1415")
VICLIP_MODEL = ("OpenGVLab/ViCLIP", "67fbc34595ae6e5f56ab630fe421d3fc45c8670d")
VICLIP_FILE = "ViClip-InternVid-10M-FLT.pth"
COTRACKER_HUB = "facebookresearch/co-tracker:82e02e8029753ad4ef13cf06be7f4fc5facdda4d"


def _features(out) -> torch.Tensor:
    # transformers>=5 returns a ModelOutput from get_*_features; older versions return a tensor.
    return out if isinstance(out, torch.Tensor) else out.pooler_output


def _normalize(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


class LPIPS:
    def __init__(self, device: str):
        import lpips

        self.model = lpips.LPIPS(net="alex").eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, pairs: list[tuple[np.ndarray, np.ndarray]], batch_size: int = 8) -> list[float]:
        """pairs of RGB uint8 frames -> per-pair LPIPS distance."""
        out = []
        for i in range(0, len(pairs), batch_size):
            chunk = pairs[i : i + batch_size]
            a = torch.from_numpy(np.stack([p[0] for p in chunk])).permute(0, 3, 1, 2).float().div(255).to(self.device)
            b = torch.from_numpy(np.stack([p[1] for p in chunk])).permute(0, 3, 1, 2).float().div(255).to(self.device)
            out.extend(self.model(a, b, normalize=True).flatten().tolist())
        return out


class CLIP:
    def __init__(self, device: str):
        from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer

        repo, rev = CLIP_MODEL
        self.model = CLIPModel.from_pretrained(repo, revision=rev).eval().to(device)
        self.processor = CLIPImageProcessor.from_pretrained(repo, revision=rev)
        self.tokenizer = CLIPTokenizer.from_pretrained(repo, revision=rev)
        self.device = device

    @torch.no_grad()
    def encode_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images_rgb, return_tensors="pt").to(self.device)
        return _normalize(_features(self.model.get_image_features(**inputs))).cpu().numpy()

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        inputs = self.tokenizer([text], padding=True, truncation=True, return_tensors="pt").to(self.device)
        return _normalize(_features(self.model.get_text_features(**inputs))).cpu().numpy()[0]


class ViCLIP:
    """InternVideo's ViCLIP-L (vendored under third_party/viclip): fixed 8-frame, 224x224 clips."""

    NUM_FRAMES = 8
    _MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    _STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, device: str, checkpoint: str | None = None):
        from .third_party.viclip import ViCLIP as _ViCLIP

        if checkpoint is None:
            from huggingface_hub import hf_hub_download

            repo, rev = VICLIP_MODEL
            checkpoint = hf_hub_download(repo_id=repo, filename=VICLIP_FILE, revision=rev)  # gated repo
        self.model = _ViCLIP(tokenizer=None, size="l", pretrain=checkpoint, freeze_text=True).eval().to(device)
        self.device = device
        self._text_cache: dict = {}

    @torch.no_grad()
    def encode_clip(self, frames_rgb: list[np.ndarray]) -> np.ndarray:
        clip = np.stack([(cv2.resize(f, (224, 224)).astype(np.float32) / 255.0 - self._MEAN) / self._STD for f in frames_rgb])
        tensor = torch.from_numpy(clip).permute(0, 3, 1, 2)[None].float().to(self.device)  # (1, T, 3, H, W)
        return self.model.get_vid_features(tensor).cpu().numpy()[0]

    @torch.no_grad()
    def encode_text(self, text: str) -> np.ndarray:
        return self.model.get_text_features(text, self.model.tokenizer, self._text_cache).cpu().numpy()[0]


class PickScore:
    def __init__(self, device: str):
        from transformers import AutoModel, AutoProcessor

        self.processor = AutoProcessor.from_pretrained(PICKSCORE_PROCESSOR[0], revision=PICKSCORE_PROCESSOR[1])
        self.model = AutoModel.from_pretrained(PICKSCORE_MODEL[0], revision=PICKSCORE_MODEL[1]).eval().to(device)
        self.device = device

    @torch.no_grad()
    def __call__(self, frames_rgb: list[np.ndarray], text: str, batch_size: int = 8) -> list[float]:
        """Per-frame logit-scaled image-text similarity against one prompt."""
        t = self.processor(text=[text], padding=True, truncation=True, max_length=77, return_tensors="pt").to(self.device)
        text_emb = _normalize(_features(self.model.get_text_features(**t)))
        scores = []
        for i in range(0, len(frames_rgb), batch_size):
            x = self.processor(images=frames_rgb[i : i + batch_size], return_tensors="pt").to(self.device)
            img_emb = _normalize(_features(self.model.get_image_features(**x)))
            scores.extend((self.model.logit_scale.exp() * (img_emb @ text_emb.T)).flatten().tolist())
        return scores


class DINO:
    def __init__(self, device: str):
        from transformers import AutoImageProcessor, AutoModel

        repo, rev = DINO_MODEL
        self.processor = AutoImageProcessor.from_pretrained(repo, revision=rev)
        self.model = AutoModel.from_pretrained(repo, revision=rev).eval().to(device)
        self.device = device

    @torch.no_grad()
    def encode_images(self, images_rgb: list[np.ndarray]) -> np.ndarray:
        inputs = self.processor(images=images_rgb, return_tensors="pt").to(self.device)
        return _normalize(self.model(**inputs).pooler_output).cpu().numpy()


class NIQE:
    def __init__(self, device: str):
        import pyiqa

        self.metric = pyiqa.create_metric("niqe", device=device)
        self.device = device

    @torch.no_grad()
    def __call__(self, frames_rgb: list[np.ndarray], batch_size: int = 8) -> list[float]:
        out = []
        for i in range(0, len(frames_rgb), batch_size):
            x = torch.from_numpy(np.stack(frames_rgb[i : i + batch_size])).permute(0, 3, 1, 2).float().div(255)
            out.extend(self.metric(x.to(self.device)).flatten().tolist())
        return out


class CoTracker:
    def __init__(self, device: str):
        self.model = torch.hub.load(COTRACKER_HUB, "cotracker3_offline", trust_repo=True).eval().to(device)
        self.device = device

    @torch.no_grad()
    def track(self, frames_rgb: list[np.ndarray], grid_size: int, box_mask: np.ndarray) -> np.ndarray:
        """Tracks a grid of points sampled inside `box_mask`. Returns (N, T, 2) pixel trajectories."""
        video = torch.from_numpy(np.stack(frames_rgb)).permute(0, 3, 1, 2)[None].float().to(self.device)
        segm = torch.from_numpy(box_mask.astype(np.float32))[None, None].to(self.device)
        tracks, _ = self.model(video, grid_size=grid_size, segm_mask=segm)
        return tracks[0].permute(1, 0, 2).cpu().numpy()


class Models:
    def __init__(self, device: str, viclip_checkpoint: str | None = None):
        self.device = device
        self.viclip_checkpoint = viclip_checkpoint

    def load_all(self) -> None:
        """Loads (and downloads) every model up front, so e.g. missing access to the gated
        ViCLIP checkpoint fails immediately instead of after hours of evaluation."""
        for name in ("lpips", "clip", "viclip", "pickscore", "dino", "niqe", "cotracker"):
            getattr(self, name)

    @cached_property
    def lpips(self) -> LPIPS:
        return LPIPS(self.device)

    @cached_property
    def clip(self) -> CLIP:
        return CLIP(self.device)

    @cached_property
    def viclip(self) -> ViCLIP:
        return ViCLIP(self.device, self.viclip_checkpoint)

    @cached_property
    def pickscore(self) -> PickScore:
        return PickScore(self.device)

    @cached_property
    def dino(self) -> DINO:
        return DINO(self.device)

    @cached_property
    def niqe(self) -> NIQE:
        return NIQE(self.device)

    @cached_property
    def cotracker(self) -> CoTracker:
        return CoTracker(self.device)
