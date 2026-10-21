"""Minimal client for an OpenAI-compatible chat-completions endpoint (e.g. `vllm serve`)."""

from __future__ import annotations

import base64
import io
import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

import numpy as np
from PIL import Image


def parse_json_object(text: str) -> dict:
    """First top-level {...} object in the model output (tolerates code fences / trailing text)."""
    t = text.strip()
    m = re.match(r"^```(?:json)?\s*\n?(.*?)\n?```\s*$", t, re.DOTALL | re.IGNORECASE)
    t = m.group(1).strip() if m else t
    start = t.find("{")
    if start < 0:
        raise ValueError("no JSON object in model output")
    depth, in_str, esc, quote = 0, False, False, ""
    for i, ch in enumerate(t[start:], start):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == quote:
                in_str = False
        elif ch in "\"'":
            in_str, quote = True, ch
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start : i + 1])
    raise ValueError("unclosed JSON object in model output")


class VLMOutputError(RuntimeError):
    """The judge answered, but not with the expected JSON (e.g. truncated at max_tokens). This is
    deterministic at temperature 0, so the judgment is excluded rather than retried later, exactly as
    in the paper's evaluation."""


class VLMClient:
    def __init__(
        self, base_url: str, model: str, api_key: str = "EMPTY", *, temperature: float = 0.0,
        max_tokens: int = 512, timeout: float = 300.0, max_retries: int = 5, image_max_side: int = 768,
    ):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_retries = max_retries
        self.image_max_side = image_max_side

    def check_server(self) -> None:
        """Fails fast if the endpoint is unreachable or does not serve `self.model`."""
        req = urllib.request.Request(self.url.rsplit("/chat/completions", 1)[0] + "/models",
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                served = [m.get("id") for m in json.loads(resp.read().decode("utf-8")).get("data", [])]
        except (urllib.error.URLError, OSError) as e:
            raise RuntimeError(f"cannot reach the VLM server at {self.url}: {e} (see scripts/serve_vlm.sh)") from e
        if served and self.model not in served:
            raise RuntimeError(f"VLM server does not serve {self.model!r} (serves: {served})")

    def encode_image(self, rgb: np.ndarray) -> str:
        """RGB frame -> JPEG data URL, longest side capped at image_max_side."""
        im = Image.fromarray(rgb, mode="RGB")
        w, h = im.size
        if max(w, h) > self.image_max_side:
            scale = self.image_max_side / max(w, h)
            im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))), resample=Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=92)
        return "data:image/jpeg;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")

    def _post(self, messages: list[dict[str, Any]]) -> str:
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": self.max_tokens}
        req = urllib.request.Request(
            self.url, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace") if e.fp else ""
            raise RuntimeError(f"HTTP {e.code} {e.reason}: {detail[:2000]}") from e
        content = ((payload.get("choices") or [{}])[0].get("message") or {}).get("content")
        if not isinstance(content, str):
            raise RuntimeError(f"unexpected response: {str(payload)[:2000]}")
        return content

    def chat_json(self, messages: list[dict[str, Any]]) -> dict:
        """Sends `messages` and returns the parsed JSON answer, retrying transient/parse failures."""
        last_err: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return parse_json_object(self._post(messages))
            except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as e:
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2.0 * attempt, 30.0))
            except (json.JSONDecodeError, ValueError) as e:
                last_err = e
        error = VLMOutputError if isinstance(last_err, ValueError) else RuntimeError
        raise error(f"VLM call failed after {self.max_retries} attempts: {last_err}") from last_err
