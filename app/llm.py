"""Local LLM layer — in-process Qwen2.5 via llama-cpp-python.

Design intent
-------------
The LLM is used ONLY for semantic work the design assigns to it: sentiment
analysis and natural-language decision summaries. It is explicitly *not* allowed
to do math (that's the deterministic engine). To make the LLM's output usable by
deterministic downstream code, we provide ``structured()`` which forces JSON and
validates it.

Testability
-----------
The heavy 9 GB model is never required to test the rest of the system. We:
  * load the model LAZILY (only on first real call), and
  * allow a ``backend`` callable to be injected — a fake that maps messages →
    string — so nodes and JSON parsing can be unit-tested instantly.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from app.config import settings

logger = logging.getLogger("local-llm")

# A backend takes a list of chat messages (+ kwargs) and returns the raw text.
Message = Dict[str, str]
Backend = Callable[[List[Message]], str]


class ModelNotAvailableError(RuntimeError):
    """Raised when neither a real model nor an injected backend is usable."""


def _extract_json(text: str) -> Dict[str, Any]:
    """Best-effort parse of a JSON object from an LLM response.

    Handles ```json fenced blocks and stray prose around the object by grabbing
    the outermost ``{...}`` span. Raises ValueError if nothing parses.
    """
    if not text or not text.strip():
        raise ValueError("Empty LLM response.")
    cleaned = text.strip()
    # Strip markdown code fences if present.
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.DOTALL)
    if fence:
        cleaned = fence.group(1).strip()
    # Try the whole thing first, then the outermost brace span.
    for candidate in (cleaned, _outermost_braces(cleaned)):
        if candidate is None:
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")


def _outermost_braces(text: str) -> Optional[str]:
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return None


class LocalLLM:
    def __init__(
        self,
        *,
        model_path: Optional[str] = None,
        n_ctx: Optional[int] = None,
        n_gpu_layers: Optional[int] = None,
        temperature: Optional[float] = None,
        backend: Optional[Backend] = None,
    ) -> None:
        self.model_path = model_path or settings.llm_model_path
        self.n_ctx = n_ctx if n_ctx is not None else settings.llm_context_size
        self.n_gpu_layers = n_gpu_layers if n_gpu_layers is not None else settings.llm_gpu_layers
        self.temperature = temperature if temperature is not None else settings.llm_temperature
        self._backend = backend          # injected fake, or None → load real model
        self._llm = None                 # lazily-loaded llama_cpp.Llama instance

    # ── loading ───────────────────────────────────────────────────────
    def _ensure_loaded(self) -> None:
        if self._backend is not None or self._llm is not None:
            return
        if not Path(self.model_path).exists():
            raise ModelNotAvailableError(
                f"Model file not found at {self.model_path}. Set LLM_MODEL_PATH in .env "
                f"or inject a backend for testing."
            )
        try:
            from llama_cpp import Llama  # heavy import, done lazily
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailableError(
                "llama-cpp-python is not installed. `pip install llama-cpp-python`."
            ) from exc
        logger.info("Loading local LLM from %s (n_ctx=%s, gpu_layers=%s)",
                    self.model_path, self.n_ctx, self.n_gpu_layers)
        self._llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )

    @property
    def is_loaded(self) -> bool:
        return self._llm is not None

    # ── core completion ─────────────────────────────────────────────--
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: Optional[float] = None,
        max_tokens: int = 768,
    ) -> str:
        """Run a chat completion and return the assistant's raw text."""
        msgs = list(messages)
        if self._backend is not None:
            return self._backend(msgs)
        self._ensure_loaded()
        assert self._llm is not None
        result = self._llm.create_chat_completion(
            messages=msgs,
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=max_tokens,
        )
        return result["choices"][0]["message"]["content"] or ""

    def chat(self, system: str, user: str, **kwargs: Any) -> str:
        """Convenience: single system+user turn → assistant text."""
        return self.complete(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            **kwargs,
        )

    # ── structured JSON output ───────────────────────────────────────-
    def structured(
        self,
        system: str,
        user: str,
        *,
        required_keys: Optional[Sequence[str]] = None,
        retries: int = 2,
        max_tokens: int = 768,
    ) -> Dict[str, Any]:
        """Force the model to return a JSON object and validate it.

        On a parse failure (or a missing required key) we re-prompt with a
        corrective instruction, up to ``retries`` times. This bridges the gap
        between a probabilistic text generator and the deterministic code that
        consumes its verdicts.
        """
        sys_prompt = (
            f"{system}\n\nRespond with ONLY a single valid JSON object and no other text. "
            "Do not wrap it in markdown fences."
        )
        messages: List[Message] = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user},
        ]
        last_error = ""
        for attempt in range(retries + 1):
            raw = self.complete(messages, temperature=0.0, max_tokens=max_tokens)
            try:
                obj = _extract_json(raw)
            except ValueError as exc:
                last_error = str(exc)
            else:
                missing = [k for k in (required_keys or []) if k not in obj]
                if not missing:
                    return obj
                last_error = f"Missing required keys: {missing}"
            # Corrective re-prompt for the next attempt.
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"That was invalid ({last_error}). Return ONLY a valid JSON object"
                + (f" containing keys {list(required_keys)}." if required_keys else "."),
            })
            logger.warning("structured() attempt %d failed: %s", attempt + 1, last_error)
        raise ValueError(f"LLM failed to produce valid JSON after {retries + 1} attempts: {last_error}")


# ── module-level singleton ────────────────────────────────────────────
_singleton: Optional[LocalLLM] = None


def get_llm(backend: Optional[Backend] = None) -> LocalLLM:
    """Return a process-wide LocalLLM. Pass ``backend`` to force a fresh,
    fake-backed instance (used in tests)."""
    global _singleton
    if backend is not None:
        return LocalLLM(backend=backend)
    if _singleton is None:
        _singleton = LocalLLM()
    return _singleton
