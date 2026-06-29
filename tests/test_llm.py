"""Tests for the local LLM layer using an injected fake backend.

No model weights and no llama-cpp-python are required: we inject a backend
callable that returns canned responses, so we can exercise completion,
JSON extraction, validation, and the corrective-retry loop deterministically.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LocalLLM, _extract_json, get_llm, ModelNotAvailableError


def test_extract_json_plain():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_fenced():
    text = "Sure!\n```json\n{\"sentiment\": \"POSITIVE\", \"score\": 4}\n```\nDone."
    assert _extract_json(text) == {"sentiment": "POSITIVE", "score": 4}


def test_extract_json_with_surrounding_prose():
    text = 'Here is my answer: {"verdict": "approve"} — hope that helps.'
    assert _extract_json(text) == {"verdict": "approve"}


def test_extract_json_failure():
    try:
        _extract_json("no json here")
        assert False, "should have raised"
    except ValueError:
        pass


def test_complete_uses_backend():
    llm = LocalLLM(backend=lambda msgs: f"echo:{msgs[-1]['content']}")
    out = llm.chat("you are a bot", "hello")
    assert out == "echo:hello"
    assert llm.is_loaded is False  # never loaded a real model


def test_structured_happy_path():
    backend = lambda msgs: '{"sentiment": "NEGATIVE", "score": 2}'
    llm = LocalLLM(backend=backend)
    obj = llm.structured("Score sentiment.", "Bad news.", required_keys=["sentiment", "score"])
    assert obj["sentiment"] == "NEGATIVE" and obj["score"] == 2


def test_structured_retries_then_succeeds():
    # First call returns junk, second returns valid JSON → should recover.
    calls = {"n": 0}

    def flaky_backend(msgs):
        calls["n"] += 1
        return "I cannot comply" if calls["n"] == 1 else '{"ok": true}'

    llm = LocalLLM(backend=flaky_backend)
    obj = llm.structured("x", "y", required_keys=["ok"], retries=2)
    assert obj["ok"] is True and calls["n"] == 2


def test_structured_missing_key_retries():
    # Returns valid JSON but missing the required key on attempt 1.
    calls = {"n": 0}

    def backend(msgs):
        calls["n"] += 1
        return '{"foo": 1}' if calls["n"] == 1 else '{"score": 5}'

    llm = LocalLLM(backend=backend)
    obj = llm.structured("x", "y", required_keys=["score"], retries=2)
    assert obj["score"] == 5 and calls["n"] == 2


def test_structured_gives_up():
    llm = LocalLLM(backend=lambda msgs: "still not json")
    try:
        llm.structured("x", "y", retries=1)
        assert False, "should have raised"
    except ValueError as exc:
        assert "valid JSON" in str(exc)


def test_missing_model_raises_cleanly():
    # No backend + nonexistent path → clear, catchable error (not a crash).
    llm = LocalLLM(model_path="/nonexistent/model.gguf")
    try:
        llm.chat("a", "b")
        assert False, "should have raised"
    except ModelNotAvailableError:
        pass


def test_get_llm_backend_returns_fresh():
    llm = get_llm(backend=lambda msgs: "x")
    assert llm.complete([{"role": "user", "content": "hi"}]) == "x"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
            passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
