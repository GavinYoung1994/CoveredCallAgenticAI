"""Optional REAL-model smoke test for the local Qwen2.5 GGUF.

Unlike tests/test_llm.py (which uses a fake backend), this actually loads the
9 GB model via llama-cpp-python and asks it for a tiny JSON sentiment verdict.
It is intentionally NOT part of the fast unit-test suite.

Run:  ./venv/bin/python scripts/smoke_test_llm.py
Requires: `pip install llama-cpp-python` and the .gguf present at LLM_MODEL_PATH.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.llm import LocalLLM, ModelNotAvailableError


def main() -> int:
    llm = LocalLLM()
    print(f"Model path : {llm.model_path}")
    print(f"n_ctx={llm.n_ctx}  n_gpu_layers={llm.n_gpu_layers}  temp={llm.temperature}")
    print("Loading model (first call may take a while)...")
    t0 = time.time()
    try:
        verdict = llm.structured(
            system="You are a financial news sentiment classifier.",
            user=(
                "Headline: 'Acme Corp crushes earnings, raises full-year guidance.'\n"
                "Classify sentiment as one of VERY_NEGATIVE, NEGATIVE, NEUTRAL, "
                "POSITIVE, VERY_POSITIVE and give a 1-2 sentence reason."
            ),
            required_keys=["sentiment", "reason"],
        )
    except ModelNotAvailableError as exc:
        print(f"\n⏭  SKIPPED: {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n💥 Model error: {type(exc).__name__}: {exc}")
        return 1
    print(f"\n✅ Loaded + responded in {time.time() - t0:.1f}s")
    print(f"   sentiment = {verdict.get('sentiment')}")
    print(f"   reason    = {verdict.get('reason')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
