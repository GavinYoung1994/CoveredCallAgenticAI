"""Tests for per-run artifact logging (temp dir, no network)."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.runlog import save_run, load_run, format_rejections


def test_format_rejections_groups_and_tallies():
    rejected = [
        {"symbol": "A", "stage": "QUANT", "reason": "Unaffordable: 100 shares cost $200,000, cash $0."},
        {"symbol": "B", "stage": "QUANT", "reason": "Unaffordable: 100 shares cost $150,000, cash $0."},
        {"symbol": "C", "stage": "QUANT", "reason": "Downtrend (Downward (Bearish)); prefer flat/up."},
        {"symbol": "D", "stage": "NEWS", "reason": "Sentiment NEGATIVE below floor NEUTRAL.",
         "sources": [{"publisher": "Wire", "title": "D tanks", "url": "http://x"}]},
    ]
    md = format_rejections(rejected)
    assert "## Rejections (4)" in md
    assert "### QUANT — 3" in md and "### NEWS — 1" in md
    assert "- Unaffordable: 2" in md          # tally collapses the 2 unaffordable reasons
    assert "- Downtrend: 1" in md
    assert "[Wire] D tanks http://x" in md     # news source shown for the human to review


def test_format_rejections_empty():
    assert "None" in format_rejections([])


def test_save_and_load_run():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        recs = [{"symbol": "AAPL", "grade": "A", "score": 80.0}]
        paths = save_run("run-123", "FULL SUMMARY TEXT", recs, run_timestamp="2026-06-23T10:00:00Z", runs_dir=d)
        assert Path(paths["markdown"]).exists()
        assert Path(paths["json"]).exists()
        # Markdown holds the full summary (no truncation) + a rejection section.
        md = Path(paths["markdown"]).read_text()
        assert md.startswith("FULL SUMMARY TEXT")
        assert "## Rejections" in md
        # JSON is the proposals record the feedback CLI reads.
        loaded = load_run("run-123", runs_dir=d)
        assert loaded["status"] == "PENDING_APPROVAL"
        assert loaded["recommendations"][0]["symbol"] == "AAPL"
        assert loaded["run_id"] == "run-123"


def test_load_missing_run_returns_none():
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as d:
        assert load_run("nope", runs_dir=d) is None


def test_save_run_survives_bad_dir():
    # A bad path must not raise — logging failures can't kill a screener run.
    paths = save_run("r", "summary", [], runs_dir="/proc/should-not-be-writable/x")
    assert "markdown" not in paths  # write failed, but no exception


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  ✅ {t.__name__}"); passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
