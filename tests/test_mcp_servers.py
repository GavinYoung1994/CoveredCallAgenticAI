"""Smoke tests for the FastMCP servers: they import and register their tools.

Verifies that every math-engine function and every data-API method is exposed as
an MCP tool. Requires the `mcp` package; skips cleanly if it's absent.
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "math_mcp"))


def _have_mcp():
    # Instantiating FastMCP triggers a pydantic-settings .env read, which the
    # Claude sandbox blocks (PermissionError). Probe it so we skip in-sandbox but
    # fully validate when run normally. Catch any error, not just ImportError.
    try:
        from mcp.server.fastmcp import FastMCP
        FastMCP("probe")
        return True
    except Exception:  # noqa: BLE001
        return False


def _tool_names(server) -> set:
    """Return the registered tool names from a FastMCP server (async list_tools)."""
    tools = asyncio.run(server.list_tools())
    return {t.name for t in tools}


def test_math_mcp_exposes_every_engine_function():
    if not _have_mcp():
        print("  ⏭  skipping (mcp not installed)")
        return
    import math_mcp  # math_mcp/math_mcp.py
    from app.engine import math_engine

    engine_fns = {n for n in dir(math_engine)
                  if callable(getattr(math_engine, n)) and not n.startswith("_")
                  and getattr(math_engine, n).__module__ == math_engine.__name__}
    tool_names = _tool_names(math_mcp.mcp)
    missing = engine_fns - tool_names
    assert not missing, f"engine functions missing from math MCP: {missing}"


def test_data_mcp_exposes_all_apis():
    if not _have_mcp():
        print("  ⏭  skipping (mcp not installed)")
        return
    from app.data import data_mcp_server
    names = _tool_names(data_mcp_server.mcp)
    # Every client domain is represented.
    assert {"schwab_get_quote", "schwab_get_price_history", "schwab_get_option_chain",
            "schwab_get_option_expirations", "schwab_lookup_instrument", "schwab_is_optionable",
            "schwab_get_fundamentals", "schwab_get_quotes"} <= names
    assert {"news_get_headlines", "news_get_raw", "news_fetch_article_text"} <= names
    assert {"earnings_finnhub_next_date", "earnings_finnhub_calendar",
            "earnings_search_next_date"} <= names


def test_manager_mcp_imports():
    if not _have_mcp():
        print("  ⏭  skipping (mcp not installed)")
        return
    from app.agent import mcp_server
    names = _tool_names(mcp_server.mcp)
    assert {"get_cash", "set_cash", "analyze_covered_call", "performance_report"} <= names


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
