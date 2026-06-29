"""Central configuration for the Covered Call Agentic AI system.

Everything that is environment-specific (secrets, file paths) or a tunable
business rule (filter thresholds, the target delta band) lives here so that:

  1. Secrets are read from the gitignored `.env` file, never hardcoded.
  2. Strategy parameters can be tuned in ONE place (the design doc's
     "Human Critique" feedback loop adjusts these numbers over time).

Import the singletons at the bottom (`settings`, `rules`) everywhere else.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from dotenv import load_dotenv

# ──────────────────────────────────────────────────────────────────────
# Paths — resolved relative to the project root so the app runs from
# anywhere (cron job, IDE, shell) without depending on the CWD.
# ──────────────────────────────────────────────────────────────────────
PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

# Load .env from the project root exactly once, at import time.
load_dotenv(PROJECT_ROOT / ".env")


def _get(key: str, default: str | None = None, required: bool = False) -> str:
    """Read an env var, optionally enforcing that it is present."""
    val = os.getenv(key, default)
    if required and (val is None or val == "" or str(val).startswith("your-")):
        raise EnvironmentError(
            f"Required environment variable '{key}' is missing or still a "
            f"placeholder. Set it in {PROJECT_ROOT / '.env'}."
        )
    return val  # type: ignore[return-value]


def _get_bool(key: str, default: bool) -> bool:
    return str(_get(key, "true" if default else "false")).strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class Settings:
    """Secrets and infrastructure settings, sourced from `.env`."""

    # Charles Schwab Developer API
    schwab_app_key: str = field(default_factory=lambda: _get("SCHWAB_APP_KEY", ""))
    schwab_app_secret: str = field(default_factory=lambda: _get("SCHWAB_APP_SECRET", ""))
    schwab_redirect_uri: str = field(
        default_factory=lambda: _get("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
    )

    # massive.com news API (Polygon-compatible /v2/reference/news schema)
    massive_api_key: str = field(default_factory=lambda: _get("MASSIVE_API_KEY", ""))
    massive_api_base_url: str = field(
        default_factory=lambda: _get("MASSIVE_API_BASE_URL", "https://api.massive.com")
    )
    massive_news_path: str = field(
        default_factory=lambda: _get("MASSIVE_NEWS_PATH", "/v2/reference/news")
    )
    # Free tier = 5 requests/minute. The news client enforces this.
    massive_rate_limit_calls: int = field(
        default_factory=lambda: int(_get("MASSIVE_RATE_LIMIT_CALLS", "5"))
    )
    massive_rate_limit_period_sec: float = field(
        default_factory=lambda: float(_get("MASSIVE_RATE_LIMIT_PERIOD_SEC", "60"))
    )
    # Optional publisher allowlist (comma-separated, case-insensitive substring
    # match against each article's publisher name). Empty = accept all sources.
    # e.g. NEWS_ALLOWED_PUBLISHERS="Reuters,Bloomberg,The Wall Street Journal"
    news_allowed_publishers: List[str] = field(
        default_factory=lambda: [
            s.strip() for s in _get("NEWS_ALLOWED_PUBLISHERS", "").split(",") if s.strip()
        ]
    )
    # Optional publisher blocklist (same matching). Takes PRECEDENCE over the
    # allowlist: a blocked publisher is dropped even if it also matches the
    # allowlist. Empty = block nothing.
    news_disallowed_publishers: List[str] = field(
        default_factory=lambda: [
            s.strip() for s in _get("NEWS_DISALLOWED_PUBLISHERS", "").split(",") if s.strip()
        ]
    )
    # Fetch and analyze the FULL article body (not just the headline/description).
    # Best-effort: paywalled/JS-rendered pages may fail and fall back gracefully.
    news_fetch_full_content: bool = field(
        default_factory=lambda: _get_bool("NEWS_FETCH_FULL_CONTENT", True)
    )
    # Per-article body is truncated to protect the LLM context window.
    news_article_max_chars: int = field(
        default_factory=lambda: int(_get("NEWS_ARTICLE_MAX_CHARS", "2000"))
    )
    # Bound how many articles we fetch full content for (each is a separate HTTP
    # GET to the publisher, so this caps latency).
    news_content_max_articles: int = field(
        default_factory=lambda: int(_get("NEWS_CONTENT_MAX_ARTICLES", "5"))
    )

    # Finnhub earnings calendar (free tier ~60 req/min). Used by the News node's
    # earnings guardrail. If the key is absent, earnings is treated as UNKNOWN
    # (flagged but allowed) rather than crashing.
    finnhub_api_key: str = field(default_factory=lambda: _get("FINNHUB_API_KEY", ""))
    earnings_api_base_url: str = field(
        default_factory=lambda: _get("EARNINGS_API_BASE_URL", "https://finnhub.io/api/v1")
    )
    earnings_path: str = field(
        default_factory=lambda: _get("EARNINGS_PATH", "/calendar/earnings")
    )
    earnings_rate_limit_calls: int = field(
        default_factory=lambda: int(_get("EARNINGS_RATE_LIMIT_CALLS", "30"))
    )
    earnings_rate_limit_period_sec: float = field(
        default_factory=lambda: float(_get("EARNINGS_RATE_LIMIT_PERIOD_SEC", "60"))
    )
    # Google-search earnings fallback (used when Finnhub returns no date). Scrapes
    # the search result page for an earnings date; best-effort + may be blocked.
    earnings_search_enabled: bool = field(
        default_factory=lambda: _get_bool("EARNINGS_SEARCH_ENABLED", False)
    )
    earnings_search_url: str = field(
        default_factory=lambda: _get("EARNINGS_SEARCH_URL", "https://www.google.com/search")
    )
    earnings_search_query: str = field(
        default_factory=lambda: _get("EARNINGS_SEARCH_QUERY", "{ticker} stock next earnings date")
    )
    earnings_search_rate_limit_calls: int = field(
        default_factory=lambda: int(_get("EARNINGS_SEARCH_RATE_LIMIT_CALLS", "10"))
    )
    earnings_search_rate_limit_period_sec: float = field(
        default_factory=lambda: float(_get("EARNINGS_SEARCH_RATE_LIMIT_PERIOD_SEC", "60"))
    )
    # Use the LLM to disambiguate WHICH extracted date is the earnings date
    # (hybrid: LLM picks only from regex-grounded candidates → no hallucinated
    # dates; deterministic code still does any quarterly inference).
    earnings_search_use_llm: bool = field(
        default_factory=lambda: _get_bool("EARNINGS_SEARCH_USE_LLM", False)
    )

    # Schwab market-data base URL
    schwab_base_url: str = field(
        default_factory=lambda: _get("SCHWAB_BASE_URL", "https://api.schwabapi.com/marketdata/v1")
    )
    # Schwab documents ~120 req/min; default conservatively to avoid throttling
    # a 200-symbol run. Applies to every Schwab HTTP call.
    schwab_rate_limit_calls: int = field(
        default_factory=lambda: int(_get("SCHWAB_RATE_LIMIT_CALLS", "100"))
    )
    schwab_rate_limit_period_sec: float = field(
        default_factory=lambda: float(_get("SCHWAB_RATE_LIMIT_PERIOD_SEC", "60"))
    )
    # How many symbols to request per batched /quotes call.
    schwab_quote_batch_size: int = field(
        default_factory=lambda: int(_get("SCHWAB_QUOTE_BATCH_SIZE", "25"))
    )

    # Discord HITL webhook
    discord_webhook_url: str = field(
        default_factory=lambda: _get("DISCORD_WEBHOOK_URL", "")
    )

    # ChromaDB semantic memory
    chroma_collection: str = field(
        default_factory=lambda: _get("CHROMA_COLLECTION", "trade_lessons")
    )
    embedding_model: str = field(
        default_factory=lambda: _get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    # Embedding backend: "auto" prefers sentence-transformers if installed, else
    # falls back to ChromaDB's built-in ONNX MiniLM (no torch needed). Force with
    # "sentence_transformers" or "default".
    embedding_backend: str = field(
        default_factory=lambda: _get("EMBEDDING_BACKEND", "auto")
    )

    # Local LLM (llama-cpp-python)
    llm_model_path: str = field(
        default_factory=lambda: str(
            PROJECT_ROOT
            / _get("LLM_MODEL_PATH", "model/qwen2.5-coder-14b-instruct-q4_k_m.gguf")
        )
    )
    llm_context_size: int = field(default_factory=lambda: int(_get("LLM_CONTEXT_SIZE", "50000")))
    llm_gpu_layers: int = field(default_factory=lambda: int(_get("LLM_GPU_LAYERS", "-1")))
    llm_temperature: float = field(default_factory=lambda: float(_get("LLM_TEMPERATURE", "0.2")))

    # Key file locations
    watchlist_path: Path = PROJECT_ROOT / "app" / "watchlist.json"
    # All persistence lives under app/memory/ (SQL + vector store side by side).
    sql_db_path: Path = PROJECT_ROOT / "app" / "memory" / "sql" / "trading_agent.db"
    sql_schema_path: Path = PROJECT_ROOT / "app" / "memory" / "sql" / "schema.sql"
    chroma_dir: Path = PROJECT_ROOT / "app" / "memory" / "chroma"
    schwab_token_path: Path = PROJECT_ROOT / "charles_schwab_mcp" / "schwab_tokens.json"
    # Per-run artifacts: full recommendation summary (.md) + proposals (.json,
    # status PENDING_APPROVAL) that the human-feedback CLI later reads.
    runs_dir: Path = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class StrategyRules:
    """Tunable business rules. The design doc's feedback loop edits THESE.

    Grouped to match the agent nodes that enforce them.
    """

    # ── Scout: universe / liquidity filters ──────────────────────────
    # The watchlist is already curated (volume/dividend/optionable applied when
    # the symbols were chosen). When True, the Scout skips those redundant
    # filters and only does a liveness/price check. Flip to False to screen a
    # raw watchlist with the full filter battery.
    watchlist_is_prefiltered: bool = True
    min_avg_daily_volume: int = 1_000_000      # "highly liquid (1M+ trades/day)"
    min_dividend_yield_pct: float = 2.0        # ">2% annual dividend payout"
    require_optionable: bool = True            # must have listed options

    # ── Quant: price history ──────────────────────────────────────────
    # Months of DAILY candles to fetch. The indicators need 50 (SMA-50) +
    # trend_lookback_days trading days; 3 months (~63 trading days) is NOT
    # enough. 6 months (~126 trading days) clears the 70-day minimum with margin
    # and stays under 200, so the long-term SMA-200 is skipped (as intended).
    price_history_months: int = 6
    trend_lookback_days: int = 20
    # Cap how many Scout survivors the Quant node fully analyzes per run. Each
    # candidate costs ~2 rate-limited Schwab calls (history + chain), so a 200+
    # watchlist takes several minutes. 0 = no cap (analyze all). Deferred names
    # are recorded in the audit trail (no silent truncation).
    max_quant_candidates: int = 0

    # ── Quant: contract selection ────────────────────────────────────
    target_delta: float = 0.40                 # midpoint of the 0.30–0.40 band
    delta_band: tuple[float, float] = (0.30, 0.50)
    min_days_to_expiration: int = 20           # theta-decay sweet spot (design §2)
    max_days_to_expiration: int = 60
    min_iv_rank: float = 50.0                  # used if a true 52-wk IV history is available
    # IV-Rank proxy: Schwab gives current IV but not a 52-week IV history, so we
    # compare current option IV to the stock's realized (historical) volatility.
    # IV >= this multiple of realized vol ⇒ premiums are "rich" (seller's edge).
    iv_richness_min_ratio: float = 1.1
    require_rich_iv: bool = False               # reject candidates that fail the IV check
    # Reject non-uptrend names? Covered calls prefer sideways/up. Downtrends risk
    # the underlying tanking. When True, "Downward (Bearish)" trend is rejected.
    reject_downtrend: bool = True

    # ── Position sizing ───────────────────────────────────────────────
    max_allocation_per_trade_pct: float = 100.0  # cap of account cash per single trade

    # ── Liquidity guard (options) ────────────────────────────────────
    max_bid_ask_spread_pct: float = 15.0       # disqualify wide/illiquid chains
    min_option_open_interest: int = 100
    min_option_volume: int = 10

    # ── Risk Manager: target alignment ───────────────────────────────
    min_annualized_yield_pct: float = 10.0     # the headline ">10%" mission target
    # Which annualized figure the >10% gate uses: "flat" (premium income if the
    # stock is unchanged) or "assigned" (total return if called away).
    yield_target_metric: str = "flat"
    top_n_candidates: int = 5                  # how many to surface to the human

    # Composite score weights (the §5 feedback loop tunes THESE). Must be keyed
    # yield/iv/sentiment/buffer/prob; they are normalized to sum to 1.0.
    score_weights: Dict[str, float] = field(
        default_factory=lambda: {
            "yield": 0.35,
            "iv": 0.20,
            "sentiment": 0.20,
            "buffer": 0.15,
            "prob": 0.10,
        }
    )
    # Score (0–100) → letter grade cutoffs.
    grade_thresholds: Dict[str, float] = field(
        default_factory=lambda: {"A": 75.0, "B": 60.0, "C": 45.0}
    )

    # ── Defense monitor: downside trigger ────────────────────────────
    downside_breach_pct: float = -5.0          # % drop from entry that triggers ToT

    # ── News node ─────────────────────────────────────────────────────
    headlines_per_symbol: int = 30              # how many headlines to feed the LLM
    # Cap how many Quant survivors get news-screened per run (protects the
    # massive.com 5/min free tier + LLM time). Top-N by annualized yield are
    # screened; the rest are recorded (no silent truncation).
    news_max_candidates: int = 100
    # Deterministic catastrophic-risk scan (case-insensitive, WORD-BOUNDARY match
    # over title + description + full article body). By default these are ADVISORY
    # — surfaced for the human, but the LLM's content-based read is what decides
    # catastrophic risk (a stray keyword in an unrelated paragraph shouldn't veto
    # a benign stock). Set catastrophic_keyword_veto=True to make any match a hard
    # auto-reject regardless of the LLM.
    catastrophic_keyword_veto: bool = False
    catastrophic_keywords: List[str] = field(
        default_factory=lambda: [
            # Insolvency / financial distress
            "bankruptcy", "bankrupt", "chapter 11", "chapter 7", "insolvency", "insolvent",
            "going concern", "debt default", "defaults on", "missed payment", "liquidation",
            "restructuring", "debt restructuring", "covenant breach", "creditor protection",
            "cash crunch", "liquidity crisis", "write-down", "writedown", "impairment charge",
            # Fraud / accounting / governance
            "fraud", "accounting scandal", "accounting irregularities", "restatement",
            "restate earnings", "misstated", "ponzi", "embezzlement", "money laundering",
            "whistleblower", "auditor resigns", "material weakness", "books cooked",
            "cfo resigns", "ceo resigns", "ceo steps down", "executive departure",
            # Legal / regulatory / criminal
            "sec investigation", "sec probe", "sec charges", "doj investigation",
            "ftc investigation", "antitrust", "subpoena", "criminal charges", "indictment",
            "indicted", "guilty plea", "settlement", "class action", "lawsuit", "sued",
            "litigation", "injunction", "consent decree", "fined", "penalty", "sanctions",
            "regulatory action", "license revoked", "delisting", "delisted", "halted",
            # Operational disasters
            "product recall", "recall", "data breach", "data leak", "ransomware",
            "cyberattack", "hacked", "security breach", "explosion", "fire at", "spill",
            "contamination", "safety violation", "plant shutdown", "production halt",
            "factory fire", "outage", "fda rejection", "fda warning", "clinical trial failure",
            "trial failed", "phase 3 failure", "patent invalidated", "drug recall",
            # Severe business deterioration
            "profit warning", "guidance cut", "slashes guidance", "cuts guidance",
            "withdraws guidance", "earnings miss", "massive miss", "dividend cut",
            "suspends dividend", "layoffs", "mass layoffs", "plunges", "craters",
            "collapses", "short seller", "short report", "downgraded to sell", "fraud allegations",
            "merger collapses", "deal terminated", "acquisition falls through", "credit downgrade",
        ]
    )

    # ── Sentiment scoring scale (News node) ──────────────────────────
    # LLM maps headlines to one of these; Risk Manager filters on the floor.
    sentiment_scale: List[str] = field(
        default_factory=lambda: ["VERY_NEGATIVE", "NEGATIVE", "NEUTRAL", "POSITIVE", "VERY_POSITIVE"]
    )
    min_acceptable_sentiment: str = "NEUTRAL"  # reject anything below this


# Module-level singletons — import these everywhere.
settings = Settings()
rules = StrategyRules()
