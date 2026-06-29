"""massive.com news API client (Polygon-compatible /v2/reference/news schema).

Key responsibility beyond fetching: respect the free tier's **5 requests per
minute** cap. A sliding-window ``RateLimiter`` blocks (sleeps) before a call
would exceed the quota, so the agent can loop over a watchlist without ever
getting a 429.

The client also *normalizes* each article into a compact dict (and surfaces the
API's own ``insights.sentiment``) so the News node feeds the LLM short,
context-window-friendly summaries rather than raw payloads.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.data.rate_limiter import RateLimiter  # re-exported for back-compat

logger = logging.getLogger("news-client")

# Minimal HTML → text extraction (no extra dependency). Drops script/style
# blocks and tags, decodes a few common entities, collapses whitespace.
_SCRIPT_STYLE_RE = re.compile(r"<(script|style)[^>]*>.*?</\1>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"


def html_to_text(html: str) -> str:
    text = _SCRIPT_STYLE_RE.sub(" ", html)
    text = _TAG_RE.sub(" ", text)
    for ent, ch in (("&amp;", "&"), ("&quot;", '"'), ("&#39;", "'"), ("&apos;", "'"),
                    ("&nbsp;", " "), ("&lt;", "<"), ("&gt;", ">")):
        text = text.replace(ent, ch)
    return _WS_RE.sub(" ", text).strip()


class NewsClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        news_path: Optional[str] = None,
        http_client: Optional[httpx.Client] = None,
        rate_limiter: Optional[RateLimiter] = None,
        allowed_publishers: Optional[List[str]] = None,
        disallowed_publishers: Optional[List[str]] = None,
        fetch_content: Optional[bool] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.massive_api_key
        self._base_url = (base_url or settings.massive_api_base_url).rstrip("/")
        self._news_path = news_path or settings.massive_news_path
        self._client = http_client or httpx.Client(timeout=30.0)
        self._limiter = rate_limiter or RateLimiter(
            settings.massive_rate_limit_calls, settings.massive_rate_limit_period_sec
        )
        # Lower-cased publisher allow/block lists; empty allowlist = accept all.
        allow = allowed_publishers if allowed_publishers is not None else settings.news_allowed_publishers
        block = disallowed_publishers if disallowed_publishers is not None else settings.news_disallowed_publishers
        self._allowed_publishers = [p.lower() for p in allow]
        self._disallowed_publishers = [p.lower() for p in block]
        self._fetch_content = (
            fetch_content if fetch_content is not None else settings.news_fetch_full_content)

    def fetch_article_text(self, url: Optional[str]) -> str:
        """Best-effort fetch of an article's body text. Returns "" on any failure
        (paywall, bot-block, timeout, non-HTML) so the caller falls back to the
        headline/description. Truncated to protect the LLM context window."""
        if not url:
            return ""
        try:
            resp = self._client.get(
                url, headers={"User-Agent": _BROWSER_UA}, timeout=10.0, follow_redirects=True)
            if resp.status_code >= 400:
                return ""
            ctype = resp.headers.get("content-type", "")
            if ctype and "html" not in ctype and "text" not in ctype:
                return ""
            return html_to_text(resp.text)[: settings.news_article_max_chars]
        except Exception as exc:  # noqa: BLE001 — never let a bad URL break screening
            logger.debug("Article fetch failed for %s: %s", url, exc)
            return ""

    @property
    def _filtering(self) -> bool:
        return bool(self._allowed_publishers or self._disallowed_publishers)

    def _is_allowed(self, publisher: Optional[str]) -> bool:
        pl = str(publisher).lower() if publisher else ""
        # Blocklist wins: a blocked publisher is always dropped.
        if pl and any(b in pl for b in self._disallowed_publishers):
            return False
        # Empty allowlist => accept anything not blocked.
        if not self._allowed_publishers:
            return True
        if not publisher:
            return False
        return any(a in pl for a in self._allowed_publishers)

    def _headers(self) -> Dict[str, str]:
        # Polygon-style APIs accept the key as a Bearer header; we also pass it
        # as the apiKey query param (below) for maximum compatibility.
        return {"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"}

    def get_news_raw(
        self, ticker: str, limit: int = 10, order: str = "desc", sort: str = "published_utc"
    ) -> Dict[str, Any]:
        """Raw API response for one ticker (rate-limited)."""
        self._limiter.acquire()
        params = {
            "ticker": ticker.upper(),
            "limit": limit,
            "order": order,
            "sort": sort,
            "apiKey": self._api_key,
        }
        url = f"{self._base_url}{self._news_path}"
        resp = self._client.get(url, headers=self._headers(), params=params)
        resp.raise_for_status()
        return resp.json()

    @staticmethod
    def _ticker_sentiment(article: Dict[str, Any], ticker: str) -> Optional[Dict[str, Any]]:
        """Find this ticker's entry in the article's insights[] array, if any."""
        for ins in article.get("insights", []) or []:
            if str(ins.get("ticker", "")).upper() == ticker.upper():
                return {
                    "sentiment": ins.get("sentiment"),
                    "sentiment_reasoning": ins.get("sentiment_reasoning"),
                }
        return None

    def get_headlines(
        self, ticker: str, limit: int = 10, fetch_content: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        """Normalized headline list — what the News node consumes.

        Each item: title, description, url, published_utc, publisher, the API's
        own per-ticker sentiment, and (when enabled) ``content`` = the full
        article body so the LLM can double-check sentiment against the actual
        text rather than the headline alone. Content fetch is best-effort and
        capped (``news_content_max_articles``); failures leave ``content`` empty.
        """
        # When filtering, over-fetch so we still end up with ~limit after filter.
        fetch_limit = limit if not self._filtering else min(max(limit * 5, 30), 100)
        payload = self.get_news_raw(ticker, limit=fetch_limit)
        results = payload.get("results", []) or []

        want_content = self._fetch_content if fetch_content is None else fetch_content
        content_budget = settings.news_content_max_articles

        headlines: List[Dict[str, Any]] = []
        for art in results:
            publisher = art.get("publisher") or {}
            pub_name = publisher.get("name") if isinstance(publisher, dict) else publisher
            if not self._is_allowed(pub_name):
                continue
            url = art.get("article_url") or art.get("amp_url")
            content = ""
            if want_content and content_budget > 0:
                content = self.fetch_article_text(url)
                content_budget -= 1
            headlines.append(
                {
                    "title": art.get("title"),
                    "description": art.get("description"),
                    "content": content,
                    "url": url,
                    "published_utc": art.get("published_utc"),
                    "publisher": pub_name,
                    "tickers": art.get("tickers", []),
                    "api_sentiment": self._ticker_sentiment(art, ticker),
                }
            )
            if len(headlines) >= limit:
                break
        return headlines

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "NewsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
