"""Google-search earnings-date engine + a composite provider.

Free structured earnings calendars (e.g. Finnhub) frequently return no date for
a given symbol. This engine is a fallback: it searches the web for the stock's
earnings date and parses the result. It can either:

  * use the NEXT earnings date when the search surfaces one, or
  * find the most recent PAST earnings date and infer the next one assuming a
    quarterly cadence (+3 months until the date is in the future).

Reality check: scraping a search engine is best-effort — consent/captcha pages,
bot-blocking, and changing HTML mean it can return nothing. Every failure path
degrades to ``None`` so the caller treats earnings as "unknown" (flag-but-allow),
never crashing. Use it behind ``CompositeEarningsClient`` so the structured API
is tried first.
"""

from __future__ import annotations

import logging
import re
from datetime import date
from typing import Any, Callable, List, Optional

from app.config import settings
from app.data.news_client import html_to_text
from app.data.rate_limiter import RateLimiter

logger = logging.getLogger("earnings-search")

_BROWSER_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

# Date formats commonly seen in search snippets.
_RE_MONTHNAME = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", re.I)
_RE_ISO = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")
_RE_MDY = re.compile(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b")


def _safe_date(y: int, m: int, d: int) -> Optional[date]:
    try:
        return date(y, m, d)
    except ValueError:
        return None


def _parse_iso(text: Optional[str]) -> Optional[date]:
    if not text:
        return None
    m = _RE_ISO.search(text)
    return _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None


def add_months(d: date, n: int) -> date:
    """Add ``n`` months, clamping the day to the target month's length."""
    month_index = d.month - 1 + n
    year = d.year + month_index // 12
    month = month_index % 12 + 1
    # Clamp day (e.g. Jan 31 + 1mo → Feb 28/29).
    for day in range(d.day, 27, -1):
        safe = _safe_date(year, month, day)
        if safe:
            return safe
    return date(year, month, min(d.day, 28))


def find_earnings_dates(text: str) -> List[date]:
    """Extract candidate dates, preferring ones near 'earnings'/'report'."""
    low = text.lower()
    found: List[tuple] = []  # (position, date)

    for m in _RE_MONTHNAME.finditer(text):
        mon = _MONTHS.get(m.group(1)[:3].lower())
        if mon:
            d = _safe_date(int(m.group(3)), mon, int(m.group(2)))
            if d:
                found.append((m.start(), d))
    for m in _RE_ISO.finditer(text):
        d = _safe_date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        if d:
            found.append((m.start(), d))
    for m in _RE_MDY.finditer(text):
        d = _safe_date(int(m.group(3)), int(m.group(1)), int(m.group(2)))
        if d:
            found.append((m.start(), d))

    # Prefer dates within a small window after the word earnings/report.
    strong = [d for pos, d in found if "earnings" in low[max(0, pos - 60):pos + 12]
              or "report" in low[max(0, pos - 60):pos + 12]]
    chosen = strong or [d for _, d in found]
    # Dedup, preserve order.
    seen, out = set(), []
    for d in chosen:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


_LLM_SYSTEM = (
    "You identify a company's earnings report date from messy web-search text. "
    "You will be given the text plus a list of candidate dates already extracted "
    "from it. Choose the ONE date that is the company's earnings report date — "
    "prefer the upcoming/next one; otherwise the most recent past one. Ignore "
    "ex-dividend dates, analyst price-target dates, and unrelated dates. You MUST "
    "pick a date from the provided candidate list, or null if none is an earnings date."
)


class EarningsSearchClient:
    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        query: Optional[str] = None,
        http_client: Any = None,
        rate_limiter: Optional[RateLimiter] = None,
        llm: Any = None,
        use_llm: Optional[bool] = None,
        quarter_months: int = 3,
    ) -> None:
        import httpx  # local import keeps module light if unused
        self._base_url = base_url or settings.earnings_search_url
        self._query = query or settings.earnings_search_query
        self._client = http_client or httpx.Client(timeout=10.0)
        self._limiter = rate_limiter or RateLimiter(
            settings.earnings_search_rate_limit_calls,
            settings.earnings_search_rate_limit_period_sec, name="earnings-search")
        self._llm = llm
        self._use_llm = settings.earnings_search_use_llm if use_llm is None else use_llm
        self._quarter_months = quarter_months

    def search_raw(self, ticker: str) -> str:
        self._limiter.acquire()
        resp = self._client.get(
            self._base_url, params={"q": self._query.format(ticker=ticker.upper())},
            headers={"User-Agent": _BROWSER_UA}, follow_redirects=True)
        resp.raise_for_status()
        return resp.text

    def _llm_pick(self, text: str, symbol: str, candidates: List[date]) -> Optional[date]:
        """Let the LLM choose which candidate date is the earnings date.

        Hybrid guardrail: the LLM may ONLY return a date that is already in the
        regex-extracted candidate set — any other answer (a hallucinated date) is
        rejected and we fall back to the heuristic. The LLM does no date math.
        """
        cand_iso = [d.isoformat() for d in candidates]
        user = (
            f"Company: {symbol}\nSearch text:\n{text[:2000]}\n\n"
            f"Candidate dates extracted from the text: {', '.join(cand_iso)}\n\n"
            'Return JSON: {"earnings_date": "YYYY-MM-DD" chosen from the candidates, or null}.'
        )
        try:
            obj = self._llm.structured(_LLM_SYSTEM, user, required_keys=["earnings_date"])
        except Exception as exc:  # noqa: BLE001
            logger.debug("Earnings LLM pick failed for %s: %s", symbol, exc)
            return None
        picked = _parse_iso(obj.get("earnings_date"))
        if picked and picked in set(candidates):  # grounding check: must be real
            return picked
        if obj.get("earnings_date"):
            logger.debug("Earnings LLM returned ungrounded date %s for %s; ignoring.",
                         obj.get("earnings_date"), symbol)
        return None

    def _choose_date(self, text: str, symbol: str, candidates: List[date], today: date) -> date:
        """Pick the single best earnings date: LLM choice if grounded, else the
        heuristic (earliest future, otherwise latest past)."""
        if self._llm is not None and self._use_llm:
            picked = self._llm_pick(text, symbol, candidates)
            if picked is not None:
                return picked
        future = sorted(d for d in candidates if d > today)
        return future[0] if future else max(candidates)

    def _project_to_next(self, chosen: date, today: date) -> date:
        """Roll a chosen date forward to the next future occurrence, quarterly.
        A future date is returned unchanged; a past date is inferred forward."""
        nxt = chosen
        while nxt <= today:
            nxt = add_months(nxt, self._quarter_months)
        return nxt

    def get_next_earnings_date(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> Optional[str]:
        """Best-effort next earnings date (YYYY-MM-DD), or None.

        ``from_date`` is treated as 'today' (deterministic for tests). The LLM (if
        configured) disambiguates which extracted date is the earnings date;
        deterministic code projects a past date forward quarterly.
        """
        today = _parse_iso(from_date) or date.today()
        try:
            text = html_to_text(self.search_raw(symbol))
        except Exception as exc:  # noqa: BLE001 — never crash on a blocked/failed search
            logger.debug("Earnings search failed for %s: %s", symbol, exc)
            return None

        candidates = find_earnings_dates(text)
        if not candidates:
            return None

        chosen = self._choose_date(text, symbol, candidates, today)
        nxt = self._project_to_next(chosen, today)
        how = "found" if chosen > today else f"inferred quarterly from {chosen}"
        logger.info("Earnings search: %s next earnings %s (%s).", symbol, nxt, how)
        return nxt.isoformat()

    def close(self) -> None:
        self._client.close()


class CompositeEarningsClient:
    """Try multiple earnings providers in order; return the first date found.

    Default order: structured API (Finnhub) first, then the Google-search engine.
    Exposes the same ``get_next_earnings_date`` interface the News node expects.
    """

    def __init__(self, providers: List[Any]) -> None:
        self._providers = [p for p in providers if p is not None]

    def get_next_earnings_date(
        self, symbol: str, from_date: Optional[str] = None, to_date: Optional[str] = None
    ) -> Optional[str]:
        for provider in self._providers:
            try:
                d = provider.get_next_earnings_date(symbol, from_date, to_date)
            except Exception as exc:  # noqa: BLE001
                logger.debug("Earnings provider %s failed for %s: %s",
                             type(provider).__name__, symbol, exc)
                d = None
            if d:
                return d
        return None
