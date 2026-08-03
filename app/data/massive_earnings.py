"""massive.com earnings client (Benzinga earnings endpoint).

Mirrors ``EarningsClient``'s ``get_next_earnings_date(symbol, from_date,
to_date)`` so it drops straight into the ``CompositeEarningsClient`` chain used
by the entry screener and defense monitor. Backed by
``/benzinga/v1/earnings`` (upcoming rows carry ``date_status == "projected"``).

Auth: ``MASSIVE_API_KEY`` as a Bearer header + ``apiKey`` query param.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import httpx

from app.config import settings
from app.data.rate_limiter import RateLimiter, get_shared_limiter

logger = logging.getLogger("massive-earnings")


class MassiveEarningsClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        path: str = "/benzinga/v1/earnings",
        http_client: Optional[httpx.Client] = None,
        rate_limiter: Optional[RateLimiter] = None,
    ) -> None:
        self._api_key = api_key if api_key is not None else settings.massive_api_key
        self._base_url = (base_url or settings.massive_api_base_url).rstrip("/")
        self._path = path
        self._client = http_client or httpx.Client(timeout=30.0)
        # Shared with every other massive.com client (one account quota).
        self._limiter = rate_limiter or get_shared_limiter(
            "massive", settings.massive_rate_limit_calls, settings.massive_rate_limit_period_sec)

    @property
    def enabled(self) -> bool:
        return bool(self._api_key) and not self._api_key.startswith("your-")

    def get_earnings_calendar(self, symbol: str, from_date: str, to_date: str) -> List[Dict[str, Any]]:
        """Earnings rows for a symbol in [from_date, to_date] (soonest first)."""
        if not self.enabled:
            return []
        self._limiter.acquire()
        params = {
            "ticker": symbol.upper(),
            "date.gte": from_date,
            "date.lte": to_date,
            "sort": "date.asc",
            "limit": 100,
            "apiKey": self._api_key,
        }
        resp = self._client.get(
            f"{self._base_url}{self._path}",
            headers={"Authorization": f"Bearer {self._api_key}", "Accept": "application/json"},
            params=params,
        )
        resp.raise_for_status()
        data = resp.json() or {}
        return data.get("results", []) or []

    def get_next_earnings_date(self, symbol: str, from_date: str, to_date: str) -> Optional[str]:
        """Earliest earnings date (YYYY-MM-DD) in the window, or None if unknown.

        None means no key, no scheduled earnings in range, or an API error — all
        treated by callers as 'earnings unknown'.
        """
        try:
            rows = self.get_earnings_calendar(symbol, from_date, to_date)
        except httpx.HTTPError as exc:
            logger.warning("Massive earnings lookup failed for %s: %s", symbol, exc)
            return None
        dates = sorted(str(r.get("date")) for r in rows if r.get("date"))
        return dates[0] if dates else None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MassiveEarningsClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
