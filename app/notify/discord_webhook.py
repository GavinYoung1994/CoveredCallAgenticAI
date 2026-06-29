"""Discord webhook notifier for the HITL checkpoint.

``format_recommendations`` turns the Risk Manager's graded picks into a readable
Markdown summary (premium, annualized yield, IV richness, sentiment, the news
sources checked, and the reasoning). ``DiscordNotifier.send`` POSTs it to the
configured webhook.

Testability: the HTTP poster is injectable, so tests capture the exact payload
without any network call. The ``requests`` import is lazy so the module loads
even where requests isn't installed.
"""

from __future__ import annotations

import logging
from urllib.parse import quote_plus
from typing import Any, Callable, Dict, List, Optional

from app.config import settings

logger = logging.getLogger("discord-notifier")

# A poster takes (url, json_payload) and returns an HTTP status code.
Poster = Callable[[str, Dict[str, Any]], int]

_DISCORD_LIMIT = 2000  # Discord message content hard limit


def _earnings_search_url(symbol: str) -> str:
    """A one-click Google search for the ticker's next earnings date."""
    query = settings.earnings_search_query.format(ticker=symbol.upper())
    return f"https://www.google.com/search?q={quote_plus(query)}"


def _default_poster(url: str, payload: Dict[str, Any]) -> int:
    import requests  # lazy import
    resp = requests.post(url, json=payload, timeout=15)
    return resp.status_code


def format_recommendations(
    recommendations: List[Dict[str, Any]],
    *,
    run_id: str = "",
    account_cash: Optional[float] = None,
) -> str:
    """Build the Markdown HITL summary the human reviews before executing."""
    if not recommendations:
        return ("📭 **Covered-Call Screener** — no candidates passed all guardrails "
                f"this run{f' ({run_id})' if run_id else ''}.")

    lines: List[str] = []
    lines.append("📈 **Covered-Call Candidates — HUMAN APPROVAL REQUIRED**")
    if run_id:
        lines.append(f"_Run: {run_id}_")
    if account_cash is not None:
        lines.append(f"_Account cash: ${account_cash:,.0f}_")
    lines.append("⚠️ Autonomous trading is disabled. Review and execute manually.\n")

    for i, rec in enumerate(recommendations, 1):
        c = rec.get("contract", {})
        ym = rec.get("yield_metrics", {})
        lines.append(
            f"**{i}. {rec['symbol']}  —  Grade {rec.get('grade', '?')} "
            f"(score {rec.get('score', 0):.0f})**"
        )
        strike = c.get("strike")
        exp = str(c.get("expiration_key", "")).split(":")[0]
        dte = c.get("days_to_expiration")
        lines.append(
            f"• Sell {strike} call exp {exp} ({dte}d), Δ{c.get('delta', 0):.2f}, "
            f"mark ${c.get('mark', 0):.2f}"
        )
        lines.append(
            f"• Annualized: {rec.get('annualized_yield_percent', 0):.1f}% "
            f"| downside buffer {ym.get('downside_buffer_percent', 0):.1f}%"
        )
        if rec.get("iv_to_hv_ratio") is not None:
            lines.append(f"• IV/HV richness: {rec['iv_to_hv_ratio']:.2f}x")
        lines.append(f"• Sentiment: {rec.get('sentiment', 'N/A')}")
        if not rec.get("earnings_known", True):
            url = _earnings_search_url(rec.get("symbol", ""))
            lines.append(f"• ⚠️ Earnings date UNKNOWN — verify: {url}")
        elif rec.get("earnings_date"):
            lines.append(f"• Earnings: {rec['earnings_date']} (clear of expiration)")
        if rec.get("rationale"):
            lines.append(f"• Reasoning: {rec['rationale']}")
        srcs = rec.get("sources") or []
        if srcs:
            lines.append("• News checked:")
            for s in srcs[:4]:
                pub = s.get("publisher") or "source"
                url = s.get("url") or ""
                title = s.get("title") or ""
                lines.append(f"    - [{pub}] {title} {url}".rstrip())
        lines.append("")

    # Full message — no truncation. The notifier splits it across multiple
    # webhook calls to respect Discord's 2000-char-per-message hard limit.
    return "\n".join(lines)


class DiscordNotifier:
    def __init__(
        self,
        *,
        webhook_url: Optional[str] = None,
        poster: Optional[Poster] = None,
    ) -> None:
        self._webhook_url = webhook_url if webhook_url is not None else settings.discord_webhook_url
        self._poster = poster or _default_poster

    @property
    def enabled(self) -> bool:
        url = self._webhook_url or ""
        return url.startswith("https://") and "xxxx" not in url

    @staticmethod
    def _chunk(content: str, limit: int = _DISCORD_LIMIT) -> List[str]:
        """Split content into <=limit pieces at line boundaries (Discord's hard
        2000-char cap). A single over-long line is hard-split as a last resort."""
        chunks: List[str] = []
        current = ""
        for line in content.split("\n"):
            while len(line) > limit:  # pathological long line
                if current:
                    chunks.append(current)
                    current = ""
                chunks.append(line[:limit])
                line = line[limit:]
            candidate = f"{current}\n{line}" if current else line
            if len(candidate) > limit:
                if current:
                    chunks.append(current)
                current = line
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks or [""]

    def send(self, content: str) -> bool:
        """POST a message to Discord, split across as many calls as needed to
        deliver the FULL content. Returns True only if every chunk succeeded."""
        if not self.enabled:
            logger.warning("Discord webhook not configured; skipping notification.")
            return False
        chunks = self._chunk(content)
        for idx, chunk in enumerate(chunks, 1):
            try:
                status = self._poster(self._webhook_url, {"content": chunk})
            except Exception as exc:  # noqa: BLE001
                logger.error("Discord send failed on chunk %d/%d: %s", idx, len(chunks), exc)
                return False
            if not (200 <= status < 300):
                logger.error("Discord webhook returned HTTP %s on chunk %d/%d", status, idx, len(chunks))
                return False
        logger.info("Discord notification sent in %d message(s).", len(chunks))
        return True
