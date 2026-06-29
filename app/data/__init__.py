"""External data clients: Charles Schwab market data + massive.com news.

Both clients are written to be *dependency-injectable*: you can pass in a custom
``httpx`` client (e.g. one backed by ``httpx.MockTransport``) and, for Schwab, a
token provider. That lets the whole data layer be unit-tested offline with
deterministic fixture payloads — no live API keys or network required.
"""

from app.data.schwab_client import SchwabClient
from app.data.news_client import NewsClient
from app.data.earnings_client import EarningsClient
from app.data.earnings_search import EarningsSearchClient, CompositeEarningsClient
from app.data.rate_limiter import RateLimiter

__all__ = [
    "SchwabClient", "NewsClient", "EarningsClient",
    "EarningsSearchClient", "CompositeEarningsClient", "RateLimiter",
]
