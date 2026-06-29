"""Shared sliding-window rate limiter.

Used by BOTH the massive.com news client (free tier: 5 req/min) and the Schwab
market-data client (so a 200-symbol run never hammers the API past its cap).

``time_func``/``sleep_func`` are injectable so tests can drive a fake clock and
assert throttling behaviour without real delays.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Callable, Deque

logger = logging.getLogger("rate-limiter")


class RateLimiter:
    def __init__(
        self,
        max_calls: int,
        period: float,
        *,
        name: str = "api",
        time_func: Callable[[], float] = time.monotonic,
        sleep_func: Callable[[float], None] = time.sleep,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self.name = name
        self._time = time_func
        self._sleep = sleep_func
        self._calls: Deque[float] = deque()

    def acquire(self) -> float:
        """Block if needed, then record a call. Returns seconds actually slept."""
        slept_total = 0.0
        while True:
            now = self._time()
            while self._calls and (now - self._calls[0]) >= self.period:
                self._calls.popleft()
            if len(self._calls) < self.max_calls:
                self._calls.append(now)
                return slept_total
            wait = self.period - (now - self._calls[0])
            if wait > 0:
                logger.info("[%s] rate limit reached; sleeping %.2fs", self.name, wait)
                self._sleep(wait)
                slept_total += wait
