"""CLI to verify / refresh the Schwab token set.

Run:  ./venv/bin/python charles_schwab_mcp/refresh_token.py

Delegates to token_manager (single source of truth) instead of duplicating the
refresh logic. Prints a clear, actionable message if the refresh token has
expired (Schwab refresh tokens last 7 days) and you must re-run auth.py.
"""

import sys
from pathlib import Path

from loguru import logger

# Allow running as a plain script from the project root or this directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from token_manager import get_valid_access_token, load_tokens, refresh_access_token


def main() -> int:
    try:
        tokens = load_tokens()
    except FileNotFoundError:
        logger.error("No schwab_tokens.json yet. Run `python charles_schwab_mcp/auth.py` first.")
        return 1

    if "refresh_token" not in tokens:
        logger.error("No refresh_token stored. Re-run `python charles_schwab_mcp/auth.py`.")
        return 1

    try:
        # Force a refresh to confirm credentials + refresh token are still valid.
        refresh_access_token(tokens["refresh_token"])
        access = get_valid_access_token()
        logger.info("✅ Token is valid. Access token starts with: {}...", access[:12])
        return 0
    except Exception as exc:  # noqa: BLE001
        logger.error("Token refresh failed: {}", exc)
        logger.error("Schwab refresh tokens expire after 7 days. Re-run "
                     "`python charles_schwab_mcp/auth.py` to mint a fresh set.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
