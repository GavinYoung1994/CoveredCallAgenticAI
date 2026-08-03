import os
import time
import base64
import requests
import webbrowser
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from loguru import logger
from dotenv import load_dotenv

from token_manager import save_tokens  # same directory; persists schwab_tokens.json

# Schwab authorization codes are single-use and expire fast (~30s). We warn the
# user if they take longer than this between opening the auth URL and pasting.
_CODE_TTL_SECONDS = 30

# Load credentials from the project-root .env so nothing is hardcoded here.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def construct_init_auth_url() -> tuple[str, str, str]:

    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")
    redirect_uri = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
    if not app_key or not app_secret:
        raise ValueError("Set SCHWAB_APP_KEY and SCHWAB_APP_SECRET in your .env file.")

    auth_url = f"https://api.schwabapi.com/v1/oauth/authorize?client_id={app_key}&redirect_uri={redirect_uri}"

    logger.info("Click to authenticate:")
    logger.info(auth_url)

    return app_key, app_secret, auth_url


def construct_headers_and_payload(returned_url, app_key, app_secret):
    redirect_uri = os.getenv("SCHWAB_REDIRECT_URI", "https://127.0.0.1")
    # Properly parse + URL-decode the `code` query param (parse_qs handles %40→@
    # and every other percent-encoding, unlike manual string slicing).
    params = parse_qs(urlparse(returned_url).query)
    codes = params.get("code")
    if not codes or not codes[0]:
        raise ValueError(
            "Pasted URL has no 'code=' parameter. Copy the FULL address bar URL "
            "(https://127.0.0.1/?code=...) right after you approve access."
        )
    response_code = codes[0]
    logger.info("Extracted authorization code ({} chars, ends with {!r}).",
                len(response_code), response_code[-1])

    credentials = f"{app_key}:{app_secret}"
    base64_credentials = base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

    headers = {
        "Authorization": f"Basic {base64_credentials}",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    payload = {
        "grant_type": "authorization_code",
        "code": response_code,
        "redirect_uri": redirect_uri,  # MUST match the app's registered redirect
    }
    return headers, payload


def retrieve_tokens(headers, payload) -> dict:
    resp = requests.post(
        url="https://api.schwabapi.com/v1/oauth/token", headers=headers, data=payload)
    tokens = resp.json()
    if resp.status_code != 200 or "access_token" not in tokens:
        body = str(tokens)
        if "invalid_grant" in body:
            logger.error(
                "Schwab rejected the authorization code (invalid_grant). This is NOT a bug in "
                "this script — the code was refused. Most likely, in order:\n"
                "  1. The code EXPIRED (Schwab codes last ~30s) — re-run and paste faster.\n"
                "  2. The code was ALREADY USED (single-use) — get a FRESH one by re-approving "
                "in the browser; never reuse a URL from a prior attempt.\n"
                "  3. Your Schwab app is NOT 'Ready For Use' yet (still pending) — check the "
                "developer dashboard.\n"
                f"  4. redirect_uri mismatch — the token request used {payload.get('redirect_uri')!r}; "
                "it must EXACTLY match the URI registered for your app.")
        raise RuntimeError(f"Token exchange failed (HTTP {resp.status_code}): {tokens}")
    return tokens


def main():
    app_key, app_secret, cs_auth_url = construct_init_auth_url()
    webbrowser.open(cs_auth_url)

    started = time.monotonic()
    logger.info("Paste the FULL redirect URL from your browser's address bar (within ~{}s):",
                _CODE_TTL_SECONDS)
    returned_url = input().strip()
    elapsed = time.monotonic() - started
    if elapsed > _CODE_TTL_SECONDS:
        logger.warning("You took {:.0f}s to paste — Schwab codes expire in ~{}s, so this may fail "
                       "with invalid_grant. If it does, re-run and paste faster.", elapsed, _CODE_TTL_SECONDS)

    headers, payload = construct_headers_and_payload(returned_url, app_key, app_secret)
    tokens = retrieve_tokens(headers=headers, payload=payload)

    # Persist them (adds the buffered expiration_timestamp) so token_manager can
    # use + auto-refresh them. THIS is the step the original flow was missing.
    save_tokens(tokens)
    logger.info("✅ Saved tokens to schwab_tokens.json. The agent can now call the API.")
    return "Done!"


if __name__ == "__main__":
    main()