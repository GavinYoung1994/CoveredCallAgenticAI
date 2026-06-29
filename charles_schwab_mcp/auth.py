import os
import base64
import requests
import webbrowser
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

from token_manager import save_tokens  # same directory; persists schwab_tokens.json

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
    if "code=" not in returned_url or "%40" not in returned_url:
        raise ValueError(
            "Pasted URL does not look like a Schwab redirect (missing 'code=...%40'). "
            "Copy the FULL address bar URL after you approve access."
        )
    response_code = f"{returned_url[returned_url.index('code=') + 5: returned_url.index('%40')]}@"

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
        raise RuntimeError(f"Token exchange failed (HTTP {resp.status_code}): {tokens}")
    return tokens


def main():
    app_key, app_secret, cs_auth_url = construct_init_auth_url()
    webbrowser.open(cs_auth_url)

    logger.info("Paste the FULL redirect URL from your browser's address bar:")
    returned_url = input().strip()

    headers, payload = construct_headers_and_payload(returned_url, app_key, app_secret)
    tokens = retrieve_tokens(headers=headers, payload=payload)

    # Persist them (adds the buffered expiration_timestamp) so token_manager can
    # use + auto-refresh them. THIS is the step the original flow was missing.
    save_tokens(tokens)
    logger.info("✅ Saved tokens to schwab_tokens.json. The agent can now call the API.")
    return "Done!"


if __name__ == "__main__":
    main()