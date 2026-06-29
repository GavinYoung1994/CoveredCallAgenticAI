import os
import json
import time
import base64
import requests
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv

# Load secrets from the project-root .env (works regardless of CWD).
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Keep the token cache next to this file, no matter where we're launched from.
TOKEN_FILE = str(Path(__file__).resolve().parent / "schwab_tokens.json")
AUTH_URL = "https://api.schwabapi.com/v1/oauth/token"

def get_credentials() -> str:
    """Safely retrieves and encodes app credentials from the environment."""
    app_key = os.getenv("SCHWAB_APP_KEY")
    app_secret = os.getenv("SCHWAB_APP_SECRET")

    if not app_key or not app_secret:
        raise ValueError("Missing SCHWAB_APP_KEY or SCHWAB_APP_SECRET environment variables.")

    credentials = f"{app_key}:{app_secret}"
    return base64.b64encode(credentials.encode("utf-8")).decode("utf-8")

def load_tokens() -> dict:
    """Loads the token state from the local JSON file."""
    if not os.path.exists(TOKEN_FILE):
        raise FileNotFoundError(f"{TOKEN_FILE} not found. You must run the manual initial auth first.")
    
    with open(TOKEN_FILE, "r") as f:
        return json.load(f)

def save_tokens(token_data: dict):
    """Saves the fresh tokens and calculates the exact expiration timestamp."""
    # Schwab access tokens expire in 1800 seconds (30 mins). We buffer by 60 seconds.
    expires_in = token_data.get("expires_in", 1800)
    token_data["expiration_timestamp"] = time.time() + expires_in - 60
    
    with open(TOKEN_FILE, "w") as f:
        json.dump(token_data, f, indent=4)
    logger.info("New tokens securely saved to local storage.")

def refresh_access_token(refresh_token: str) -> dict:
    """Executes the autonomous refresh payload.""" #
    logger.info("Access token expired or missing. Agent is fetching a new one...")
    
    headers = {
        "Authorization": f"Basic {get_credentials()}", #
        "Content-Type": "application/x-www-form-urlencoded", #
    }
    
    payload = {
        "grant_type": "refresh_token", #
        "refresh_token": refresh_token, #
    }
    
    response = requests.post(AUTH_URL, headers=headers, data=payload) #
    
    if response.status_code != 200:
        logger.error(f"Agent failed to refresh token: {response.text}") #
        response.raise_for_status()
        
    new_tokens = response.json() #
    
    # Schwab's refresh response usually contains a new access token and a new refresh token.
    # We must save both to keep the 7-day rolling window alive.
    save_tokens(new_tokens)
    return new_tokens

# In-memory cache so we don't re-read the token file (and log) on EVERY API call.
# A screening run makes hundreds of Schwab requests; without this, the disk read
# and a DEBUG line fire on each one, which both wastes I/O and looks like a hang.
_token_cache = {"access_token": None, "expiration_timestamp": 0.0}


def get_valid_access_token() -> str:
    """
    The main public accessor. Returns a live access token, refreshing silently
    when needed. Cached in memory between calls within a process.
    """
    current_time = time.time()

    # 1) Fast path: in-memory cached token still valid → no disk, no log.
    if _token_cache["access_token"] and current_time < _token_cache["expiration_timestamp"]:
        return _token_cache["access_token"]

    # 2) Load from disk; use it if still within the buffered expiry.
    tokens = load_tokens()
    if "access_token" in tokens and current_time < tokens.get("expiration_timestamp", 0):
        _token_cache["access_token"] = tokens["access_token"]
        _token_cache["expiration_timestamp"] = tokens["expiration_timestamp"]
        logger.debug("Loaded a valid access token from disk into cache.")
        return tokens["access_token"]

    # 3) Expired/missing → refresh.
    logger.info("Access token requires an update.")
    if "refresh_token" not in tokens:
        raise KeyError("No refresh_token found in local storage. Re-run auth.py.")

    new_tokens = refresh_access_token(tokens["refresh_token"])
    _token_cache["access_token"] = new_tokens["access_token"]
    _token_cache["expiration_timestamp"] = new_tokens.get(
        "expiration_timestamp", current_time + 1740)
    return new_tokens["access_token"]