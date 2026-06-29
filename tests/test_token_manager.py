"""Offline tests for the Schwab token_manager (no network).

We point TOKEN_FILE at a temp file and stub requests.post so the refresh path is
exercised deterministically.
"""

import base64
import json
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "charles_schwab_mcp"))

import token_manager as tm


class _FakeResp:
    def __init__(self, status, data):
        self.status_code = status
        self._data = data
        self.text = json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def _setup(tmp_path):
    tm.TOKEN_FILE = str(tmp_path)
    os.environ["SCHWAB_APP_KEY"] = "key123"
    os.environ["SCHWAB_APP_SECRET"] = "secret456"
    # Reset the in-memory token cache so tests don't leak state into each other.
    tm._token_cache.update(access_token=None, expiration_timestamp=0.0)


def test_get_credentials_base64():
    os.environ["SCHWAB_APP_KEY"] = "key123"
    os.environ["SCHWAB_APP_SECRET"] = "secret456"
    expected = base64.b64encode(b"key123:secret456").decode()
    assert tm.get_credentials() == expected


def test_valid_token_returned_without_refresh():
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        _setup(path)
        # Token valid for another hour → no refresh.
        tm.save_tokens({"access_token": "LIVE", "refresh_token": "R", "expires_in": 3600})
        called = {"n": 0}
        tm.requests.post = lambda *a, **k: called.__setitem__("n", called["n"] + 1)
        assert tm.get_valid_access_token() == "LIVE"
        assert called["n"] == 0
    finally:
        os.path.exists(path) and os.unlink(path)


def test_expired_token_triggers_refresh():
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        _setup(path)
        # Write an already-expired token set.
        tm.save_tokens({"access_token": "OLD", "refresh_token": "R1", "expires_in": -100})
        tm.requests.post = lambda *a, **k: _FakeResp(
            200, {"access_token": "NEW", "refresh_token": "R2", "expires_in": 1800})
        assert tm.get_valid_access_token() == "NEW"
        # File now holds the refreshed token.
        saved = json.loads(Path(path).read_text())
        assert saved["access_token"] == "NEW" and saved["refresh_token"] == "R2"
        assert saved["expiration_timestamp"] > time.time()
    finally:
        os.path.exists(path) and os.unlink(path)


def test_cached_token_avoids_disk_reread():
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        _setup(path)
        tm.save_tokens({"access_token": "LIVE", "refresh_token": "R", "expires_in": 3600})
        # Count disk reads via load_tokens.
        reads = {"n": 0}
        orig = tm.load_tokens
        def counting_load():
            reads["n"] += 1
            return orig()
        tm.load_tokens = counting_load
        try:
            first = tm.get_valid_access_token()   # reads disk once → caches
            second = tm.get_valid_access_token()   # served from cache
            third = tm.get_valid_access_token()
        finally:
            tm.load_tokens = orig
        assert first == second == third == "LIVE"
        assert reads["n"] == 1                      # only the first call hit disk
    finally:
        os.path.exists(path) and os.unlink(path)


def test_refresh_failure_raises():
    fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
    try:
        _setup(path)
        tm.save_tokens({"access_token": "OLD", "refresh_token": "BAD", "expires_in": -100})
        tm.requests.post = lambda *a, **k: _FakeResp(400, {"error": "invalid_grant"})
        try:
            tm.get_valid_access_token()
            assert False, "should have raised"
        except Exception:
            pass
    finally:
        os.path.exists(path) and os.unlink(path)


def test_missing_token_file_raises():
    tm.TOKEN_FILE = "/nonexistent/dir/schwab_tokens.json"
    try:
        tm.load_tokens()
        assert False, "should have raised"
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    import requests  # ensure real module importable too
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for t in tests:
        try:
            t(); print(f"  ✅ {t.__name__}"); passed += 1
        except AssertionError as exc:
            print(f"  ❌ {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            print(f"  💥 {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{passed}/{len(tests)} tests passed.")
    sys.exit(0 if passed == len(tests) else 1)
