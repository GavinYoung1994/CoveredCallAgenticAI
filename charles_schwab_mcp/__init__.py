"""Charles Schwab OAuth + market-data MCP package.

  auth.py            one-time interactive OAuth to mint the FIRST token set
  token_manager.py   loads schwab_tokens.json and silently auto-refreshes it
  refresh_token.py   CLI to test/force a token refresh
  schwab_mcp_server.py  optional FastMCP server exposing the market-data tools

The LangGraph app reaches Schwab via app/data/schwab_client.py, which calls
token_manager.get_valid_access_token().
"""
